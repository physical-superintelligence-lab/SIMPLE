import zmq
import mujoco
import numpy as np

from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge
from simple.robots.g1_sonic import G1Sonic
from simple.core.action import ActionCmd
from simple.teleop.pico.tcp_server import TCPControlServer
from simple.teleop.pico.tcp_video_sender import TCPVideoSender
from simple.teleop.pico.streaming import FrameBuffer, StreamingThread
from .sonic_wbc_agent import SonicWbcAgent

class PicoSonicAgent(SonicWbcAgent):

    def __init__(self, robot: G1Sonic):
        super().__init__(robot)

        self.image_publish_process = None

        self.init_unitree_bridge()
        self.init_subscriber()
        # self.init_publisher()
        self.init_pico_streamer()

        # Controlled drop state: lowered via elastic_band.length at _drop_rate m/s
        # until the robot settles on the ground, then the band is disabled.
        self._dropping = False
        self._drop_rate = 0.15  # m/s — ~1.3s to descend 20 cm

        # Set by _poll_ctrl() when a reset_env command arrives; checked by the
        # teleop main loop so env.reset() is called outside of step().
        self._reset_requested = False
        self.sim_dt = self.robot.sonic_config["SIMULATE_DT"]

    def init_unitree_bridge(self):
        self.unitree_bridge = UnitreeSdk2Bridge(self.robot.sonic_config)
        if self.robot.sonic_config["USE_JOYSTICK"]:
            self.unitree_bridge.SetupJoystick(
                device_id=self.robot.sonic_config["JOYSTICK_DEVICE"], js_type=self.robot.sonic_config["JOYSTICK_TYPE"]
            )

    def init_publisher(self):
        pass
    
    def init_subscriber(self):
        from gear_sonic.utils.teleop.zmq.zmq_poller import ZMQPoller
        pico_host = self.robot.sonic_config.get("PICO_HOST", "localhost")
        pico_port = self.robot.sonic_config.get("PICO_PORT", 5556)
        self.pico_subscriber = ZMQPoller(host=pico_host, port=pico_port, topic="pose")

        # Separate lightweight socket for sim-control commands ("ctrl" topic).
        # Uses a plain zmq socket (not ZMQPoller) to avoid per-poll print spam.
        self._ctrl_context = zmq.Context()
        self._ctrl_socket = self._ctrl_context.socket(zmq.SUB)
        self._ctrl_socket.setsockopt_string(zmq.SUBSCRIBE, "ctrl")
        self._ctrl_socket.setsockopt(zmq.CONFLATE, 1)
        self._ctrl_socket.connect(f"tcp://{pico_host}:{pico_port}")

    def init_pico_streamer(self):
        self._streaming: StreamingThread | None = None
        self._frame_buffer = FrameBuffer()

        tcp_server = TCPControlServer("0.0.0.0:13579") # TODO Config me

        def on_open_camera(camera_req):
            print(f"[Main] OPEN_CAMERA: {camera_req}")

            if self._streaming and self._streaming.is_running():
                print("[Main] Already streaming, ignoring duplicate OPEN_CAMERA")
                return

            fps     = camera_req.get("fps")     or 60
            width   = camera_req.get("width")   or 2560
            height  = camera_req.get("height")  or 720
            bitrate = camera_req.get("bitrate") or 4_000_000
            hevc    = bool(camera_req.get("enableMvHevc"))
            ip      = camera_req.get("ip")
            port    = camera_req.get("port")

            if not ip or not port:
                print("[Main] OPEN_CAMERA missing ip/port, cannot stream")
                return

            try:
                sender = TCPVideoSender(
                    ip=ip, port=port,
                    width=width, height=height, fps=fps,
                    bitrate=bitrate, hevc=hevc,
                )
            except ConnectionRefusedError:
                print(f"[Main] Connection refused to {ip}:{port} — receiver not ready")
                return

            self._streaming = StreamingThread(
                frame_buffer=self._frame_buffer,
                fps=fps,
                publishers=[sender],
                on_ended=lambda: tcp_server.close_client(),
            )
            self._streaming.start()

        def on_close_camera():
            print("[Main] CLOSE_CAMERA received")
            if self._streaming:
                self._streaming.stop()
                self._streaming = None
            tcp_server.close_client()

        tcp_server.on_open_camera = on_open_camera
        tcp_server.on_close_camera = on_close_camera
        tcp_server.start()

    def _poll_ctrl(self) -> dict | None:
        """Non-blocking poll of the 'ctrl' ZMQ topic.

        Returns a decoded field dict (e.g. ``{"drop_robot": array([True])}``)
        or ``None`` if no message is waiting.
        """
        if not self._ctrl_socket.poll(timeout=0):
            return None
        raw = self._ctrl_socket.recv(zmq.NOBLOCK)
        return self._unpack_pose_message(raw[len("ctrl"):])
    
    @staticmethod
    def _unpack_pose_message(data: bytes) -> dict:
        """Deserialize a packed pose message (topic prefix already stripped).

        Format: [1280-byte JSON header (null-padded)][concatenated binary fields]
        """
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import HEADER_SIZE

        header_bytes = data[:HEADER_SIZE]
        payload = data[HEADER_SIZE:]

        header = __import__("json").loads(header_bytes.rstrip(b"\x00").decode("utf-8"))

        dtype_map = {
            "f32": np.float32,
            "f64": np.float64,
            "i32": np.int32,
            "i64": np.int64,
            "bool": bool,
        }

        result = {}
        offset = 0
        for field in header["fields"]:
            dtype = dtype_map[field["dtype"]]
            shape = field["shape"]
            count = int(np.prod(shape)) if shape else 1
            nbytes = count * np.dtype(dtype).itemsize
            arr = np.frombuffer(payload[offset: offset + nbytes], dtype=dtype).reshape(shape)
            result[field["name"]] = arr
            offset += nbytes

        return result
    
    def publish_low_state(self, proprio):
        # boradcast through unitree bridge that teleop/policy can subscribe to 
        self.unitree_bridge.PublishLowState(proprio) # type:ignore
        if self.unitree_bridge.joystick: # type:ignore
            self.unitree_bridge.PublishWirelessController() # type:ignore

    def update_render_caches(self, observation:dict):
        if self.image_publish_process is not None:
            self.image_publish_process.update_shared_memory(observation)

        # Push a side-by-side stereo frame to the VR streaming buffer.
        # This must happen on the main thread (mjData already read above).
        if self._streaming and self._streaming.is_running():
            self._push_stereo_frame(observation)

        return observation

    def _push_stereo_frame(self, observation: dict) -> None:
        """Compose a side-by-side stereo BGR frame and hand it to the buffer."""

        left  = observation["head_stereo_left"]
        right = observation["head_stereo_right"]

        if left is None or right is None:
            return  # cameras not yet configured

        # MuJoCo renders RGB; TCPVideoSender expects BGR.
        left_bgr  = left[..., ::-1]
        right_bgr = right[..., ::-1]

        stereo = np.concatenate([left_bgr, right_bgr], axis=1)  # side-by-side
        self._frame_buffer.put(stereo)

    def get_action(self, observation, instruction=None, **kwargs):
        # Check for ctrl commands from the pico manager.
        ctrl = self._poll_ctrl()
        if ctrl is not None:
            if (
                bool(ctrl.get("drop_robot", [False])[0])
                and self.robot.elastic_band
                and self.robot.elastic_band.enable
                and not self._dropping
            ):
                self._dropping = True
                print("[SonicLocoManipEnv] Controlled drop started — lowering robot to ground")

            if bool(ctrl.get("reset_env", [False])[0]):
                self._reset_requested = True
                print("[SonicLocoManipEnv] Environment reset requested")

        # Slowly lower the elastic band target while dropping.
        # The band remains active (kp/kd still cushion the descent) until the
        # robot has settled, then it is disabled cleanly.
        if self._dropping and self.robot.elastic_band and self.robot.elastic_band.enable:
            self.robot.elastic_band.length -= self._drop_rate * self.sim_dt
            # pelvis_vz = self.robot.pelvis_vz #self.mjData.qvel[2]
            # Settle condition: band target has moved down far enough AND
            # the pelvis is nearly stationary vertically.
            if self.robot.elastic_band.length <= -0.25 and abs(self.robot.pelvis_vz) < 0.05:
                self.robot.elastic_band.enable = False
                self._dropping = False
                pelvis_z = self.robot.pelvis_z # self.mjData.qpos[2]
                print(f"[SonicLocoManipEnv] Robot landed (pelvis Z={pelvis_z:.3f} m), elastic band disabled")

        if (self.robot.elastic_band 
            and self.robot.elastic_band.enable 
            and self.robot.use_floating_root_link
        ):
            return ActionCmd("elastic_band", 
                dropping=self._dropping,
                drop_rate=self._drop_rate
            )
        
        return ActionCmd("wbc_torque",
            low_cmd=self.unitree_bridge.low_cmd,
            use_sensor=self.unitree_bridge.use_sensor,
            left_hand_cmd=self.unitree_bridge.left_hand_cmd,
            right_hand_cmd=self.unitree_bridge.right_hand_cmd,
        )
    
    def close(self):
        if hasattr(self, "_ctrl_socket"):
            self._ctrl_socket.close()
        if hasattr(self, "_ctrl_context"):
            self._ctrl_context.term()