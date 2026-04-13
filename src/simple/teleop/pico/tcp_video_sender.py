"""
TCP video sender: connects to the Pico VR client and streams H.264 video.

Wire format (matches C++ on_new_sample TCP send):
    [4 bytes BE uint32: frame_size] [H.264 NAL data]

The Pico's MediaDecoder.startServer(port, false) listens for this connection.
"""

import socket
import struct
import threading

import av
import numpy as np


class TCPVideoSender:
    """Connect to a remote TCP server and push H.264 encoded frames.

    Args:
        ip:       Remote host IP (from OPEN_CAMERA camera_req['ip']).
        port:     Remote port   (from OPEN_CAMERA camera_req['port']).
        width:    Frame width in pixels (full side-by-side width).
        height:   Frame height in pixels.
        fps:      Target frame rate.
        bitrate:  H.264 bitrate in bps (default 4 Mbps).
        hevc:     Use H.265 instead of H.264 (default False).
    """

    def __init__(
        self,
        ip: str,
        port: int,
        width: int,
        height: int,
        fps: int,
        bitrate: int = 4_000_000,
        hevc: bool = False,
    ) -> None:
        self._ip = ip
        self._port = port
        self._width = width
        self._height = height
        self._fps = fps
        self._bitrate = bitrate
        self._codec_name = "libx265" if hevc else "libx264"

        self._sock: socket.socket | None = None
        self._encoder: av.CodecContext | None = None
        self._lock = threading.Lock()
        self._closed = False

        self._connect()
        self._init_encoder()

    # ------------------------------------------------------------------
    # Public interface (matches ZMQPublisher / ZMQPushSender)
    # ------------------------------------------------------------------

    def publish_raw_frame(self, frame: np.ndarray) -> None:
        """Encode *frame* as H.264 and send over TCP.

        Args:
            frame: uint8 BGR numpy array, shape (height, width, 3).
        """
        if self._closed or self._sock is None or self._encoder is None:
            return

        # Convert BGR → YUV420p for H.264 encoder
        bgr_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        bgr_frame.pts = None  # let encoder assign timestamps

        packets = self._encoder.encode(bgr_frame)
        self._send_packets(packets)

    def close(self) -> None:
        """Flush encoder and close TCP connection."""
        if self._closed:
            return
        self._closed = True

        if self._encoder:
            try:
                self._send_packets(self._encoder.encode(None))  # flush
            except Exception:
                pass

        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

        print(f"[TCPVideoSender] Closed connection to {self._ip}:{self._port}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.connect((self._ip, self._port))
        print(f"[TCPVideoSender] Connected to {self._ip}:{self._port}")

    def _init_encoder(self) -> None:
        self._encoder = av.CodecContext.create(self._codec_name, "w")
        self._encoder.width = self._width
        self._encoder.height = self._height
        self._encoder.framerate = self._fps
        self._encoder.bit_rate = self._bitrate
        self._encoder.pix_fmt = "yuv420p"
        self._encoder.options = {
            "tune": "zerolatency",
            "preset": "ultrafast",
            "x264-params": "keyint=15:min-keyint=15",
        }
        self._encoder.open()
        print(
            f"[TCPVideoSender] Encoder ready: {self._codec_name} "
            f"{self._width}x{self._height}@{self._fps}fps "
            f"{self._bitrate // 1000}kbps"
        )

    def _send_packets(self, packets) -> None:
        for packet in packets:
            data = bytes(packet)
            if not data:
                continue
            try:
                header = struct.pack(">I", len(data))
                with self._lock:
                    if self._sock:
                        self._sock.sendall(header + data)
            except OSError as exc:
                print(f"[TCPVideoSender] Send error: {exc}")
                self._closed = True
                break
