"""
Pico VR client emulator for local debugging.

Sends OPEN_CAMERA to the server, receives the H.264 TCP stream back,
decodes it with PyAV, and displays it with OpenCV.  Press 'q' to quit.

Usage
-----
  # loopback (server and this script on the same machine):
  python3 send_open_camera.py 127.0.0.1 127.0.0.1

  # remote server, stream back to this machine:
  python3 send_open_camera.py 192.168.1.10 192.168.1.20

Wire format (incoming H.264 stream from TCPVideoSender):
  [4B BE uint32: frame_size] [H.264 NAL data]
"""

import argparse
import queue
import socket
import struct
import threading
import time

import av
import cv2
import numpy as np


CONTROL_PORT  = 13579
VIDEO_PORT    = 12345
MAGIC         = bytes([0xCA, 0xFE])
PROTO_VERSION = 1


# ── protocol serialisers ──────────────────────────────────────────────────────

def _camera_request_bytes(
    width: int, height: int, fps: int, bitrate: int,
    enable_mv_hevc: int, render_mode: int, port: int,
    camera: str, ip: str,
) -> bytes:
    cam_b = camera.encode("utf-8")
    ip_b  = ip.encode("utf-8")
    return (
        MAGIC
        + struct.pack("<B", PROTO_VERSION)
        + struct.pack("<iiiiiii", width, height, fps, bitrate,
                      enable_mv_hevc, render_mode, port)
        + struct.pack("<B", len(cam_b)) + cam_b
        + struct.pack("<B", len(ip_b))  + ip_b
    )


def _frame(command: str, data: bytes = b"") -> bytes:
    cmd_b = command.encode("utf-8")
    body = (
        struct.pack("<i", len(cmd_b)) + cmd_b
        + struct.pack("<i", len(data)) + data
    )
    return struct.pack(">I", len(body)) + body


def _send(sock: socket.socket, command: str, data: bytes = b"") -> None:
    sock.sendall(_frame(command, data))
    print(f"[Client] Sent '{command}'")


# ── H.264 TCP receiver thread ─────────────────────────────────────────────────

def _recv_exactly(conn: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        try:
            chunk = conn.recv(n - len(buf))
        except OSError:
            return None
        if not chunk:
            return None
        buf += chunk
    return buf


def _video_receiver(stream_port: int, frame_q: queue.Queue, stop: threading.Event) -> None:
    """Listen for the server's TCP H.264 connection, decode, push BGR frames to queue."""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", stream_port))
    server_sock.listen(1)
    server_sock.settimeout(1.0)
    print(f"[Receiver] Listening for H.264 stream on port {stream_port} ...")

    conn = None
    while not stop.is_set():
        try:
            conn, addr = server_sock.accept()
            break
        except socket.timeout:
            continue

    server_sock.close()
    if conn is None:
        return

    print(f"[Receiver] Video connection from {addr}")
    codec = av.CodecContext.create("h264", "r")

    try:
        while not stop.is_set():
            # Read 4-byte BE length header
            header = _recv_exactly(conn, 4)
            if header is None:
                break
            (size,) = struct.unpack(">I", header)

            # Read H.264 NAL data
            nal = _recv_exactly(conn, size)
            if nal is None:
                break

            # Decode
            try:
                packet = av.Packet(nal)
                for av_frame in codec.decode(packet):
                    bgr = av_frame.to_ndarray(format="bgr24")
                    # Drop oldest frame if queue is full to avoid lag
                    if frame_q.full():
                        try:
                            frame_q.get_nowait()
                        except queue.Empty:
                            pass
                    frame_q.put_nowait(bgr)
            except Exception as exc:
                print(f"[Receiver] Decode error: {exc}")
    finally:
        conn.close()
        print("[Receiver] Video connection closed")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pico VR client emulator — sends OPEN_CAMERA and displays H.264 stream"
    )
    parser.add_argument("server_ip",  help="IP of the video sender server")
    parser.add_argument("headset_ip", help="IP this machine's video listener is reachable on")
    parser.add_argument("--server-port",  type=int, default=CONTROL_PORT)
    parser.add_argument("--stream-port",  type=int, default=VIDEO_PORT)
    parser.add_argument("--width",        type=int, default=2560)
    parser.add_argument("--height",       type=int, default=720)
    parser.add_argument("--fps",          type=int, default=60)
    parser.add_argument("--bitrate",      type=int, default=4_000_000)
    parser.add_argument("--hevc",         action="store_true")
    parser.add_argument("--render-mode",  type=int, default=2)
    parser.add_argument("--camera",       default="ZED")
    args = parser.parse_args()

    stop = threading.Event()
    frame_q: queue.Queue = queue.Queue(maxsize=4)

    # Start video receiver thread BEFORE sending OPEN_CAMERA so the port is
    # ready when the server tries to connect back.
    recv_thread = threading.Thread(
        target=_video_receiver,
        args=(args.stream_port, frame_q, stop),
        daemon=True,
    )
    recv_thread.start()
    time.sleep(0.1)  # give the socket a moment to bind

    # Connect to server and send OPEN_CAMERA
    print(f"[Client] Connecting to {args.server_ip}:{args.server_port} ...")
    ctrl_sock = socket.create_connection((args.server_ip, args.server_port), timeout=5)
    print("[Client] Connected.")

    cam_data = _camera_request_bytes(
        width=args.width, height=args.height, fps=args.fps,
        bitrate=args.bitrate, enable_mv_hevc=int(args.hevc),
        render_mode=args.render_mode, port=args.stream_port,
        camera=args.camera, ip=args.headset_ip,
    )
    _send(ctrl_sock, "OPEN_CAMERA", cam_data)
    print(f"[Client] Streaming {args.width}x{args.height}@{args.fps}fps "
          f"→ {args.headset_ip}:{args.stream_port}  |  Press 'q' to quit")

    # Display loop (must run on main thread for OpenCV)
    frame_count = 0
    fps_time = time.time()
    fps_display = 0.0

    try:
        while True:
            try:
                bgr = frame_q.get(timeout=0.1)
            except queue.Empty:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            frame_count += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps_display = frame_count / (now - fps_time)
                frame_count = 0
                fps_time = now

            display = cv2.resize(bgr, (1280, 360))
            cv2.putText(display, f"{bgr.shape[1]}x{bgr.shape[0]}  {fps_display:.1f} FPS",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("XRoboToolkit Stream", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        cv2.destroyAllWindows()
        _send(ctrl_sock, "CLOSE_CAMERA")
        ctrl_sock.close()
        recv_thread.join(timeout=3)
        print("[Client] Disconnected.")


if __name__ == "__main__":
    main()
