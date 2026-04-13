"""
TCP control server (stdlib socket).

Listens for OPEN_CAMERA / CLOSE_CAMERA commands from the Pico VR client.

Wire protocol (observed from Pico):
    [4 bytes BE uint32: body length] [NetworkDataProtocol body]

NetworkDataProtocol body (little-endian):
    [4 bytes LE int32: cmd_len] [command string]
    [4 bytes LE int32: data_len] [data bytes]

Accepts one client at a time; on disconnect waits for the next connection.
Runs in a background daemon thread.
"""

import socket
import struct
import threading
from typing import Callable

from simple.teleop.pico import protocol


class TCPControlServer:
    """TCP server that dispatches OPEN_CAMERA / CLOSE_CAMERA commands.

    Args:
        address: ``"host:port"`` string, e.g. ``"0.0.0.0:13579"``.

    Callbacks (assign before calling start()):
        on_open_camera(camera_request: dict) -> None
        on_close_camera()                    -> None
    """

    def __init__(self, address: str) -> None:
        host, _, port_str = address.rpartition(":")
        self._host = host or "0.0.0.0"
        self._port = int(port_str)

        self.on_open_camera: Callable[[dict], None] | None = None
        self.on_close_camera: Callable[[], None] | None = None

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._current_conn: socket.socket | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the background listener thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="TCPControlServer", daemon=True
        )
        self._thread.start()
        print(f"[TCPServer] Listening on {self._host}:{self._port}")

    def stop(self) -> None:
        """Signal the listener thread to stop and wait for it."""
        self._stop_event.set()
        self.close_client()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    def close_client(self) -> None:
        """Forcefully close the current client connection.

        Call this when streaming ends so the server stops waiting and is
        ready to accept the next connection immediately.
        """
        conn = self._current_conn
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
            self._current_conn = None
            print("[TCPServer] Client connection closed by server")

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self._host, self._port))
        server_sock.listen(1)
        server_sock.settimeout(1.0)

        print(f"[TCPServer] Ready for connections on {self._host}:{self._port}")

        while not self._stop_event.is_set():
            try:
                conn, addr = server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            print(f"[TCPServer] Client connected from {addr}")
            self._handle_client(conn)
            print("[TCPServer] Client disconnected, waiting for next connection")

        server_sock.close()
        print("[TCPServer] Listener stopped")

    def _handle_client(self, conn: socket.socket) -> None:
        self._current_conn = conn
        conn.settimeout(1.0)
        try:
            while not self._stop_event.is_set():
                body = self._read_framed(conn)
                if body is None:
                    break

                try:
                    command, data = protocol.parse_message(body)
                except ValueError as exc:
                    print(f"[TCPServer] Protocol parse error: {exc}")
                    continue

                print(f"[TCPServer] Command received: '{command}'")

                if command == "OPEN_CAMERA":
                    try:
                        camera_req = protocol.parse_camera_request(data)
                    except ValueError as exc:
                        print(f"[TCPServer] CameraRequest parse error: {exc}")
                        camera_req = {}
                    if self.on_open_camera:
                        self.on_open_camera(camera_req)
                    self._send_ack(conn, "OPEN_CAMERA")

                elif command == "CLOSE_CAMERA":
                    self._send_ack(conn, "CLOSE_CAMERA")
                    if self.on_close_camera:
                        self.on_close_camera()

                else:
                    print(f"[TCPServer] Unknown command: '{command}'")
        finally:
            self._current_conn = None
            conn.close()

    def _send_ack(self, conn: socket.socket, command: str) -> None:
        """Send an ACK using the same framing as incoming messages."""
        try:
            cmd_b = command.encode("utf-8")
            body  = struct.pack("<I", len(cmd_b)) + cmd_b + struct.pack("<I", 0)
            conn.sendall(struct.pack(">I", len(body)) + body)
            print(f"[TCPServer] ACK sent for '{command}'")
        except OSError as exc:
            print(f"[TCPServer] Failed to send ACK: {exc}")

    def _read_framed(self, conn: socket.socket) -> bytes | None:
        """Read one [4 BE: len][body] message; return None on disconnect."""
        header = self._recv_exactly(conn, 4)
        if header is None:
            return None
        (body_len,) = struct.unpack(">I", header)
        return self._recv_exactly(conn, body_len)

    def _recv_exactly(self, conn: socket.socket, n: int) -> bytes | None:
        """Read exactly n bytes; return None on disconnect or stop."""
        buf = b""
        while len(buf) < n:
            if self._stop_event.is_set():
                return None
            try:
                chunk = conn.recv(n - len(buf))
            except socket.timeout:
                continue
            except OSError:
                return None
            if not chunk:
                return None
            buf += chunk
        return buf
