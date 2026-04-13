import time
import threading

import numpy as np


class FrameBuffer:
    """Thread-safe single-slot frame buffer (latest-frame-wins).

    The main sim thread calls ``put()`` after rendering; the streaming
    thread calls ``get()`` to retrieve the most recent frame.

    Because MuJoCo's ``renderer.render()`` already returns an independent
    numpy array, no deep-copy is needed — the buffer only needs a lock
    for the pointer swap.
    """

    def __init__(self) -> None:
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._event = threading.Event()

    def put(self, frame: np.ndarray) -> None:
        """Store the latest frame (main thread)."""
        with self._lock:
            self._frame = frame
        self._event.set()

    def get(self, timeout: float = 0.1) -> np.ndarray | None:
        """Wait up to *timeout* seconds for a new frame (streaming thread).

        Returns ``None`` if no frame arrived within the timeout.
        """
        if not self._event.wait(timeout):
            return None
        with self._lock:
            self._event.clear()
            return self._frame


class StreamingThread:
    """Reads frames from a ``FrameBuffer`` and forwards them to publisher(s).

    The buffer is written by the main sim thread (after MuJoCo rendering)
    so this thread never touches ``mjData`` — making the design thread-safe.

    Args:
        frame_buffer: Shared ``FrameBuffer`` populated by the main thread.
        fps:          Target publish rate (used for get() timeout only).
        publishers:   Objects with a ``publish_raw_frame(frame)`` method.
        on_ended:     Optional callback invoked when the loop exits.
    """

    def __init__(self, frame_buffer: FrameBuffer, fps: int, publishers, on_ended=None):
        self._frame_buffer = frame_buffer
        self._fps = fps
        self._publishers = publishers
        self._on_ended = on_ended

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            print("[Streaming] Already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="StreamingThread", daemon=True
        )
        self._thread.start()
        print("[Streaming] Started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        frame_timeout = 1.0 / self._fps
        print(f"[Streaming] Loop at {self._fps} FPS")

        try:
            while not self._stop_event.is_set():
                frame = self._frame_buffer.get(timeout=frame_timeout)
                if frame is None:
                    continue

                for pub in self._publishers:
                    pub.publish_raw_frame(frame)
        finally:
            for pub in self._publishers:
                pub.close()
            print("[Streaming] Stopped")
            if self._on_ended:
                self._on_ended()
