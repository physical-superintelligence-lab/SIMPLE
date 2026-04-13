"""
Binary protocol: deserialise TCP control messages from the Pico VR client.

TCP envelope (big-endian outer framing):
    [4 bytes BE uint32: body length] [body bytes]

Body = NetworkDataProtocol (little-endian):
    [4 bytes LE int32: command length] [command string]
    [4 bytes LE int32: data length]   [data bytes]

CameraRequestData (data field):
    [0xCA 0xFE]                          magic bytes
    [0x01]                               version
    [4 bytes LE int32: width]
    [4 bytes LE int32: height]
    [4 bytes LE int32: fps]
    [4 bytes LE int32: bitrate]
    [4 bytes LE int32: enableMvHevc]
    [4 bytes LE int32: renderMode]
    [4 bytes LE int32: port]
    [1 byte length + bytes: camera]      e.g. "ZED"
    [1 byte length + bytes: ip]          e.g. "192.168.1.45"
"""

import struct


def parse_message(body: bytes) -> tuple[str, bytes]:
    """Deserialise a NetworkDataProtocol body.

    Returns:
        (command, data) tuple.
    Raises:
        ValueError on malformed input.
    """
    if len(body) < 8:
        raise ValueError(f"Body too small: {len(body)} bytes")

    offset = 0

    (cmd_len,) = struct.unpack_from("<i", body, offset)
    offset += 4
    if cmd_len < 0 or offset + cmd_len > len(body):
        raise ValueError(f"Invalid command length: {cmd_len}")

    command = body[offset : offset + cmd_len].decode("utf-8").rstrip("\x00")
    offset += cmd_len

    if offset + 4 > len(body):
        raise ValueError("Buffer too small for data length field")

    (data_len,) = struct.unpack_from("<i", body, offset)
    offset += 4
    if data_len < 0 or offset + data_len > len(body):
        raise ValueError(f"Invalid data length: {data_len}")

    return command, body[offset : offset + data_len]


def parse_camera_request(data: bytes) -> dict:
    """Deserialise a CameraRequestData payload.

    Returns:
        dict with keys: width, height, fps, bitrate, enableMvHevc,
                        renderMode, port, camera, ip
    Raises:
        ValueError on malformed input.
    """
    if len(data) < 10:
        raise ValueError(f"CameraRequestData too small: {len(data)} bytes")

    offset = 0

    if data[offset] != 0xCA or data[offset + 1] != 0xFE:
        raise ValueError(f"Invalid magic bytes: 0x{data[offset]:02X} 0x{data[offset+1]:02X}")
    offset += 2

    version = data[offset]
    if version != 1:
        raise ValueError(f"Unsupported protocol version: {version}")
    offset += 1

    if offset + 28 > len(data):
        raise ValueError("Not enough data for integer fields")

    width, height, fps, bitrate, enable_mv_hevc, render_mode, port = (
        struct.unpack_from("<iiiiiii", data, offset)
    )
    offset += 28

    def _read_str(buf, pos):
        if pos >= len(buf):
            raise ValueError("Not enough data to read string length")
        length = buf[pos]
        pos += 1
        if pos + length > len(buf):
            raise ValueError("Not enough data to read string content")
        return buf[pos : pos + length].decode("utf-8"), pos + length

    camera, offset = _read_str(data, offset)
    ip, offset     = _read_str(data, offset)

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "bitrate": bitrate,
        "enableMvHevc": enable_mv_hevc,
        "renderMode": render_mode,
        "port": port,
        "camera": camera,
        "ip": ip,
    }
