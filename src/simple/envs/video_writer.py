"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import os
import cv2
import shutil
import numpy as np
from PIL import Image

from simple.utils import is_ffmpeg_installed


class VideoWriter:
    def __init__(self, filename, 
                 framerate, 
                 resolution, # (w,h)
                 write_png = False):

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        if os.path.exists(filename):
            print("remove existing file:", filename)
            os.unlink(filename)

        self.resize = False
        resolution = (resolution[0]//2, resolution[1]//2) if self.resize else resolution
        # resolution = [H, W]
        self.video_writer = cv2.VideoWriter(
            filename, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            framerate, 
            resolution
        )

        self.frame_idx = 0
        self.filename = filename
        self.write_png = write_png

        self.is_ffmpeg_installed = is_ffmpeg_installed()
        if not self.is_ffmpeg_installed:
            import warnings
            warnings.warn("Warning: FFmpeg is not installed. Video files may not be playable inside VSCode.", stacklevel=2)

    def write(self, image):
        assert image.dtype == np.uint8
        if self.resize:
            h, w = image.shape[:2]
            image = cv2.resize(image, (w//2, h//2))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.video_writer.write(image)
        if self.write_png:
            # Image.fromarray(image).save(f"{self.filename}_{self.frame_idx:03d}.png")
            cv2.imwrite(f"{self.filename}_{self.frame_idx:03d}.png", image)
        self.frame_idx += 1

    def release(self, success = True):
        self.video_writer.release()
        suffix = "success" if success else "failed"

        newfilename = f"{self.filename[:-4]}_{suffix}.mp4"
        if os.path.exists(newfilename):
            print(f"remove existing file: {newfilename}")
            os.remove(newfilename)

        if self.is_ffmpeg_installed:
            os.system(f"ffmpeg -i {self.filename} -vcodec libx264 {newfilename} > /dev/null 2>&1")
            os.system(f"rm {self.filename}") 