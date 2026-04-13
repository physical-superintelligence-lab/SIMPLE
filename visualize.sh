#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path-to-usd-file>"
    exit 1
fi

args=(
    --usd-path "$1" \
    --duration 2.0 \
    --headless \
    --fps 30 \
    --output-path ./videos/mo_gen.mp4 \
    --playback-speed 1.0 \
    # --camera-distance 0.1 \
    # --camera-target "0, 0.8, 0.7" \
    # --show-bbox \
    # --auto-frame \
    # --camera-position "0.5, 0.5, 0.3" \
    # --camera-angle "40, 30" \
)

CUDA_VISIBLE_DEVICES="3" python scripts/visualization.py "${args[@]}"


# Common Camera Angles

# Top-Down View (Bird's Eye)
# --camera_distance 1.0 --camera_angle "0,90" --camera_target "0,0,0"

# 45-Degree Isometric (Standard)
# --camera_distance 1.0 --camera_angle "45,30" --camera_target "0,0,0.1"

# Front View
# --camera_distance 1.0 --camera_angle "0,0" --camera_target "0,0,0.1"

# Side View
# --camera_distance 1.0 --camera_angle "90,0" --camera_target "0,0,0.1"

# Close-Up Detail
# --camera_distance 0.3 --camera_angle "30,15" --camera_target "0,0,0.15"
