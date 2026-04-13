#!/bin/bash

# Script to generate videos from PNG sequences
# Usage: ./generate_video.sh <camera_type>

BASE_INPUT_DIR="data/datagen.g1_inspire_wholebody_locomotion/simple/G1InspireWholebodyLocomotion-v0/level-0/images"
OUTPUT_DIR="./videos"

# Check if camera type argument provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <camera_type>"
    echo "Camera types: mujoco, front_stereo_left, front_stereo_right, side_left, wrist, wrist_left, head_stereo_left, head_stereo_right, or all"
    exit 1
fi

CAMERA_TYPE=$1

# Function to generate video for a single camera
generate_video() {
    local cam_type=$1
    local folder_name=$2
    
    INPUT_DIR="$BASE_INPUT_DIR/$folder_name/episode_000000"
    OUTPUT_FILE="$OUTPUT_DIR/${cam_type}.mp4"
    
    # Check if input directory exists
    if [ ! -d "$INPUT_DIR" ]; then
        echo "Warning: Input directory '$INPUT_DIR' does not exist, skipping..."
        return 1
    fi
    
    # Check if PNG files exist
    if ! ls "$INPUT_DIR"/frame_*.png 1> /dev/null 2>&1; then
        echo "Warning: No frame_*.png files found in '$INPUT_DIR', skipping..."
        return 1
    fi
    
    echo "Generating video for camera: $cam_type"
    echo "Input directory: $INPUT_DIR"
    echo "Output file: $OUTPUT_FILE"
    
    # Run ffmpeg
    ffmpeg -y -framerate 30 -i "$INPUT_DIR/frame_%06d.png" -c:v libx264 -pix_fmt yuv420p "$OUTPUT_FILE"
    
    if [ $? -eq 0 ]; then
        echo "Video generated successfully: $OUTPUT_FILE"
        echo ""
    else
        echo "Error: Video generation failed for $cam_type"
        echo ""
        return 1
    fi
}

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Map camera type to folder name
case $CAMERA_TYPE in
    mujoco)
        FOLDER_NAME="observation.mujoco"
        generate_video "mujoco" "$FOLDER_NAME"
        ;;
    front_stereo_left)
        FOLDER_NAME="observation.rgb_front_stereo_left"
        generate_video "front_stereo_left" "$FOLDER_NAME"
        ;;
    front_stereo_right)
        FOLDER_NAME="observation.rgb_front_stereo_right"
        generate_video "front_stereo_right" "$FOLDER_NAME"
        ;;
    side_left)
        FOLDER_NAME="observation.rgb_side_left"
        generate_video "side_left" "$FOLDER_NAME"
        ;;
    wrist)
        FOLDER_NAME="observation.rgb_wrist"
        generate_video "wrist" "$FOLDER_NAME"
        ;;
    wrist_left)
        FOLDER_NAME="observation.rgb_wrist_left"
        generate_video "wrist_left" "$FOLDER_NAME"
        ;;
    head_stereo_left)
        FOLDER_NAME="observation.rgb_head_stereo_left"
        generate_video "head_stereo_left" "$FOLDER_NAME"
        ;;
    head_stereo_right)
        FOLDER_NAME="observation.rgb_head_stereo_right"
        generate_video "head_stereo_right" "$FOLDER_NAME"
        ;;
    all)
        echo "Generating videos for all cameras..."
        echo "========================================"
        generate_video "mujoco" "observation.rgb_mujoco"
        generate_video "front_stereo_left" "observation.rgb_front_stereo_left"
        generate_video "front_stereo_right" "observation.rgb_front_stereo_right"
        generate_video "side_left" "observation.rgb_side_left"
        generate_video "wrist" "observation.rgb_wrist"
        generate_video "wrist_left" "observation.rgb_wrist_left"
        generate_video "head_stereo_left" "observation.rgb_head_stereo_left"
        generate_video "head_stereo_right" "observation.rgb_head_stereo_right"
        echo "========================================"
        echo "All videos generation complete!"
        ;;
    *)
        echo "Error: Invalid camera type '$CAMERA_TYPE'"
        echo "Valid options: mujoco, front_stereo_left, front_stereo_right, side_left, wrist, wrist_left, head_stereo_left, head_stereo_right, or all"
        exit 1
        ;;
esac