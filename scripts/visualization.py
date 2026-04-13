#!/usr/bin/env python3
"""
USD Visualizer and Recorder for SIMPLE Environment (Replicator Version)

Uses Replicator's RGB annotator for reliable frame capture and video recording.
Compatible with the Python environment at /data2/kaidi/SIMPLE/.venv/bin/python

Usage:
    python visualize_usd_replicator.py --usd_path /path/to/file.usd --duration 10.0 --headless
"""

import typer
from typing_extensions import Annotated
import os
import sys
import time
import numpy as np
from simple.engines.isaac_app import create_simulation_app

# Global state matching SIMPLE's pattern
_ISAAC_LOADED = False
_SIMULATION_APP = None

def init_isaac_sim(headless: bool, webrtc: bool = False, width: int = 1920, height: int = 1080):
    """Initialize Isaac Sim following SIMPLE's exact pattern from BaseDualSim._init_isaac()"""
    global _ISAAC_LOADED, _SIMULATION_APP
    
    if _ISAAC_LOADED:
        print("Warning: Isaac Sim already loaded")
        return _SIMULATION_APP
    
    print(f"Initializing Isaac Sim (headless={headless}, webrtc={webrtc})...")
    
    import isaacsim
    from omni.isaac.kit import SimulationApp

    # Step 1: Create SimulationApp (matching SIMPLE's config)
    _SIMULATION_APP = create_simulation_app(
        SimulationApp,
        headless=headless,
        width=width,
        height=height,
        anti_aliasing=0,
        hide_ui=False,
    )
    
    # Step 2: Enable WebRTC streaming if requested (matching SIMPLE's pattern)
    if webrtc:
        from omni.isaac.core.utils.extensions import enable_extension
        
        try:
            from isaacsim import util  # Isaac Sim 4.5.0+
            _SIMULATION_APP.set_setting('/app/window/drawMouse', True)
            enable_extension('omni.kit.livestream.webrtc')
            print("WebRTC enabled (Isaac Sim 4.5.0+ mode)")
        except ImportError:
            # Isaac Sim 4.2.0
            _SIMULATION_APP.set_setting('/app/window/drawMouse', True)
            _SIMULATION_APP.set_setting('/app/livestream/proto', 'ws')
            _SIMULATION_APP.set_setting('/app/livestream/websocket/framerate_limit', 60)
            _SIMULATION_APP.set_setting('/ngx/enabled', False)
            enable_extension('omni.services.streamclient.webrtc')
            print("WebRTC enabled (Isaac Sim 4.2.0 mode)")
    
    _ISAAC_LOADED = True
    print("Isaac Sim initialized successfully")
    return _SIMULATION_APP


def inspect_scene_bounds(stage):
    """
    Inspect the USD stage to find scene bounds and recommended camera settings.
    Returns (center, size, recommended_distance) or (None, None, None) if failed.
    """
    from pxr import UsdGeom, Usd
    
    try:
        # Get scene bounding box
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
        root = stage.GetPseudoRoot()
        world_bbox = bbox_cache.ComputeWorldBound(root)
        
        if world_bbox:
            bbox_range = world_bbox.ComputeAlignedRange()
            bbox_min = np.array(bbox_range.GetMin())
            bbox_max = np.array(bbox_range.GetMax())
            bbox_center = (bbox_min + bbox_max) / 2.0
            bbox_size = bbox_max - bbox_min
            
            # Calculate recommended distance (1.5x diagonal)
            diagonal = np.linalg.norm(bbox_size)
            recommended_distance = diagonal * 1.5
            
            return bbox_center, bbox_size, recommended_distance
        
    except Exception as e:
        print(f"Warning: Failed to compute scene bounds: {e}")
    
    return None, None, None


def project_3d_to_2d(point_3d, camera_position, camera_target, width, height, fov=60.0):
    """
    Project a 3D point to 2D screen coordinates.
    Simple perspective projection for visualization.
    """
    # Camera coordinate system
    forward = camera_target - camera_position
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, np.array([0, 0, 1]))
    if np.linalg.norm(right) < 0.001:
        right = np.cross(forward, np.array([0, 1, 0]))
    right = right / np.linalg.norm(right)
    
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # Transform point to camera space
    point_cam = point_3d - camera_position
    x = np.dot(point_cam, right)
    y = np.dot(point_cam, up)
    z = np.dot(point_cam, forward)
    
    # Perspective projection
    if z <= 0:
        return None  # Behind camera
    
    fov_rad = np.radians(fov)
    aspect = width / height
    
    # Project to normalized device coordinates
    x_ndc = x / (z * np.tan(fov_rad / 2) * aspect)
    y_ndc = -y / (z * np.tan(fov_rad / 2))
    
    # Convert to screen coordinates
    x_screen = int((x_ndc + 1) * width / 2)
    y_screen = int((y_ndc + 1) * height / 2)
    
    # Check if in bounds
    if 0 <= x_screen < width and 0 <= y_screen < height:
        return (x_screen, y_screen)
    return None


def draw_bbox_on_frame(frame, bbox_min, bbox_max, camera_position, camera_target, color, thickness):
    """
    Draw a 3D bounding box on the frame.
    """
    import cv2
    
    height, width = frame.shape[:2]
    
    # 8 corners of the bounding box
    corners_3d = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_max[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]],
    ])
    
    # Project to 2D
    corners_2d = []
    for corner in corners_3d:
        pt_2d = project_3d_to_2d(corner, camera_position, camera_target, width, height)
        corners_2d.append(pt_2d)
    
    # Draw edges
    # Bottom face (z=min)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom rectangle
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top rectangle
        (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
    ]
    
    for i, j in edges:
        pt1 = corners_2d[i]
        pt2 = corners_2d[j]
        if pt1 is not None and pt2 is not None:
            cv2.line(frame, pt1, pt2, color, thickness)
    
    return frame


def main(
    usd_path: Annotated[str, typer.Option(help="Path to USD file to visualize")] = None,
    duration: Annotated[float, typer.Option(help="Recording duration in seconds (None = continuous)")] = None,
    output_path: Annotated[str, typer.Option(help="Output video file path")] = "./output_video.mp4",
    fps: Annotated[int, typer.Option(help="Video recording FPS")] = 30,
    resolution: Annotated[str, typer.Option(help="Resolution as WIDTHxHEIGHT")] = "1920x1080",
    headless: Annotated[bool, typer.Option(help="Run in headless mode")] = True,
    webrtc: Annotated[bool, typer.Option(help="Enable WebRTC streaming")] = False,
    camera_path: Annotated[str, typer.Option(help="USD camera path")] = "/OmniverseKit_Persp",
    record_video: Annotated[bool, typer.Option(help="Enable video recording")] = True,
    # Camera positioning options
    auto_frame: Annotated[bool, typer.Option(help="Auto-frame camera to fit scene")] = False,
    camera_position: Annotated[str, typer.Option(help="Camera position as 'x,y,z' (e.g., '1.0,1.0,1.0')")] = None,
    camera_target: Annotated[str, typer.Option(help="Camera look-at target as 'x,y,z' (e.g., '0,0,0')")] = None,
    camera_distance: Annotated[float, typer.Option(help="Camera distance from target (auto-computes position)")] = None,
    camera_angle: Annotated[str, typer.Option(help="Camera angle as 'azimuth,elevation' in degrees (e.g., '45,30')")] = None,
    # Speed control options
    playback_speed: Annotated[float, typer.Option(help="Playback speed multiplier (0.5=slow-mo, 2.0=fast, 1.0=normal)")] = 1.0,
    # Visualization options
    show_bbox: Annotated[bool, typer.Option(help="Draw scene bounding box in video")] = False,
    bbox_color: Annotated[str, typer.Option(help="Bounding box color as 'r,g,b' (0-255)")] = "0,255,0",
    bbox_thickness: Annotated[int, typer.Option(help="Bounding box line thickness in pixels")] = 2,
):
    """
    Visualize and optionally record a USD file using Isaac Sim.
    
    This script uses Replicator's RGB annotator for reliable frame capture.
    """
    
    if usd_path is None:
        print("Error: --usd_path is required")
        raise typer.Exit(1)
    
    # Verify USD file exists
    if not os.path.exists(usd_path):
        print(f"Error: USD file not found: {usd_path}")
        raise typer.Exit(1)
    
    # Parse resolution
    try:
        width, height = map(int, resolution.split('x'))
    except:
        print(f"Error: Invalid resolution format '{resolution}'. Use WIDTHxHEIGHT (e.g., 1920x1080)")
        raise typer.Exit(1)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Initialize Isaac Sim (must happen before other imports)
    simulation_app = init_isaac_sim(headless=headless, webrtc=webrtc, width=width, height=height)
    
    # Import Isaac modules after SimulationApp is created
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import open_stage
    import omni.replicator.core as rep
    import carb
    
    # Video writer setup
    video_writer = None
    render_product = None
    rgb_annotator = None
    
    try:
        # Load USD file
        print(f"Loading USD file: {usd_path}")
        open_stage(usd_path)
        
        # Initialize World
        print("Initializing World...")
        world = World()
        world.reset()
        
        # Speed control will be handled by simulation step rate
        # (stepping multiple times or with delays)
        if playback_speed != 1.0:
            print(f"\nPlayback speed: {playback_speed}x")
            if playback_speed < 1.0:
                print(f"  -> Slow motion: Each second of video = {1/playback_speed:.1f}s of simulation")
            else:
                print(f"  -> Fast forward: Each second of video = {1/playback_speed:.1f}s of simulation")
        
        # Let stage settle
        print("Letting stage settle...")
        for _ in range(10):
            world.step(render=True)
            simulation_app.update()
        
        # Setup camera positioning
        from omni.isaac.core.utils.viewports import set_camera_view
        from pxr import UsdGeom, Gf
        import omni.usd
        
        stage = omni.usd.get_context().get_stage()
        
        # Inspect scene if no camera target provided
        scene_center = None
        scene_size = None
        recommended_distance = None
        bbox_min_global = None
        bbox_max_global = None
        
        if camera_target is None and not auto_frame:
            print("\n" + "="*70)
            print("No camera target specified - inspecting scene...")
            print("="*70)
            
            scene_center, scene_size, recommended_distance = inspect_scene_bounds(stage)
            
            if scene_center is not None:
                # Store bbox coordinates for visualization
                bbox_min_global = scene_center - scene_size / 2.0
                bbox_max_global = scene_center + scene_size / 2.0
                
                print(f"Scene center:   [{scene_center[0]:7.3f}, {scene_center[1]:7.3f}, {scene_center[2]:7.3f}]")
                print(f"Scene size:     [{scene_size[0]:7.3f}, {scene_size[1]:7.3f}, {scene_size[2]:7.3f}]")
                print(f"Scene diagonal: {np.linalg.norm(scene_size):.3f}")
                print(f"Recommended distance: {recommended_distance:.3f}")
                print("\nUsing scene center as camera target.")
                print("="*70 + "\n")
                
                # Use scene center as target
                camera_target = f"{scene_center[0]:.6f},{scene_center[1]:.6f},{scene_center[2]:.6f}"
                
                # If no distance specified, use recommended
                if camera_distance is None:
                    camera_distance = recommended_distance
                    print(f"Using recommended distance: {camera_distance:.3f}\n")
            else:
                print("Warning: Could not determine scene bounds. Using default target (0,0,0)")
                print("="*70 + "\n")
        
        # If show_bbox is enabled but we don't have bbox yet, compute it
        if show_bbox and bbox_min_global is None:
            scene_center_temp, scene_size_temp, _ = inspect_scene_bounds(stage)
            if scene_center_temp is not None:
                bbox_min_global = scene_center_temp - scene_size_temp / 2.0
                bbox_max_global = scene_center_temp + scene_size_temp / 2.0
        
        # Parse camera positioning options
        cam_position = None
        cam_target = None
        
        if camera_target:
            try:
                cam_target = np.array([float(x) for x in camera_target.split(',')])
                print(f"Camera target set to: {cam_target}")
            except:
                print(f"Warning: Invalid camera_target format. Use 'x,y,z'")
                cam_target = None
        
        if camera_position:
            try:
                cam_position = np.array([float(x) for x in camera_position.split(',')])
                print(f"Camera position set to: {cam_position}")
            except:
                print(f"Warning: Invalid camera_position format. Use 'x,y,z'")
                cam_position = None
        
        # Auto-frame or calculate camera position
        if auto_frame:
            print("Auto-framing camera to scene bounds...")
            # Get scene bounding box
            bbox_cache = UsdGeom.BBoxCache(0, ['default'])
            root_prim = stage.GetPseudoRoot()
            bbox = bbox_cache.ComputeWorldBound(root_prim)
            
            if bbox:
                bbox_range = bbox.ComputeAlignedRange()
                bbox_min = np.array(bbox_range.GetMin())
                bbox_max = np.array(bbox_range.GetMax())
                
                # Calculate center and size
                center = (bbox_min + bbox_max) / 2.0
                size = np.linalg.norm(bbox_max - bbox_min)
                
                # Position camera at 45 degree angle
                distance = size * 1.5  # 1.5x the bounding box diagonal
                cam_position = center + np.array([distance * 0.7, distance * 0.7, distance * 0.5])
                cam_target = center
                
                print(f"Scene center: {center}, size: {size:.3f}")
                print(f"Auto-positioned camera at: {cam_position}")
        
        elif cam_position is not None and cam_target is not None:
            # Explicit position and target already set
            pass
        
        elif camera_distance is not None:
            # Camera distance specified - need to compute position
            
            # Default target if not specified
            if cam_target is None:
                cam_target = np.array([0.0, 0.0, 0.0])
                print(f"Using default target: {cam_target}")
            
            # Parse angle if provided
            azimuth = 45.0  # default
            elevation = 30.0  # default
            
            if camera_angle:
                try:
                    azimuth, elevation = [float(x) for x in camera_angle.split(',')]
                    print(f"Using camera angle: azimuth={azimuth}°, elevation={elevation}°")
                except:
                    print("Warning: Invalid camera_angle format. Using default (45°, 30°)")
            else:
                print(f"Using default camera angle: azimuth={azimuth}°, elevation={elevation}°")
            
            # Convert to radians
            azimuth_rad = np.radians(azimuth)
            elevation_rad = np.radians(elevation)
            
            # Spherical to Cartesian
            x = camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
            y = camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
            z = camera_distance * np.sin(elevation_rad)
            
            cam_position = cam_target + np.array([x, y, z])
            print(f"Computed camera position at distance {camera_distance}m: {cam_position}")
        
        # Apply camera view if position is set
        final_cam_position = None
        final_cam_target = None
        
        if cam_position is not None and cam_target is not None:
            print(f"\n=== Setting Camera ===")
            print(f"Position: {cam_position}")
            print(f"Target:   {cam_target}")
            print(f"Distance: {np.linalg.norm(cam_position - cam_target):.3f}m")
            print(f"======================\n")
            
            # Store for bbox drawing
            final_cam_position = cam_position.copy()
            final_cam_target = cam_target.copy()
            
            try:
                set_camera_view(
                    eye=cam_position,
                    target=cam_target,
                    camera_prim_path=camera_path
                )
                # Let camera update
                for _ in range(5):
                    simulation_app.update()
                print("Camera view applied successfully")
            except Exception as e:
                print(f"Warning: Failed to set camera view: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nNo camera positioning applied (using default camera)")
            if cam_position is None:
                print("  Reason: camera position not computed")
            if cam_target is None:
                print("  Reason: camera target not set")
        
        # Parse bbox color
        bbox_color_bgr = (0, 255, 0)  # Default green in BGR
        if show_bbox:
            try:
                r, g, b = [int(x) for x in bbox_color.split(',')]
                bbox_color_bgr = (b, g, r)  # OpenCV uses BGR
                print(f"\nBounding box visualization enabled (color: RGB({r},{g},{b}), thickness: {bbox_thickness})")
            except:
                print(f"Warning: Invalid bbox_color format. Using default green.")
        
        # Setup video recording if requested
        if record_video:
            print(f"Setting up video recording to: {output_path}")
            try:
                import cv2
                
                # Create render product
                render_product = rep.create.render_product(camera_path, (width, height))
                
                # Create RGB annotator
                rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                rgb_annotator.attach([render_product])
                
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                print("Video recording initialized successfully")
                
            except ImportError:
                print("Error: OpenCV (cv2) is required for video recording")
                print("Install it with: pip install opencv-python")
                record_video = False
            except Exception as e:
                print(f"Warning: Failed to initialize video recording: {e}")
                import traceback
                traceback.print_exc()
                record_video = False
        
        # Calculate total frames
        if duration is not None:
            total_frames = int(duration * fps)
            print(f"\nRecording for {duration} seconds ({total_frames} frames) at {fps} FPS...")
        else:
            total_frames = None
            print("\nRunning continuously (press Ctrl+C to stop)...")
        
        if webrtc:
            print("\nWebRTC streaming enabled. Connect via browser to view.")
            print("Check Isaac Sim console for WebRTC connection URL.")
        
        # Main loop
        frame_count = 0
        start_time = time.time()
        last_print_time = start_time
        target_frame_time = 1.0 / fps
        
        # Calculate how many simulation steps per frame for speed control
        # For slow-mo (0.5x): we advance physics slower, video plays slower
        # For fast-forward (2x): we advance physics faster, video plays faster
        steps_per_frame = max(1, int(playback_speed))
        
        if playback_speed != 1.0:
            print(f"\n=== Speed Control ===")
            print(f"  Playback speed: {playback_speed}x")
            print(f"  Steps per frame: {steps_per_frame}")
            print(f"====================\n")
        
        try:
            while simulation_app.is_running():
                frame_start = time.time()
                
                # Step the world (multiple times for fast-forward)
                for step_idx in range(steps_per_frame):
                    # Only render the last step
                    should_render = (step_idx == steps_per_frame - 1)
                    world.step(render=should_render)
                
                # Step replicator to update annotator
                if record_video and rgb_annotator:
                    rep.orchestrator.step(rt_subframes=4)
                
                simulation_app.update()
                
                # Capture frame if recording
                if record_video and rgb_annotator and video_writer:
                    try:
                        # Get RGB data from annotator
                        rgb_data = rgb_annotator.get_data()
                        
                        if rgb_data is not None and len(rgb_data) > 0:
                            # rgb_data is already a numpy array in the correct format
                            frame_rgb = np.array(rgb_data, copy=True)
                            
                            # Ensure correct shape
                            if frame_rgb.shape[0] == height and frame_rgb.shape[1] == width:
                                # Convert RGB to BGR for OpenCV
                                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                                
                                # Draw bounding box if enabled
                                if show_bbox and bbox_min_global is not None and bbox_max_global is not None:
                                    if final_cam_position is not None and final_cam_target is not None:
                                        frame_bgr = draw_bbox_on_frame(
                                            frame_bgr,
                                            bbox_min_global,
                                            bbox_max_global,
                                            final_cam_position,
                                            final_cam_target,
                                            bbox_color_bgr,
                                            bbox_thickness
                                        )
                                
                                video_writer.write(frame_bgr)
                            elif frame_count == 0:
                                print(f"Warning: Frame shape mismatch. Expected ({height}, {width}, 3), got {frame_rgb.shape}")
                    except Exception as e:
                        if frame_count == 0:
                            print(f"Warning: Failed to capture frame: {e}")
                            import traceback
                            traceback.print_exc()
                
                frame_count += 1
                
                # Check if duration reached
                if total_frames is not None and frame_count >= total_frames:
                    print(f"\nCompleted {frame_count} frames")
                    break
                
                # Print progress every second
                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    elapsed = current_time - start_time
                    if total_frames:
                        progress = (frame_count / total_frames) * 100
                        actual_fps = frame_count / elapsed if elapsed > 0 else 0
                        print(f"Progress: {elapsed:.1f}s / {duration:.1f}s ({progress:.1f}%) - {frame_count} frames @ {actual_fps:.1f} FPS")
                    else:
                        actual_fps = frame_count / elapsed if elapsed > 0 else 0
                        print(f"Recording... {elapsed:.1f}s elapsed ({frame_count} frames @ {actual_fps:.1f} FPS)")
                    last_print_time = current_time
                
                # Frame rate limiting
                frame_elapsed = time.time() - frame_start
                if frame_elapsed < target_frame_time:
                    time.sleep(target_frame_time - frame_elapsed)
        
        except KeyboardInterrupt:
            print(f"\n\nInterrupted by user. Recorded {frame_count} frames.")
        
        # Finalize video
        if record_video and video_writer:
            print("\nFinalizing video...")
            video_writer.release()
            print(f"Video saved to: {os.path.abspath(output_path)}")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)
    
    finally:
        # Clean shutdown (matching SIMPLE's pattern)
        print("\nShutting down Isaac Sim...")
        if video_writer is not None:
            video_writer.release()
        
        if not headless:
            # Spin for a moment to allow viewing final state
            timeout = 2.0
            start = time.monotonic()
            while simulation_app.is_running():
                simulation_app.update()
                if time.monotonic() - start > timeout:
                    break
        
        simulation_app.close()
        print("Shutdown complete")


if __name__ == "__main__":
    typer.run(main)
