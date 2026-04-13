"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import numpy as np
from .base_agent import BaseAgent
from simple.constants import GripperAction
try: 
    from simple.datagen.planner import CuroboPlanner

    import curobo.util_file
    from curobo.types.math import Pose
    from curobo.geom.types import WorldConfig, Mesh, Cuboid
    from curobo.wrap.reacher.ik_solver import IKSolver
    from curobo.types.base import TensorDeviceType
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
except:
    raise ImportError("CuRobo is not installed. Please follow the instructions in the tutorials/installation")

import torch
import transforms3d as t3d
import sys
# from third_party.gsnet.util_gsnet import GSNet
from scipy.spatial.transform import Rotation as R
from PIL import Image
import cv2
import time
from pathlib import Path
import os
import random
import threading
import select
import termios
import tty
import fcntl

class KeyboardController:
    def __init__(self):
        self.current_key = None
        self.running = True
        self.old_settings = None
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            self._setup_nonblocking_input()
        except Exception as e:
            print(f"Keyboard controller initialization failed: {e}")
            self.running = False
        
    def _setup_nonblocking_input(self):
        try:
            fd = sys.stdin.fileno()
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            
            new_settings = termios.tcgetattr(sys.stdin)
            new_settings[3] = new_settings[3] & ~(termios.ICANON | termios.ECHO)
            new_settings[6][termios.VMIN] = 0
            new_settings[6][termios.VTIME] = 0
            termios.tcsetattr(sys.stdin, termios.TCSANOW, new_settings)
            
        except Exception as e:
            print(f"Failed to setup non-blocking input: {e}")
    
    def get_key(self):
        """Get keyboard input"""
        try:
            key = sys.stdin.read(1)
            if key:
                self.current_key = key
                return key
        except BlockingIOError:
            pass
        except Exception as e:
            print(f"Failed to read key: {e}")
        return None
    
    def stop(self):
        """Stop keyboard listening and restore terminal settings"""
        self.running = False
        if self.old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSANOW, self.old_settings)
                # Restore blocking IO
                fd = sys.stdin.fileno()
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
            except Exception as e:
                print(f"Failed to restore terminal settings: {e}")

class KeyboardAgent(BaseAgent):
    def __init__(self, robot):
        super().__init__(robot)
        self._gt_step_idx = 0
        
        self.planner = CuroboPlanner(
            robot=robot,
            robot_cfg=robot.robot_cfg,
            plan_batch_size=40,
            plan_dt=1.0/10,
            easy_motion_gen=False,
        )
        
        world_cfg = WorldConfig(cuboid=[Cuboid(
            'table', pose=[0., 0., -0.2, 1., 0., 0., 0.], scale=[1., 1., 1.], dims=[2., 2., 0.5]
            )])
        
        self.planner.motion_gen.update_world(world_cfg)
        

        self._last_pred_eef_pose = None
        self._last_pred_action = None
        
        # Keyboard control parameters
        self.keyboard_controller = KeyboardController()
        self.control_mode = "keyboard"  # "keyboard" or "auto"
        self.joint_step = 0.05  # Joint movement step size
        self.cartesian_step = 0.01  # Cartesian space movement step size
        self.rotation_step = 0.05  # Rotation step size (radians)
        self.control_space = "cartesian"  # "joint" or "cartesian"
        self.gripper_open = True
        self.selected_joint = 0  # Default to first joint
        
        # Image visualization settings
        self.enable_visualization = True  # Whether to enable real-time visualization
        self.window_name = "Wrist Camera View"
        if self.enable_visualization:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 640, 480)
        
        self._print_instructions()

    def get_action(self, observation, 
                   info, 
                   target_info=None,
                   instruction=None, 
                   conditions=None, 
                   replay_gt_action=None, 
                   check_error=False):
        
        self._last_observation = observation
        curr_qpos = observation["agent"]
        self._last_qpos = curr_qpos
        self._gt_step_idx += 1

        front_left_frame = observation["front_stereo_left"]
        front_left_image = Image.fromarray(front_left_frame)
        wrist_frame = observation["wrist"]
        wrist_image = Image.fromarray(wrist_frame)
        
        # Real-time display of wrist camera image
        if self.enable_visualization:
            self._display_wrist_camera(wrist_frame)


        # Keyboard control mode
        if self.control_mode == "keyboard":
            keyboard_action = self._handle_keyboard_input(observation)
            if keyboard_action is not None:
                self._last_pred_action = keyboard_action
                
                # Calculate end-effector pose
                try:
                    eef = self.robot.fk(keyboard_action[:7])
                    if isinstance(eef, tuple) and len(eef) == 2:
                        eef_pos = eef[0]
                        eef_quat = eef[1]
                        if hasattr(eef_pos, 'cpu'):
                            eef_pos = eef_pos.cpu().numpy()
                        if hasattr(eef_quat, 'cpu'):
                            eef_quat = eef_quat.cpu().numpy()
                        eef_combined = np.concatenate([eef_pos, eef_quat])
                        self._last_pred_eef_pose = eef_combined
                except Exception as e:
                    print(f"Failed to calculate end-effector pose: {e}")
                
                return keyboard_action
            else:
                # No key input, maintain current position
                return self._get_default_action(observation)
        
        # Auto mode - keep original curobo planning logic
        else:
            return self._get_default_action(observation)

    def query_action(self, observation, instruction, gt_action=None):
        if self._last_pred_action is not None:
            return self._last_pred_action, 1.0
        else:
            return self._get_default_action(observation), 1.0

    def reset(self):
        self._gt_step_idx = 0
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None

    def _display_wrist_camera(self, wrist_frame):
        """Display wrist camera image"""
        try:
            # Convert BGR to RGB (OpenCV uses BGR format)
            if len(wrist_frame.shape) == 3 and wrist_frame.shape[2] == 3:
                display_frame = cv2.cvtColor(wrist_frame, cv2.COLOR_RGB2BGR)
            else:
                display_frame = wrist_frame
            
            # Add some information to the image
            info_text = [
                f"Mode: {self.control_mode}",
                f"Space: {self.control_space}",
                f"Gripper: {'Open' if self.gripper_open else 'Close'}"
            ]
            
            # Add text information to the image
            for i, text in enumerate(info_text):
                cv2.putText(display_frame, text, (10, 25 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # Display image
            cv2.imshow(self.window_name, display_frame)
            cv2.waitKey(1)  # Non-blocking wait, allows window update
            
        except Exception as e:
            print(f"Failed to display wrist camera image: {e}")    
            
    def _print_instructions(self):
        """Print keyboard control instructions"""
        print("\n" + "="*60)
        print("Keyboard Control Instructions:")
        print("="*60)
        print("Mode switching:")
        print("  'm' - Toggle control mode (keyboard/auto)")
        print("  'c' - Toggle control space (joint/cartesian)")
        print("  'v' - Toggle wrist camera visualization")
        print("")
        print("Joint space control (joint mode):")
        print("  '1-7' - Select joint 1-7")
        print("  '+/-' - Increase/decrease selected joint angle")
        print("")
        print("Cartesian space control (cartesian mode):")
        print("  Position control:")
        print("    'w/s' - X-axis forward/backward")
        print("    'a/d' - Y-axis left/right")
        print("    'q/e' - Z-axis up/down")
        print("  Rotation control:")
        print("    'i/k' - Rotation around X-axis (pitch)")
        print("    'j/l' - Rotation around Y-axis (yaw)")
        print("    'u/o' - Rotation around Z-axis (roll)")
        print("")
        print("Gripper control:")
        print("  'g' - Toggle gripper open/close")
        print("")
        print("Others:")
        print("  'h' - Show help")
        print("  'r' - Reset to initial position")
        print("  ESC(ctrl+c) - Stop all actions")
        print("="*60)
        print(f"Current mode: {self.control_mode}")
        print(f"Current control space: {self.control_space}")
        print(f"Current selected joint: {self.selected_joint + 1}")
        print(f"Wrist camera visualization: {'ON' if self.enable_visualization else 'OFF'}")

    def _handle_keyboard_input(self, observation):
        """Handle keyboard input"""
        key = self.keyboard_controller.get_key()
        if key is None:
            return None
            
        print(f"Key detected: '{key}'")
            
        curr_qpos = observation["agent"]
        action = curr_qpos.copy()
        
        # Mode switching
        if key == 'm':
            self.control_mode = "auto" if self.control_mode == "keyboard" else "keyboard"
            print(f"Switched to {self.control_mode} mode")
            return self._get_default_action(observation)
        elif key == 'c':
            self.control_space = "cartesian" if self.control_space == "joint" else "joint"
            print(f"Switched to {self.control_space} control space")
            return self._get_default_action(observation)
        elif key == 'h':
            self._print_instructions()
            return self._get_default_action(observation)
        elif key == '\x03':  # Ctrl+C
            print("Stop all actions")
            return self._get_default_action(observation)
        elif key == 'v':
            # Toggle wrist camera visualization
            self.enable_visualization = not self.enable_visualization
            if self.enable_visualization:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, 640, 480)
                print("Wrist camera visualization: ON")
            else:
                cv2.destroyWindow(self.window_name)
                print("Wrist camera visualization: OFF")
            return self._get_default_action(observation)
            
        # Gripper control
        if key == 'g':
            self.gripper_open = not self.gripper_open
            print(f"Gripper: {'Open' if self.gripper_open else 'Close'}")
            return self._get_default_action(observation)
            
        # Reset
        if key == 'r':
            return self._get_reset_action()
            
        # Joint space control
        if self.control_space == "joint":
            joint_idx = None
            if key in '1234567':
                joint_idx = int(key) - 1
                self.selected_joint = joint_idx
                print(f"Selected joint {key}")
                return self._get_default_action(observation)
            elif key == '+' or key == '=':
                action[self.selected_joint] += self.joint_step
                print(f"Joint {self.selected_joint + 1} increased to {action[self.selected_joint]:.3f}")
            elif key == '-' or key == '_':
                action[self.selected_joint] -= self.joint_step
                print(f"Joint {self.selected_joint + 1} decreased to {action[self.selected_joint]:.3f}")
            else:
                return self._get_default_action(observation)
                
        # Cartesian space control
        elif self.control_space == "cartesian":
            delta_pos = np.zeros(3)
            delta_rot = np.zeros(3)  # Euler angle increment [roll, pitch, yaw]
            
            # Position control
            if key == 'w':
                delta_pos[0] = self.cartesian_step
                print("Moving in positive X direction")
            elif key == 's':
                delta_pos[0] = -self.cartesian_step
                print("Moving in negative X direction")
            elif key == 'a':
                delta_pos[1] = self.cartesian_step
                print("Moving in positive Y direction")
            elif key == 'd':
                delta_pos[1] = -self.cartesian_step
                print("Moving in negative Y direction")
            elif key == 'q':
                delta_pos[2] = self.cartesian_step
                print("Moving in positive Z direction")
            elif key == 'e':
                delta_pos[2] = -self.cartesian_step
                print("Moving in negative Z direction")
            
            # Rotation control
            elif key == 'i':
                delta_rot[0] = self.rotation_step  # Positive rotation around X-axis
                print("Rotating positively around X-axis")
            elif key == 'k':
                delta_rot[0] = -self.rotation_step  # Negative rotation around X-axis
                print("Rotating negatively around X-axis")
            elif key == 'j':
                delta_rot[1] = self.rotation_step  # Positive rotation around Y-axis
                print("Rotating positively around Y-axis")
            elif key == 'l':
                delta_rot[1] = -self.rotation_step  # Negative rotation around Y-axis
                print("Rotating negatively around Y-axis")
            elif key == 'u':
                delta_rot[2] = self.rotation_step  # Positive rotation around Z-axis
                print("Rotating positively around Z-axis")
            elif key == 'o':
                delta_rot[2] = -self.rotation_step  # Negative rotation around Z-axis
                print("Rotating negatively around Z-axis")
            else:
                return self._get_default_action(observation)
                
            if np.any(delta_pos != 0) or np.any(delta_rot != 0):
                # Calculate current end-effector position and orientation
                try:
                    current_eef = self.robot.fk(curr_qpos[:7])
                    if isinstance(current_eef, tuple):
                        current_pos = current_eef[0]
                        current_quat = current_eef[1]
                    else:
                        current_pos = current_eef[:3]
                        current_quat = current_eef[3:7]
                        
                    # Convert to numpy arrays
                    if hasattr(current_pos, 'cpu'):
                        current_pos = current_pos.cpu().numpy()
                    if hasattr(current_quat, 'cpu'):
                        current_quat = current_quat.cpu().numpy()
                    
                    # Calculate target position
                    target_pos = current_pos + delta_pos
                    
                    # Calculate target orientation
                    target_quat = current_quat.copy()
                    if np.any(delta_rot != 0):
                        # Convert quaternion to rotation matrix
                        current_rotation = R.from_quat(current_quat)
                        
                        # Apply Euler angle increment (using 'xyz' order)
                        delta_rotation = R.from_euler('xyz', delta_rot)
                        
                        # Combine rotations
                        new_rotation = current_rotation * delta_rotation
                        
                        # Convert back to quaternion
                        target_quat = new_rotation.as_quat()
                        
                        print(f"Rotation increment: roll={delta_rot[0]:.3f}, pitch={delta_rot[1]:.3f}, yaw={delta_rot[2]:.3f}")
                    
                    # Solve IK
                    target_qpos = self.robot.ik(target_pos, target_quat, current_joint=curr_qpos[:7])
                    if target_qpos is not None:
                        action[:7] = target_qpos
                        if np.any(delta_pos != 0):
                            print(f"Moving to position: {target_pos}")
                        if np.any(delta_rot != 0):
                            # Display current Euler angles
                            euler_angles = R.from_quat(target_quat).as_euler('xyz', degrees=True)
                            print(f"Current orientation(deg): roll={euler_angles[0]:.1f}, pitch={euler_angles[1]:.1f}, yaw={euler_angles[2]:.1f}")
                    else:
                        print("IK solution failed, cannot move to target position/orientation")
                        return self._get_default_action(observation)
                except Exception as e:
                    print(f"Cartesian control error: {e}")
                    return self._get_default_action(observation)
        
        # Set gripper action
        gripper_action = GripperAction.open if self.gripper_open else GripperAction.close
        final_action = np.concatenate([action[:7], [gripper_action]])
        
        return final_action

    def _get_reset_action(self):
        """Get reset action"""
        # Reset to initial joint positions
        reset_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])  # Franka default position
        gripper_action = GripperAction.open
        action = np.concatenate([reset_qpos, [gripper_action]])
        print("Resetting to initial position")
        return action

    def _get_default_action(self, observation):
        """Get default action (maintain current position)"""
        curr_qpos = observation["agent"]
        gripper_action = GripperAction.open if self.gripper_open else GripperAction.close
        action = np.concatenate([curr_qpos[:7], [gripper_action]])
        return action