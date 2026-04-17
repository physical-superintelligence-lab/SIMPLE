"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import numpy as np
import torch
from .base_agent import BaseAgent
from simple.core.action import ActionCmd
from scipy.spatial.transform import Rotation as R
from PIL import Image
import cv2
import sys
import os
import termios
import fcntl

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.geom.types import WorldConfig, Cuboid
from simple.mp.curobo import CuRoboPlanner

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
        
        # Joint names for creating ActionCmd dicts
        self._joint_names = list(robot.init_joint_states.keys())
        self._arm_joint_names = [n for n in self._joint_names if not n.startswith("finger")]

        self._last_pred_eef_pose = None
        self._last_pred_action = None
        
        # Initialize CuRoboPlanner for IK solving and motion planning
        self.planner = None
        try:
            self.planner = CuRoboPlanner(
                robot=robot,
                plan_batch_size=40,
                plan_dt=1.0 / 10,
                easy_motion_gen=False,
            )
            world_cfg = WorldConfig(cuboid=[Cuboid(
                'table', pose=[0., 0., -0.2, 1., 0., 0., 0.],
                scale=[1., 1., 1.], dims=[2., 2., 0.5]
            )])
            self.planner.motion_gen.update_world(world_cfg)
            print("CuRoboPlanner initialized for keyboard control.")
        except Exception as e:
            print(f"Failed to init CuRoboPlanner, falling back to robot.ik(): {e}")
        
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

    def get_action(self, observation, instruction=None, **kwargs):
        
        self._last_observation = observation
        curr_qpos = observation["agent"]
        self._last_qpos = dict(zip(self._joint_names, curr_qpos))
        self._gt_step_idx += 1

        # Display wrist camera if available
        if "wrist" in observation:
            wrist_frame = observation["wrist"]
            if self.enable_visualization:
                self._display_wrist_camera(wrist_frame)

        # Keyboard control mode
        if self.control_mode == "keyboard":
            keyboard_action = self._handle_keyboard_input(observation)
            if keyboard_action is not None:
                self._last_pred_action = keyboard_action
                
                # Calculate end-effector pose for display
                try:
                    target_qpos = keyboard_action["target_qpos"]
                    if target_qpos is not None:
                        eef_pos, eef_quat = self.robot.fk(target_qpos)
                        self._last_pred_eef_pose = np.concatenate([eef_pos, eef_quat])
                except Exception as e:
                    print(f"Failed to calculate end-effector pose: {e}")
                
                return keyboard_action
            else:
                # No key input, maintain current position
                return self._make_hold_action(curr_qpos)
        
        # Auto mode - maintain current position
        else:
            return self._make_hold_action(curr_qpos)

    def query_action(self, obs_image, instruction, gt_action=None):
        if self._last_pred_action is not None:
            return self._last_pred_action, 1.0
        else:
            raise NotImplementedError

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._gt_step_idx = 0

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
        """Handle keyboard input, returns ActionCmd or None"""
        key = self.keyboard_controller.get_key()
        if key is None:
            return None
            
        print(f"Key detected: '{key}'")
            
        curr_qpos = observation["agent"]
        
        # Mode switching
        if key == 'm':
            self.control_mode = "auto" if self.control_mode == "keyboard" else "keyboard"
            print(f"Switched to {self.control_mode} mode")
            return self._make_hold_action(curr_qpos)
        elif key == 'c':
            self.control_space = "cartesian" if self.control_space == "joint" else "joint"
            print(f"Switched to {self.control_space} control space")
            return self._make_hold_action(curr_qpos)
        elif key == 'h':
            self._print_instructions()
            return self._make_hold_action(curr_qpos)
        elif key == '\x03':  # Ctrl+C
            print("Stop all actions")
            return self._make_hold_action(curr_qpos)
        elif key == 'v':
            self.enable_visualization = not self.enable_visualization
            if self.enable_visualization:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, 640, 480)
                print("Wrist camera visualization: ON")
            else:
                cv2.destroyWindow(self.window_name)
                print("Wrist camera visualization: OFF")
            return self._make_hold_action(curr_qpos)
            
        # Gripper control
        if key == 'g':
            self.gripper_open = not self.gripper_open
            eef_type = "open_eef" if self.gripper_open else "close_eef"
            print(f"Gripper: {'Open' if self.gripper_open else 'Close'}")
            qpos_dict = dict(zip(self._joint_names, curr_qpos))
            return ActionCmd(eef_type, target_qpos=qpos_dict)
            
        # Reset
        if key == 'r':
            return self._get_reset_action()
        
        # Build target arm qpos array (copy from current)
        arm_qpos = curr_qpos[:len(self._arm_joint_names)].copy()
            
        # Joint space control
        if self.control_space == "joint":
            if key in '1234567':
                self.selected_joint = int(key) - 1
                print(f"Selected joint {key}")
                return self._make_hold_action(curr_qpos)
            elif key == '+' or key == '=':
                arm_qpos[self.selected_joint] += self.joint_step
                print(f"Joint {self.selected_joint + 1} increased to {arm_qpos[self.selected_joint]:.3f}")
            elif key == '-' or key == '_':
                arm_qpos[self.selected_joint] -= self.joint_step
                print(f"Joint {self.selected_joint + 1} decreased to {arm_qpos[self.selected_joint]:.3f}")
            else:
                return self._make_hold_action(curr_qpos)
                
        # Cartesian space control
        elif self.control_space == "cartesian":
            delta_pos = np.zeros(3)
            delta_rot = np.zeros(3)
            
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
            elif key == 'i':
                delta_rot[0] = self.rotation_step
                print("Rotating positively around X-axis")
            elif key == 'k':
                delta_rot[0] = -self.rotation_step
                print("Rotating negsssatively around X-axis")
            elif key == 'j':
                delta_rot[1] = self.rotation_step
                print("Rotating positively around Y-axis")
            elif key == 'l':
                delta_rot[1] = -self.rotation_step
                print("Rotating negatively around Y-axis")
            elif key == 'u':
                delta_rot[2] = self.rotation_step
                print("Rotating positively around Z-axis")
            elif key == 'o':
                delta_rot[2] = -self.rotation_step
                print("Rotating negatively around Z-axis")
            else:
                return self._make_hold_action(curr_qpos)
                
            if np.any(delta_pos != 0) or np.any(delta_rot != 0):
                try:
                    current_pos, current_quat = self.robot.fk(curr_qpos)
                    current_pos = np.asarray(current_pos)
                    current_quat = np.asarray(current_quat)
                    
                    target_pos = current_pos + delta_pos
                    target_quat = current_quat.copy()
                    
                    if np.any(delta_rot != 0):
                        current_rotation = R.from_quat(current_quat)
                        delta_rotation = R.from_euler('xyz', delta_rot)
                        new_rotation = current_rotation * delta_rotation
                        target_quat = new_rotation.as_quat()
                        print(f"Rotation increment: roll={delta_rot[0]:.3f}, pitch={delta_rot[1]:.3f}, yaw={delta_rot[2]:.3f}")
                    
                    arm_qpos = self._solve_ik(target_pos, target_quat, curr_qpos)
                    
                    if np.any(delta_pos != 0):
                        print(f"Moving to position: {target_pos}")
                    if np.any(delta_rot != 0):
                        euler_angles = R.from_quat(target_quat).as_euler('xyz', degrees=True)
                        print(f"Current orientation(deg): roll={euler_angles[0]:.1f}, pitch={euler_angles[1]:.1f}, yaw={euler_angles[2]:.1f}")
                except RuntimeError:
                    print("IK solution failed, cannot move to target position/orientation")
                    return self._make_hold_action(curr_qpos)
                except Exception as e:
                    print(f"Cartesian control error: {e}")
                    return self._make_hold_action(curr_qpos)
            else:
                return self._make_hold_action(curr_qpos)
        
        # Build target_qpos dict from arm joints + current finger joints
        target_qpos = {}
        for i, name in enumerate(self._arm_joint_names):
            target_qpos[name] = float(arm_qpos[i])
        for name in self._joint_names:
            if name not in target_qpos:
                target_qpos[name] = float(curr_qpos[self._joint_names.index(name)])
                
        eef_state = "open_eef" if self.gripper_open else "close_eef"
        return ActionCmd("move_qpos_with_eef", target_qpos=target_qpos, eef_state=eef_state)

    def _solve_ik(self, target_pos, target_quat, full_qpos):
        """Solve IK using CuRoboPlanner's lift_ik_solver if available, otherwise fall back to robot.ik()
        
        Args:
            target_pos: target end-effector position
            target_quat: target end-effector quaternion
            full_qpos: full joint state array (all DOFs including fingers)
        """
        if self.planner is not None:
            tensor_args = TensorDeviceType()
            p = tensor_args.to_device(torch.tensor(target_pos, dtype=torch.float32))
            q = tensor_args.to_device(torch.tensor(target_quat, dtype=torch.float32))
            goal = Pose(p, q)
            retract_cfg = torch.tensor(full_qpos, dtype=torch.float32).cuda()
            num_seeds = self.planner.lift_ik_solver.num_seeds
            seed_cfg = retract_cfg.unsqueeze(0).repeat(num_seeds, 1).cuda().unsqueeze(0)
            result = self.planner.lift_ik_solver.solve_single(goal, retract_cfg, seed_cfg)
            if result.success[0, 0]:
                return result.solution[0, 0].cpu().numpy()[:len(self._arm_joint_names)]
            raise RuntimeError("CuRobo IK failed")
        else:
            result = self.robot.ik(target_pos, target_quat, current_joint=full_qpos)
            return result[:len(self._arm_joint_names)]

    def _get_reset_action(self):
        """Get reset action - move to robot's initial joint positions"""
        self.gripper_open = True
        print("Resetting to initial position")
        return ActionCmd("move_qpos_with_eef", 
                         target_qpos=dict(self.robot.init_joint_states),
                         eef_state="open_eef")

    def _make_hold_action(self, curr_qpos):
        """Create an ActionCmd that holds the current position with current gripper state"""
        qpos_dict = dict(zip(self._joint_names, [float(v) for v in curr_qpos]))
        eef_state = "open_eef" if self.gripper_open else "close_eef"
        return ActionCmd("move_qpos_with_eef", target_qpos=qpos_dict, eef_state=eef_state)