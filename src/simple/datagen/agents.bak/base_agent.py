from collections import deque
import torch
import transforms3d as t3d
import numpy as np

class ActionCommand:
    def __init__(self, command, ideal_control=False, params=None):
        self.command = command
        self.ideal_control = ideal_control
        self.params = params

class BaseAgent:
    """
    Provides common support for different robot types (Franka, ALOHA).
    """
    def __init__(self, robot, ik_solver=None, robot_type="franka", ee_link="right_gripper_site"):
        self.__ctrl_queue = deque()
        self.__ctrl_queue_callbacks = []
        self.robot = robot
        self.__finger_state = 'open'
        self.ik_solver = ik_solver
        
        # Configure robot-specific parameters
        self.robot_type = robot_type.lower()
        self.ee_link = ee_link
        self._configure_robot()

    def _configure_robot(self):
        """Configure robot-specific parameters."""
        if "franka" in self.robot_type:
            self.arm_dof = 7
            self.gripper_indices = [7, 8]
            self.openness_index = 7
            
            # Franka gripper values
            self.gripper_close_intermediate = [0.04, 0.04]
            self.gripper_close_final = [0.01, 0.01]
            self.gripper_open_intermediate = [0.022, 0.022]
            self.gripper_open_final = [0.0205, 0.0205]
            
            # Franka finger length for IK
            self.finger_length = 0.1034  # FRANKA_FINGER_LENGTH
            
        elif self.robot_type == "aloha":
            self.arm_dof = 13  # ALOHA dual arm
            self.is_right_arm = self.ee_link == "right_gripper_site"
            self.openness_index = 13
            
            if self.is_right_arm:
                self.gripper_indices = [13]
                self.gripper_close_intermediate = [0.038]
                self.gripper_close_final = [0.02]
                self.gripper_open_intermediate = [0.22]
                self.gripper_open_final = [0.205]
            else:
                self.gripper_indices = [6]
                self.gripper_close_intermediate = [0.038]
                self.gripper_close_final = [0.02]
                self.gripper_open_intermediate = [0.22]
                self.gripper_open_final = [0.205]
                
            # ALOHA finger length (if needed for IK)
            self.finger_length = 0.0  # Approximate value
            
        else:
            raise ValueError(f"Unsupported robot type: {self.robot_type}")

    def _process_openness(self, openness):
        """
        Process openness command based on robot type.
        openness: < -0.9 for close, > 0.9 for open, otherwise no change
        """
        desired_finger_state = None
        if openness < -0.9 and self.__finger_state != 'close':
            desired_finger_state = 'close'
        elif openness > 0.9 and self.__finger_state != 'open':
            desired_finger_state = 'open'
        
        def set_finger_state(self=self, s=desired_finger_state):
            self.__finger_state = s
        
        if desired_finger_state == 'close' and self.__finger_state == 'open':
            self._add_gripper_close_commands()
            self.__ctrl_queue_callbacks.append(set_finger_state)
        elif desired_finger_state == 'open' and self.__finger_state != 'open':
            self._add_gripper_open_commands()
            self.__ctrl_queue_callbacks.append(set_finger_state)

    def _add_gripper_close_commands(self):
        """Add gripper close commands based on robot type."""
        if "franka" in self.robot_type:
            # Franka: gradual closing for both fingers
            close_commands = []
            for val in self.gripper_close_intermediate:
                close_commands.append([(7, val), (8, val)])
            self.__ctrl_queue.extend(close_commands * 5)
            
            final_command = [(7, self.gripper_close_final[0]), (8, self.gripper_close_final[1])]
            self.__ctrl_queue.append(final_command)
            
        elif self.robot_type == "aloha":
            # ALOHA: single gripper
            intermediate_val = self.gripper_close_intermediate[0]
            final_val = self.gripper_close_final[0]
            gripper_idx = self.gripper_indices[0]
            
            self.__ctrl_queue.extend([[(gripper_idx, intermediate_val)]] * 5)
            self.__ctrl_queue.append([(gripper_idx, final_val)])

    def _add_gripper_open_commands(self):
        """Add gripper open commands based on robot type."""
        if "franka" in self.robot_type:
            # Franka: gradual opening for both fingers
            open_commands = []
            for val in self.gripper_open_intermediate:
                open_commands.append([(7, val), (8, val)])
            self.__ctrl_queue.extend(open_commands * 5)
            
            final_command = [(7, self.gripper_open_final[0]), (8, self.gripper_open_final[1])]
            self.__ctrl_queue.append(final_command)
            
        elif self.robot_type == "aloha":
            # ALOHA: single gripper
            intermediate_val = self.gripper_open_intermediate[0]
            final_val = self.gripper_open_final[0]
            gripper_idx = self.gripper_indices[0]
            
            self.__ctrl_queue.extend([[(gripper_idx, intermediate_val)]] * 5)
            self.__ctrl_queue.append([(gripper_idx, final_val)])

    def step(self):
        """
        Should be called by inherited agents, and return the next action to execute.
        Raises StopIteration if no more actions are available.
        """
        if len(self.__ctrl_queue) == 0:
            # Execute pending callbacks
            for cb in self.__ctrl_queue_callbacks:
                cb()
            self.__ctrl_queue_callbacks.clear()

            action_command = self._step()
            
            if action_command.command == 'move_joints':
                self.__ctrl_queue.append(action_command.params)
                
            elif action_command.command == 'move_joints_with_openness':
                # Extract arm positions based on robot type
                arm_qpos = action_command.params[:self.arm_dof]
                self.__ctrl_queue.append(list(enumerate(arm_qpos)))
                self._process_openness(action_command.params[self.openness_index])
                
            elif action_command.command == 'move_eef':
                # Move eef to target pose (x, y, z, roll, pitch, yaw, openness)
                hand_pose = self._get_hand_pose_from_eef_pose(
                    action_command.params[:3], 
                    action_command.params[3:6]
                )
                arm_qpos = self._solve_ik(hand_pose)
                self.__ctrl_queue.append(list(enumerate(arm_qpos)))
                self._process_openness(action_command.params[6])
            else:
                raise NotImplementedError(f"Unknown command: {action_command.command}")
                
        return self.__ctrl_queue.popleft()

    def _step(self) -> ActionCommand:
        """
        To be implemented by inherited agents.
        Return the next action to execute, or raise StopIteration if no more actions are available.
        """
        raise NotImplementedError()

    def set_task(self, task: str):
        """Set the language description of the task."""
        self.__task = task

    def get_task(self):
        """Get the current task description."""
        return getattr(self, '_BaseAgent__task', None)

    def _get_finger_joint_ctrls(self, target_widths):
        """
        Get finger joint controls based on robot type.
        For Franka: Two fingers. Move at most 0.04/10 per step.
        For ALOHA: Single gripper value.
        """
        if "franka" in self.robot_type:
            target_widths = np.array(target_widths)
            current_positions = self.robot.get_joint_positions()
            init_widths = np.array(current_positions[7:9])
            STEP = 0.04/10
            ret = []
            
            while True:
                delta = target_widths - init_widths
                if np.all(np.abs(delta) < 0.0001):
                    break
                step = np.clip(delta, -STEP, STEP)
                init_widths += step
                ret.append(list(zip([7, 8], init_widths)))
            return ret
            
        elif self.robot_type == "aloha":
            # ALOHA gripper control is typically handled differently
            current_positions = self.robot.get_joint_positions()
            gripper_idx = self.gripper_indices[0]
            current_width = current_positions[gripper_idx]
            target_width = target_widths[0] if isinstance(target_widths, (list, tuple)) else target_widths
            
            STEP = 0.04/10  # Adjust as needed for ALOHA
            ret = []
            
            while abs(target_width - current_width) > 0.0001:
                delta = target_width - current_width
                step = np.clip(delta, -STEP, STEP)
                current_width += step
                ret.append([(gripper_idx, current_width)])
                
            return ret

    def _get_hand_pose_from_eef_pose(self, pred_eef_position, pred_eef_orientation):
        """
        Get target robot hand pose from end-effector pose.
        Args:
            pred_eef_position: target eef position
            pred_eef_orientation: target eef orientation (euler angles)
        """
        EEF_TO_HAND = np.array([0., 0., -self.finger_length])
        target_rotation_matrix = t3d.euler.euler2mat(
            pred_eef_orientation[0], 
            pred_eef_orientation[1], 
            pred_eef_orientation[2]
        )
        target_eef_position = np.array(pred_eef_position)

        target_hand_position = target_rotation_matrix @ EEF_TO_HAND + target_eef_position
        target_hand_orientation = t3d.quaternions.mat2quat(target_rotation_matrix)
        return np.concatenate([target_hand_position, target_hand_orientation])

    def _solve_ik(self, hand_pose):
        """
        Solve inverse kinematics for the given hand pose.
        Returns joint positions for the arm.
        """
        if not hasattr(self, 'ik_solver') or self.ik_solver is None:
            # Fallback: return current joint positions
            current_joint = self.robot.get_joint_positions()[:self.arm_dof]
            return current_joint
            
        try:
            from curobo.types.math import Pose

            if not hasattr(self, 'curobo_lock'):
                import threading
                self.curobo_lock = threading.Lock()

            with self.curobo_lock:
                ee_position = torch.tensor(hand_pose[:3]).cuda().float()
                ee_quaternion = torch.tensor(hand_pose[3:]).cuda().float()
                goal = Pose(ee_position, ee_quaternion)

                current_joint = self.robot.get_joint_positions()[:self.arm_dof]
                retract_cfg = torch.tensor(current_joint).cuda().float()
                seed_cfg = torch.tensor(current_joint).float()
                seed_cfg = seed_cfg.unsqueeze(0).repeat(64, 1).cuda().unsqueeze(0)

                result = self.ik_solver.solve_single(goal, retract_cfg, seed_cfg)
                if not result.success[0][0]:
                    qpos = current_joint
                    self.ik_state = ' IK Fails'
                else:
                    qpos = result.solution[result.success][0].cpu().numpy()
                    self.ik_state = ' IK Succeeds'
                    
                return qpos
                
        except Exception as e:
            print(f"IK solving failed: {e}")
            # Return current joint positions as fallback
            current_joint = self.robot.get_joint_positions()[:self.arm_dof]
            return current_joint

    def get_robot_state(self):
        """Get current robot state information."""
        return {
            'finger_state': self.__finger_state,
            'robot_type': self.robot_type,
            'ee_link': self.ee_link,
            'queue_length': len(self.__ctrl_queue)
        }

    def clear_queue(self):
        """Clear the control queue and callbacks."""
        self.__ctrl_queue.clear()
        self.__ctrl_queue_callbacks.clear()