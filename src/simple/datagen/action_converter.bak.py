import numpy as np
import torch
import transforms3d as t3d



class ActionConverterMujoco():
    def __init__(self, physics_hz, robot_type="aloha", ee_link="right_gripper_site"):
        """
        Initialize the ActionConverterMujoco for different robot types.
        
        Args:
            physics_hz: Physics simulation frequency
            robot_type: Type of robot ("aloha" or "franka")
            ee_link: End effector link for ALOHA ("right_gripper_site" or "left_gripper_site")
        """
        self.physics_hz = physics_hz
        self.robot_type = robot_type.lower()
        self.ee_link = ee_link
        
        # Configure robot-specific parameters
        self._configure_robot()

    def _configure_robot(self):
        """Configure robot-specific actuator mappings and parameters."""
        if "franka" in self.robot_type:
            # Franka Panda has 7 DOF arm + gripper
            self.arm_dof = 7
            self.arm_indices = list(range(7))  # 0-6 for arm
            self.gripper_indices = [7, 8]     # 7, 8 for gripper fingers
            self.total_actuators = 9
            self.default_gripper_values = [0.04, 0.04]
            
        elif self.robot_type == "aloha":
            # ALOHA has dual arms with 6 DOF each + grippers
            self.arm_dof = 6
            self.is_right_arm = self.ee_link == "right_gripper_site"
            
            # ALOHA actuator layout:
            # 0-5: left arm, 6: left gripper, 7-12: right arm, 13: right gripper
            if self.is_right_arm:
                self.arm_indices = list(range(7, 13))  # Right arm: 7-12
                self.gripper_indices = [13]            # Right gripper
                self.other_arm_indices = list(range(6)) # Left arm: 0-5
                self.other_gripper_indices = [6]       # Left gripper
            else:
                self.arm_indices = list(range(6))      # Left arm: 0-5
                self.gripper_indices = [6]             # Left gripper
                self.other_arm_indices = list(range(7, 13)) # Right arm: 7-12
                self.other_gripper_indices = [13]      # Right gripper
            
            self.total_actuators = 14
            self.default_gripper_value = 0.4
            
        else:
            raise ValueError(f"Unsupported robot type: {self.robot_type}")

    def convert_plan_to_action(self, plan, finger_init):
        approach_trajectory, lift_trajectory = plan[2:4] # np.array of N*7 or N*6
        action_commands = []
        
        action_commands.append(self.__make_init_action(approach_trajectory[0], finger_init, "approach"))
        
        for i in range(1, len(approach_trajectory)):
            action_commands.append(self.__make_action_command(approach_trajectory[i], "approach"))
        
        # Configure finger actions based on robot type
        if "franka" in self.robot_type:
            finger_init_vals = finger_init if isinstance(finger_init, list) else [0.04, 0.04]
        else:  # aloha
            finger_init_vals = [0.04] if isinstance(finger_init, list) and len(finger_init) == 1 else [finger_init[0]]
            
        action_commands.extend(self.__make_finger_actions(finger_init_vals, 0.0, int(1.0 * self.physics_hz), "grasp"))
        
        for i in range(len(lift_trajectory)):
            action_commands.append(self.__make_action_command(lift_trajectory[i], "lift"))
            
        for i in range(1 * self.physics_hz):
            action_commands.append(("wait", []))
            
        return action_commands

    def __make_init_action(self, traj, position, state):
        """Create initial action based on robot type."""
        actions = []
        traj_list = traj.tolist()
        
        if "franka" in self.robot_type:
            # Franka: direct mapping to arm actuators
            for i, action in enumerate(traj_list[:self.arm_dof]):
                actions.append((i, action))
            
            # Set gripper fingers
            if isinstance(position, list) and len(position) >= 2:
                actions.extend([(7, position[0]), (8, position[1])])
            else:
                pos_val = position[0] if isinstance(position, list) else position
                actions.extend([(7, pos_val), (8, pos_val)])
                
        elif self.robot_type == "aloha":
            # ALOHA: depends on which arm we're controlling
          
            # Map trajectory to left arm (0-5) and right arm extension
            for i, action in enumerate(traj_list):
                if i < 6:
                    actions.append((i, action))  # Left arm
                else:
                    actions.append((i + 1, action))  # Skip gripper index 6
            
            # Set both grippers for coordination
            pos_val = position[0] if isinstance(position, list) else position
            actions.append((6, pos_val))   # Left gripper
            actions.append((13, pos_val))  # Right gripper
        
        
        actions.sort(key=lambda x: x[0])
        return (state, actions)
    
    def __make_finger_actions(self, init_position, target_position, num_steps, state):
        """Create finger actions based on robot type."""
        finger_actions = []
        
        for i in range(1, num_steps + 1):
            finger_action = []
            
            for j, init_width in enumerate(init_position):
                interpolated_value = init_width + (i / num_steps) * (target_position - init_width)
                
                if "franka" in self.robot_type:
                    # Franka has two gripper fingers
                    finger_action.append((j + 7, interpolated_value))
                    
                elif self.robot_type == "aloha":
                    # ALOHA gripper index depends on arm
                    if self.is_right_arm:
                        finger_action.append((j + 13, interpolated_value))
                    else:
                        finger_action.append((j + 6, interpolated_value))
            
            finger_actions.append((state, finger_action))
            
        return finger_actions

    def __make_action_command(self, traj, state):
        """Create action command based on robot type."""
        actions = []
        traj_list = traj.tolist()
        
        if "franka" in self.robot_type:
            # Franka: direct mapping to arm actuators
            for i, action in enumerate(traj_list[:self.arm_dof]):
                actions.append((i, action))
            
            # Add default gripper values
            actions.extend([(7, self.default_gripper_values[0]), 
                           (8, self.default_gripper_values[1])])
                           
        elif self.robot_type == "aloha":
            if self.is_right_arm:
                # ALOHA right arm: map to right arm actuators (7-12)
                for i, action in enumerate(traj_list):
                    if i < 6:
                        actions.append((i + 7, action))
                    else:
                        # Handle additional DOF if present
                        actions.append((i - 6, action))
                
                # Set default left gripper value for coordination
                actions.append((6, self.default_gripper_value))
            else:
                # ALOHA left arm: direct mapping to left arm (0-5)
                for i, action in enumerate(traj_list):
                    if i < 6:
                        actions.append((i, action))
                    else:
                        # Handle additional DOF if present
                        actions.append((i + 1, action))
                actions.append((13, self.default_gripper_value))

                # No need to set other gripper for left arm operation
        
        actions.sort(key=lambda x: x[0])
        return (state, actions)