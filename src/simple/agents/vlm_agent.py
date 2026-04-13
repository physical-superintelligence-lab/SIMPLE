"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""


from simple.core.types import Pose
from simple.robots.mixin import CuRoboMixin
from .base_agent import BaseAgent
from simple.constants import GripperAction
from simple.core.task import Task
from simple.robots.protocols import Controllable, HasKinematics
from simple.mp.planner import MotionPlanner
# from simple.action_primitives.move_right import move_right
# from simple.action_primitives.move_left import move_left
# from simple.action_primitives.move_up import move_up
# from simple.action_primitives.move_down import move_down
import numpy as np
from queue import Queue
import itertools
from typing import Union
import transforms3d as t3d

from .primitive_agent import PrimitiveAgent

from openai import OpenAI
import re
import base64
from typing import Optional, Tuple
import io
from PIL import Image


def vec_to_mat44(vec: list[float] | np.ndarray) -> np.ndarray:
    """ Convert a 7D vector (x,y,z,qw,qx,qy,qz) to a 4x4 transformation matrix. """
    mat = np.eye(4, dtype=np.float32)
    mat[:3, 3] = vec[0:3]
    mat[:3, :3] = t3d.quaternions.quat2mat(np.array(vec[3:7], dtype=np.float32))
    return mat

def mat44_to_vec(mat: np.ndarray) -> np.ndarray:
    """ Convert a 4x4 transformation matrix to a 7D vector (x,y,z,qw,qx,qy,qz). """
    pos = mat[:3, 3]
    quat = t3d.quaternions.mat2quat(mat[:3, :3])
    return np.concatenate([pos, quat]).astype(np.float32)

class ActiveVLMAgent(PrimitiveAgent):
    """ Agent that combines a VLM prompting and a Motion Planner to perform actions based on observations. 
    """
    robot: HasKinematics | Controllable

    def __init__(self, task:Task, planner: MotionPlanner, api_key: str, target_object: str, max_corrections: int):
        """
        Args:
            task (Task): The task instance containing the robot and environment details.
            planner (MotionPlanner): The motion planner used to generate action plans.  
        """
        super().__init__(task.robot)
        self.task = task
        self.planner = planner

        self.reset()

        # set up for GPT-5
        self.client = OpenAI(api_key=api_key)
        self.target_object = target_object
        self.max_corrections = max_corrections
        self.messages = []
        # set system prompt
        self.messages.append({"role": "system", "content": self.get_initial_prompt()})

    def reset(self):
        """Reset the agent's internal state."""
        self.current_joint_qpos = None  # in base frame
        self.current_left_eef_pose = None
        self.current_right_eef_pose = None
        self.T_right_cam_eef = None

    def validate_movement_command(self, command: str) -> Tuple[bool, dict, Optional[str]]:
        """
        Validate that the movement command follows the required format.
        
        Args:
            command: The movement command to validate
            
        Returns:
            Tuple of (is_valid, return_dict, error_message)
        """
        # Check if command is "DONE!"
        if command.strip().upper() == "DONE!":
            return True, {}, None
            
        # Pattern to match: direction + space + number + "cm"
        pattern = r'^(translate_up|translate_down|translate_left|translate_right|translate_forward|translate_backward)\s+(\d+(?:\.\d+)?)\s*cm$'
        match = re.match(pattern, command.strip().lower())
        
        if not match:
            return False, {}, f"Invalid format. Expected: 'direction Xcm' (e.g., 'translate_up 4cm', 'translate_forward 6cm') or 'DONE!'"
        
        direction, distance = match.groups()
        distance_float = float(distance)

        return_dict = {
            "direction": direction,
            "distance": distance_float
        }
        
        # Check distance constraints (reasonable range)
        if distance_float <= 0:
            return False, {}, "Distance must be positive"
        if distance_float > 50:  # 50cm max movement
            return False, {}, "Distance too large (max 50cm)"
            
        return True, return_dict, None

    def get_initial_prompt(self) -> str:
        """
        Get the initial prompt for the robot arm control task with image.
            
        Returns:
            Initial prompt string
        """

        return f"""You are controlling a robot arm with a wrist-mounted camera to capture a target object: {self.target_object}. 

        Your task is to suggest movements for the robot arm until the camera fully captures the target object: {self.target_object}.

        Available movements:
        - "translate_up Xcm" - move arm up by X centimeters
        - "translate_down Xcm" - move arm down by X centimeters  
        - "translate_left Xcm" - move arm left by X centimeters
        - "translate_right Xcm" - move arm right by X centimeters
        - "translate_forward Xcm" - move arm forward by X centimeters
        - "translate_backward Xcm" - move arm backward by X centimeters

        Rules:
        1. Only suggest ONE movement at a time
        2. Use the exact format: "direction Xcm" (e.g., "translate_up 4cm", "translate_left 2cm")
        3. When the target object is FULLY captured and visible, respond with "DONE!"
        4. Be strategic - consider the current view and what might be needed to capture the target
        """

    def get_observation_prompt(self, prev_fail) -> str:
        """
        Get the prompt for subsequent observations with image.

        Args:
            prev_fail: Whether the motion planning for the previous movement failed.

        Returns:
            Prompt string for the observation
        """
        if prev_fail:
            return f"""The previous movement could not be executed because it would move the robot outside its reachable workspace.
            This is the current camera view.

        Look at the current camera view and determine again where the robot arm should move next. Remember to respond with either:
        - A movement command: "direction Xcm" (e.g., "translate_up 4cm", "translate_left 5cm")
        - "DONE!" if the target object: {self.target_object} is FULLY captured"""
        else:
            return f"""This is the current camera view.

        Look at the current camera view and determine where the robot arm should move next. Remember to respond with either:
        - A movement command: "direction Xcm" (e.g., "translate_up 4cm", "translate_left 5cm")
        - "DONE!" if the target object: {self.target_object} is FULLY captured"""

    def get_gpt_response(self, prompt: str, wrist_image: np.ndarray) -> str:
        """
        Get response from GPT-5.
        
        Args:
            prompt: The prompt to send to GPT
            wrist_image: wrist image observation
            
        Returns:
            GPT response string
        """
        img = Image.fromarray(wrist_image)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        self.messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        })
        
        response = self.client.chat.completions.create(
            model="gpt-5",
            messages=self.messages,
            reasoning_effort="low",   # can be "minimal", "low", "medium", "high". Use "low" now as our example is simple.
            stream=False
        )
        response_content = response.choices[0].message.content.strip()
        self.messages.append({"role": "assistant", "content": response_content})
        return response_content

    def move_to_object(self, object_uid: str, info: dict, arm: str = "left", offset: np.ndarray | None = None):
        self.current_left_eef_pose = self.robot.get_link_pose("left_gripper_site", self.current_joint_qpos) # type: ignore
        self.current_right_eef_pose = self.robot.get_link_pose("right_gripper_site", self.current_joint_qpos) # type: ignore
        
        if object_uid not in info:
            print(f"[ActiveVLMAgent.move_to_object] Object '{object_uid}' not found in info")
            print(f"Available objects: {list(info.keys())}")
            if isinstance(self.current_joint_qpos, dict):
                self.queue_follow_path_with_eef([self.current_joint_qpos], "open_eef")
            return
        
        object_position = np.array(info[object_uid][:3], dtype=np.float32)
        
        if offset is None:
            offset = np.array([0.0, 0.0, 0.1], dtype=np.float32)
        target_position = object_position + offset
        
        if arm == "left":
            target_left_pose = (target_position.tolist(), self.current_left_eef_pose[3:7])
            goal_poses_right = (self.current_right_eef_pose[:3], self.current_right_eef_pose[3:7])
            
            try:
                plan = self.planner.batch_plan_for_approach([target_left_pose], link_poses={
                    "right_gripper_site": self.current_right_eef_pose
                })
                trajs, jnames = plan
                traj = []
                for qpos in trajs[-1]:
                    traj.append(dict(zip(jnames, qpos)))
                self.queue_follow_path_with_eef(traj, "open_eef")
                
            except (RuntimeError, ValueError) as e:
                print(f"[ActiveVLMAgent.move_to_object] Planning failed for left arm: {e}")
                # raise e
                if isinstance(self.current_joint_qpos, dict):
                    self.queue_follow_path_with_eef([self.current_joint_qpos], "open_eef")
                return
        elif arm == "right":
            target_right_pose = np.concatenate([target_position, self.current_right_eef_pose[3:7]])
            goal_poses = (self.current_left_eef_pose[:3], self.current_left_eef_pose[3:7])
            
            try:
                plan = self.planner.batch_plan_for_approach([goal_poses], link_poses={
                    "right_gripper_site": target_right_pose
                })
                trajs, jnames = plan
                traj = []
                for qpos in trajs[-1]:
                    traj.append(dict(zip(jnames, qpos)))
                self.queue_follow_path_with_eef(traj, "open_eef")
                
            except (RuntimeError, ValueError) as e:
                print(f"[ActiveVLMAgent.move_to_object] Planning failed for right arm: {e}")
                self.queue_move_qpos_with_eef(self.current_joint_qpos, "open_eef")
                return
        else:
            raise ValueError(f"Invalid arm '{arm}'. Must be 'left' or 'right'.")

    def move_camera(self, direction: str, distance: float=0.1, angle: float=0.0, step: float=0.01):
        """
        A single function handles all camera movements, including translation and rotation.
        distance is in meters, angle is in radians.
        Currently, we are supporting the following movements:
            - translation: translate_up, translate_down, translate_left, translate_right
            - rotation: rotate_left, rotate_right, rotate_up, rotate_down, rotate_clockwise, rotate_counterclockwise
        """
        self.current_left_eef_pose = self.robot.get_link_pose("left_gripper_site", self.current_joint_qpos) # type: ignore
        self.current_right_eef_pose = self.robot.get_link_pose("right_gripper_site", self.current_joint_qpos) # type: ignore
        self.current_right_camera_pose = self.robot.get_link_pose("right_gripper_camera", self.current_joint_qpos) # type: ignore

        # compute the transformation from the right camera to the right end-effector
        if self.T_right_cam_eef is None:
            self.T_right_cam_eef = np.linalg.inv(vec_to_mat44(self.current_right_camera_pose)) @ vec_to_mat44(self.current_right_eef_pose)

        rotation_matrix = t3d.quaternions.quat2mat(self.current_right_camera_pose[3:7])

        target_camera_pose = Pose.from_vec(self.current_right_camera_pose)
        
        if direction == "translate_up":
            target_camera_pose.position += -rotation_matrix[:, 1] * distance
        elif direction == "translate_down":
            target_camera_pose.position += rotation_matrix[:, 1] * distance
        elif direction == "translate_left":
            target_camera_pose.position += -rotation_matrix[:, 0] * distance
        elif direction == "translate_right":
            target_camera_pose.position += rotation_matrix[:, 0] * distance
        elif direction == "translate_forward":
            target_camera_pose.position += rotation_matrix[:, 2] * distance
        elif direction == "translate_backward":
            target_camera_pose.position += -rotation_matrix[:, 2] * distance
        elif direction == "rotate_left":
            rot = t3d.axangles.axangle2mat(rotation_matrix[:, 2], angle)
            new_rotation_matrix = rot @ rotation_matrix
            target_camera_pose.quaternion = t3d.quaternions.mat2quat(new_rotation_matrix).tolist()
        elif direction == "rotate_right":
            rot = t3d.axangles.axangle2mat(rotation_matrix[:, 2], -angle)
            new_rotation_matrix = rot @ rotation_matrix
            target_camera_pose.quaternion = t3d.quaternions.mat2quat(new_rotation_matrix).tolist()
        elif direction == "rotate_up":
            rot = t3d.axangles.axangle2mat(rotation_matrix[:, 0], angle)
            new_rotation_matrix = rot @ rotation_matrix
            target_camera_pose.quaternion = t3d.quaternions.mat2quat(new_rotation_matrix).tolist()
        elif direction == "rotate_down":
            rot = t3d.axangles.axangle2mat(rotation_matrix[:, 0], -angle)
            new_rotation_matrix = rot @ rotation_matrix
            target_camera_pose.quaternion = t3d.quaternions.mat2quat(new_rotation_matrix).tolist()
        elif direction == "rotate_clockwise":
            rot = t3d.axangles.axangle2mat(rotation_matrix[:, 1], -angle)
            new_rotation_matrix = rot @ rotation_matrix
            target_camera_pose.quaternion = t3d.quaternions.mat2quat(new_rotation_matrix).tolist()
        elif direction == "rotate_counterclockwise":
            rot = t3d.axangles.axangle2mat(rotation_matrix[:, 1], angle)
            new_rotation_matrix = rot @ rotation_matrix
            target_camera_pose.quaternion = t3d.quaternions.mat2quat(new_rotation_matrix).tolist()
        else:
            raise NotImplementedError

        left_pose = (self.current_left_eef_pose[:3], self.current_left_eef_pose[3:7])
        try:
            # compute the target right gripper pose
            target_right_gripper_site = vec_to_mat44(target_camera_pose.as_vec()) @ self.T_right_cam_eef

            plan = self.planner.batch_plan_for_approach([left_pose], link_poses={
                "right_gripper_site": mat44_to_vec(target_right_gripper_site) 
            })
            trajs, jnames = plan
            traj = []
            for qpos in trajs[-1]: # only take the last successful one
                traj.append(dict(zip(jnames, qpos)))
            self.queue_follow_path_with_eef(traj, "open_eef")
            return True

        except (RuntimeError, ValueError) as e:
            print(f"[ActiveVLMAgent.move_camera] Planning failed: {e}")
            # # Queue current position as neutral action
            # if isinstance(self.current_joint_qpos, dict):
            #     self.queue_follow_path_with_eef([self.current_joint_qpos], "open_eef")
            return False

    def get_action(self, observation, instruction=None, info=None, **kwargs):
        """
        Should be called by ENV every timestep, and return the next action to execute.
        
        Args:
            observation: The current observation from the environment.
            instruction: This is currently not used.
            info: oracle information from the environment.
            
        Returns:
            action (np.ndarray): 14 dim (6 dof left arm + 6 dof right arm + 2 gripper actions).
        """
        self.current_joint_qpos = observation["agent"]
        return super().get_action(observation, instruction=instruction, info=info, **kwargs)
    
    def get_movement_command(self, wrist_image: np.ndarray, prev_fail: bool = False):
        """
        This function get the movement command from the VLM agent.
        
        Args:
            wrist_image: The current wrist image observation.
            
        Returns:
            return the direction and distance/angle of the movement command.
        """

        # get GPT response
        response = self.get_gpt_response(self.get_observation_prompt(prev_fail=prev_fail), wrist_image)
        print(f"🤖 GPT Response: {response}")

        # validate movement command
        is_valid, return_dict, error = self.validate_movement_command(response)
        correction_attempts = 0
        while not is_valid and correction_attempts < self.max_corrections:
            print(f"❌ Invalid response: {error}")
            print("🔄 Asking for correction...")
            correction_prompt = f"Invalid response: {error}\nPlease provide a valid movement command or 'DONE!'"
            response = self.get_gpt_response(correction_prompt, wrist_image)
            print(f"🤖 Corrected Response: {response}")
            is_valid, return_dict, error = self.validate_movement_command(response)
            correction_attempts += 1
        
        # If still invalid, return default action
        if not is_valid:
            print(f"❌ Still invalid after {self.max_corrections} attempts: {error}")
            print(f"⚠️ Using default action: right 0.001cm")
            return "translate_right", 0.001   # 0.001 meters
            
        # Check if done
        if response.upper() == "DONE!":
            print(f"🎉 DONE! Target object: {self.target_object} is fully captured!")
            return "done", 0.0
        else:
            return return_dict["direction"], return_dict["distance"] / 100.0  # convert cm to meters
            