# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

import numpy as np
from simple.agents.primitive_agent import PrimitiveAgent
from simple.constants import GripperAction
from simple.envs.wrappers.episode_extractor import EpisodeExtractor
from simple.core.action import ActionCmd
from simple.baselines.psi0 import psi0_action_to_cmd
from simple.robots.protocols import Humanoid, DualArm, Wholebody
from PIL import Image

class ReplayAgent(PrimitiveAgent):
    def __init__(self, episode, robot, data_format="rlds", upsample_factor=1):
        super().__init__(robot)
        self._gt_step_idx = 0
        self._episode = EpisodeExtractor(episode, data_format)
        self._last_gripper_value = None
        self._upsample_factor = upsample_factor
        self._preload_episode()

    def _preload_episode(self):
        episode_length = len(self._episode)

        if isinstance(self.robot, Humanoid) and self._episode.is_psi0_format():
            # psi0 training data (36-dim `action`): execute each action exactly as Psi0Agent does in eval
            print(f"Preloading psi0-format episode with {episode_length} steps.")
            for step_idx in range(episode_length):
                psi0_action = self._episode.get_postprocess_action(step_idx)
                for _ in range(self._upsample_factor): # account for upsampling during training
                    self.queue_action(psi0_action_to_cmd(self.robot, psi0_action))
            return

        # Stand-warmup is now performed inside env.reset by StandStabilizationWrapper.
        # The agent's queue holds only dataset actions so step 0 = first dataset action.

        print(f"Preloading episode with {episode_length} steps.")
        for step_idx in range(0, episode_length-1):

            #FIXME now it only for g1,using actuators action,not qpos
            # gt_ctrl = self._episode.get_proprio_actuators_action(step_idx)
            # gt_gripper = self._episode.get_action_gripper(step_idx)

            # target_ctrl = dict(zip(self.robot.joint_names, gt_ctrl))
            # gripper_value = float(np.mean(gt_gripper))

            # self.queue_action(ActionCmd(
            #     "move_actuators", 
            #     target_qpos=target_ctrl
            # ))

            # left_image = self._episode.get_observation_image(step_idx)
            # Image.fromarray(left_image).save(f"step_{step_idx:04d}.png")
            
            if isinstance(self.robot, Humanoid):
                """ gt_ctrl = self._episode.get_proprio_actuators_action(step_idx)
                target_ctrl = dict(zip(self.robot.joint_names, gt_ctrl))

                self.queue_action(ActionCmd(
                    "replay_move_actuators", 
                    target_qpos=target_ctrl
                ))"""

                # raw recording: the AMO command and joint targets are stored directly
                gt_ctrl = self._episode.get_proprio_actuators_action(step_idx) # FIXME bad naming
                target_qpos = dict(zip(self.robot.joint_names[15:], gt_ctrl[15:])) # FIXME bad naming

                target_yaw = self._episode.get_amo_policy_target_yaw(step_idx).item()

                amo_command = self._episode.get_amo_policy_command(step_idx)
                turning_flag = self._episode.get_amo_policy_turning_flag(step_idx)
                # [vx, target_yaw, vy, d_height, torso_yaw, torso_pitch, torso_roll, turning_flag]: AMO reads
                # commands[7] (the turning flag), so a 7-element command raises IndexError in env.step
                command = [amo_command[0],target_yaw,amo_command[1],amo_command[-1]-0.75,amo_command[3],amo_command[4],amo_command[5],turning_flag]

                target_waist_qpos =  gt_ctrl[12:15]
                target_waist_qpos = dict(zip(self.robot.joint_names[12:15], target_waist_qpos))

                for _ in range(self._upsample_factor): # account for upsampling during training
                    self.queue_action(ActionCmd(
                        "eval_move_actuators", 
                        target_qpos=target_qpos,
                        action_command=command.copy(),
                        waist_qpos=target_waist_qpos
                    ))

                """ dyaw = self._episode.get_amo_policy_dyaw(step_idx).item()
                amo_command = self._episode.get_amo_policy_command(step_idx)
                command = [amo_command[0],dyaw,amo_command[1],amo_command[-1]-0.75,amo_command[3],amo_command[4],amo_command[5]]

                target_waist_qpos =  gt_ctrl[12:15]

                self.queue_action(ActionCmd(
                    "eval_move_actuators", 
                    target_qpos=target_qpos,
                    action_command=command,
                    target_waist_qpos=target_waist_qpos
                )) """

            elif isinstance(self.robot, DualArm):
                gt_qpos = self._episode.get_proprio_joint_position(step_idx)
                gt_gripper = self._episode.get_action_gripper(step_idx)
                target_qpos = dict(zip(self.robot.joint_names, gt_qpos))
                        
                left_eef="close_eef" if gt_gripper[0] < 0.01 else "open_eef"
                right_eef="close_eef" if gt_gripper[1] < 0.01 else "open_eef"
                
                self.queue_move_qpos_with_eef(
                    target_qpos=target_qpos,
                    eef_state="dual_eef",
                    left_eef=left_eef,
                    right_eef=right_eef,  
                )               
            else:
                gt_qpos = self._episode.get_proprio_joint_position(step_idx)
                gt_gripper = self._episode.get_action_gripper(step_idx)

                
                target_qpos = dict(zip(self.robot.joint_names, gt_qpos))
                
                gripper_value = float(np.mean(gt_gripper))
                
                self.queue_action(ActionCmd(
                    "move_qpos", 
                    target_qpos=target_qpos
                ))
            
                gripper_changed = (
                    self._last_gripper_value is None or 
                    abs(gripper_value - self._last_gripper_value) > 0.02
                )

                if gripper_changed:
                    if gripper_value < 0.01:
                        
                        # for _ in range(3):
                        self.queue_close_gripper()
                        # self._finger_state = "close_eef"
                    elif gripper_value > 0.04:
                        self.queue_open_gripper()
                        # self._finger_state = "open_eef"
                    self._last_gripper_value = gripper_value
            
            # self.queue_move_qpos_with_eef(
            #     target_qpos=target_qpos, 
            #     eef_state="open_eef" if gt_gripper==1 else "close_eef"
            # )

    def get_action(self, observation, instruction=None, info=None, **kwargs):
        self._last_observation = observation
        self._last_qpos = observation["joint_qpos"]
        
        action_cmd = super().get_action(observation, instruction, **kwargs)
        
        self._last_pred_action = action_cmd
        return action_cmd

    # def query_action(self, observation, instruction, gt_action=None):
    #     return self._last_pred_action, 1.0

    def reset(self):
        super().reset()  # clear queue
        self._gt_step_idx = 0
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None
        self._last_gripper_value = None
        self._finger_state = None
        
        self._preload_episode()
