"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simple.agents import BaseAgent

import os
import cv2
import json
import numpy as np
import gymnasium as gym
import logging
import shutil
import transforms3d as t3d
from PIL import Image
import matplotlib.pyplot as plt

from simple.envs.video_writer import VideoWriter
from simple.constants import GripperAction, GripperState #, GraspNet_1B_Object_Names
from simple.envs.wrappers.episode_extractor import EpisodeExtractor

def plot_err(err_proprio_between_gt_and_sim, err_ctrl_between_pred_and_proprio, err_action_per_step, 
             fname, first_close_step=0, gt_close_step=0, err_qpos_ctrl=None, obs_gt_closing=None, obs_sim_closing=None, action_token_accuracy=None):
    
    err_proprio_between_gt_and_sim = np.array(err_proprio_between_gt_and_sim, dtype=np.float32)
    err_ctrl_between_pred_and_proprio = np.array(err_ctrl_between_pred_and_proprio, dtype=np.float32)
    err_qpos_ctrl = np.array(err_qpos_ctrl, dtype=np.float32) if err_qpos_ctrl is not None else None
    err_action_per_step = np.array(err_action_per_step, dtype=np.float32)

    x = list(range(err_proprio_between_gt_and_sim.shape[0]))
    assert err_proprio_between_gt_and_sim.shape[0] == \
            err_ctrl_between_pred_and_proprio.shape[0] == \
            err_qpos_ctrl.shape[0] == len(err_action_per_step)
    # fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig, axs = plt.subplots(2, 4, figsize=(24, 12))
    
    axs[0, 0].plot(x, err_proprio_between_gt_and_sim[:, 0], 'o-', label="X")
    axs[0, 0].plot(x, err_proprio_between_gt_and_sim[:, 1], 'o-', label="Y")
    axs[0, 0].plot(x, err_proprio_between_gt_and_sim[:, 2], 'o-', label="Z")
    axs[0, 0].legend(loc="upper left")
    axs[0, 0].set_title('State err (gt-sim): Position')
    axs[0, 0].set(xlabel='timesteps', ylabel='meter')
    axs[0, 0].axvline(x=first_close_step, color = 'r')
    axs[0, 0].text(first_close_step+0.1,0,'sim closing gripper',rotation=90,color="r")
    axs[0, 0].axvline(x=gt_close_step, color='k')
    axs[0, 0].text(gt_close_step+0.1,0,'gt closing gripper',rotation=90,color="k")
    yticks = np.arange(-0.1, 0.1, 0.01) # unit cm
    axs[0, 0].set_yticks(yticks)

    axs[0, 1].plot(x, np.rad2deg(err_proprio_between_gt_and_sim[:, 3]), label="roll")
    axs[0, 1].plot(x, np.rad2deg(err_proprio_between_gt_and_sim[:, 4]), label="pitch")
    axs[0, 1].plot(x, np.rad2deg(err_proprio_between_gt_and_sim[:, 5]), label="yaw")
    axs[0, 1].legend(loc="upper left")
    axs[0, 1].set_title('State err (gt-sim): Rotation')
    axs[0, 1].set(xlabel='timesteps', ylabel='deg')
    axs[0, 1].axvline(x=first_close_step, color='r')
    axs[0, 1].axvline(x=gt_close_step, color='k')

    """ axs[0, 2].plot(x, err_proprio_between_gt_and_sim[:, 6])
    axs[0, 2].set_title('Action err Open/Close (gt-sim)')
    axs[0, 2].set(xlabel='timesteps', ylabel='open/close')
    axs[0, 2].axvline(x=first_close_step, color='r')
    axs[0, 2].axvline(x=gt_close_step, color='k') """

    action_dim = err_action_per_step.shape[1]
    for i in range(action_dim - 1):
        axs[0, 2].plot(x, err_action_per_step[:, i], '--', label=f"action dim {i}")
    
    # gripper dim
    max_fig_value = np.max(err_action_per_step[:, :-1])
    axs[0, 2].plot(x, err_action_per_step[:, -1] * max_fig_value, label="gripper")

    if action_token_accuracy is not None and len(action_token_accuracy) > 0:
        axs[0, 2].plot(x, np.array(action_token_accuracy)[:,0] * max_fig_value, label="fit token accuracy")
        axs[0, 2].plot(x, np.array(action_token_accuracy)[:,1] * max_fig_value, label="sim token accuracy")

    axs[0, 2].legend(loc="upper left")
    axs[0, 2].set_title(f"action errors (pred-gt)")
    axs[0, 2].set(xlabel='timesteps', ylabel='%')

    axs[0, 3].imshow(obs_sim_closing if obs_sim_closing is not None else np.zeros((480, 640, 3)))
    axs[0, 3].set_title("pred closing gripper")

    axs[1, 0].plot(x, err_ctrl_between_pred_and_proprio[:, 0], label="X")
    axs[1, 0].plot(x, err_ctrl_between_pred_and_proprio[:, 1], label="Y")
    axs[1, 0].plot(x, err_ctrl_between_pred_and_proprio[:, 2], label="Z")
    axs[1, 0].legend(loc="upper left")
    axs[1, 0].set_title('Control err (pred-proprio): Position')
    axs[1, 0].set(xlabel='timesteps', ylabel='meter')
    axs[1, 0].axvline(x=first_close_step, color='r')
    axs[1, 0].axvline(x=gt_close_step, color='k')
    yticks = np.arange(-0.001, 0.001, 0.0002)  # unit mm
    axs[1, 0].set_yticks(yticks)

    axs[1, 1].plot(x, np.rad2deg(err_ctrl_between_pred_and_proprio[:, 3]), label="roll")
    axs[1, 1].plot(x, np.rad2deg(err_ctrl_between_pred_and_proprio[:, 4]), label="pitch")
    axs[1, 1].plot(x, np.rad2deg(err_ctrl_between_pred_and_proprio[:, 5]), label="yaw")
    axs[1, 1].legend(loc="upper left")
    axs[1, 1].set_title('Control err (pred-proprio): Rotation')
    axs[1, 1].set(xlabel='timesteps', ylabel='deg')
    axs[1, 1].axvline(x=first_close_step, color='r')
    axs[1, 1].axvline(x=gt_close_step, color='k')

    if err_qpos_ctrl is not None:
        for i in range(err_qpos_ctrl.shape[1]): 
            axs[1, 2].plot(x, err_qpos_ctrl[:, i], label=f"joint_{i}")
        axs[1, 2].legend(loc="upper left")
        axs[1, 2].set_title("Control error: qpos")
        axs[1, 2].set(xlabel='timesteps', ylabel='deg')
        axs[1, 2].axvline(x=first_close_step, color='r')
        axs[1, 2].axvline(x=gt_close_step, color='k')

    axs[1, 3].imshow(obs_gt_closing if obs_gt_closing is not None else np.zeros((480, 640, 3))) # FIXME why none?
    axs[1, 3].set_title("gt closing gripper")

    fig.tight_layout(pad=2.0)
    fig.savefig(fname)


def diff_eef(pose1, pose2):
    p1, q1 = pose1[:3], pose1[3:]
    p2, q2 = pose2[:3], pose2[3:]
    diff_q = t3d.quaternions.qmult(q1, t3d.quaternions.qinverse(q2))
    return np.concatenate([(p1 - p2), t3d.euler.quat2euler(diff_q)]) 

def invert_gripper_actions(actions):
    return 1. - actions

class PolicyDebugger(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """ This wrapper is used to debug the policy by comparing the predicted actions with the ground truth actions.
        1) When set reply=True, the wrapper will replay the episode with the ground-truth actions, so that the user can
        check if the ground truth episode is correct. 
        2) When set remedy=True, the wrapper will try to remedy the policy by using the ground-truth actions to recover

        This wrapper will record the following information:
            a) gt-sim pripro diff: The error between the predicted actions and the ground-truth actions
            b) sim-sim ctrl diff: The error between the predicted actions and control results
            c) when gt and policy decides to close gripper
            d) policy token accuracy (maybe?)
    """
    def __init__(
        self,
        env: gym.Env,
        episode,
        agent: BaseAgent,
        task_id: str = "default_task",
        instruction: str = "",
        replay: bool = False,
        remedy: bool = False,
        work_dir: str = "eval",
        dataformat: str = "rlds", # h5py, rlds_numpy
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self, _disable_deepcopy=True, episode=episode, replay=replay, remedy=remedy, work_dir=work_dir
        )
        gym.Wrapper.__init__(self, env)

        self._episode = EpisodeExtractor(episode, dataformat)
        # self._steps = list(episode["steps"])
        # self._T = len(self._steps)
        self._T = self._episode.get_T()
        
        self._elapsed_steps = None
        self._agent = agent
        self._replay = replay
        # self._task_id = episode["uuid"].numpy().decode("utf-8")
        self._task_id = task_id
        
        # env_cfg = json.loads(episode['environment_config'].numpy().decode("utf-8"))
        # target_info = env_cfg["target_info"]
        # target_object_name = GraspNet_1B_Object_Names[target_info['id']]
        # self._instruction = f"Pick up {target_object_name}."
        self._instruction = instruction

        self.err_eef_proprio_per_step = None
        self.err_eef_ctrl_per_step = None
        self.err_action_per_step = None
        self.err_qpos_ctrl = None
        self.work_dir = work_dir
        self.REMEDY_ERR_BOUND_PQ = [0.02, 5] if remedy else None

        self.obs_gt_closing = None
        self.obs_sim_closing = None

        self._history_proprio = []

    def reset(self, **kwargs):
        observations, info = super().reset(**kwargs)
        self._elapsed_steps = 0

        # init_proprio_qpos = self._steps[self._elapsed_steps]["observation"]["proprio_joint_positions"].numpy()
        init_proprio_qpos = self._episode.get_proprio_joint_position(self._elapsed_steps)
        if not np.allclose(init_proprio_qpos, observations["agent"], atol=1e-2):
            logging.warning("Initial proprio qpos mismatch! ")

        img_res = observations["front_stereo_left"].shape[:2][::-1]
        if os.path.exists(f"{self.work_dir}/{self._task_id}"):
            logging.warning(f"Overwriting existing videos at {self.work_dir}/{self._task_id} folder")
            # shutil.rmtree(f"{self.work_dir}/{self._task_id}", ignore_errors=True)

        os.makedirs(f"{self.work_dir}/{self._task_id}", exist_ok=True)
        self.loaded_video_writer = VideoWriter(f"{self.work_dir}/{self._task_id}/loaded.mp4", 10, img_res, write_png=False) # FIXME: set fps
        
        """ # loaded_image = self._steps[self._elapsed_steps]["observation"]["rgb_front_stereo_left"].numpy()
        loaded_image = self._episode.get_observation_image(self._elapsed_steps, "rgb_front_stereo_left")
        resized_loaded_image = np.array(Image.fromarray(loaded_image).convert("RGB").resize(img_res), dtype=np.uint8)
        self.loaded_video_writer.write(resized_loaded_image) """

        self.first_close_step = 0
        self.gt_close_step = 0
        self.err_eef_proprio_per_step = []
        self.err_eef_ctrl_per_step = []
        self.err_action_per_step = []
        self.err_qpos_ctrl = []
        self.action_token_accuracy = [] # action token accuracy (for openvla)

        # self.err_
        return observations, info
    
    def _extract_gt_action(self, idx):
        gt_gripper = self._episode.get_action_gripper(self._elapsed_steps)
        gt_curr_proprio_eef = self._episode.get_proprio_eef_pose(self._elapsed_steps)
        gt_next_proprio_eef = self._episode.get_proprio_eef_pose(self._elapsed_steps+1)

        delta_action_xyz = gt_next_proprio_eef[:3] - gt_curr_proprio_eef[:3]
        delta_action_rpy = t3d.euler.quat2euler(t3d.quaternions.qmult(gt_next_proprio_eef[3:], t3d.quaternions.qinverse(gt_curr_proprio_eef[3:])))
        return np.concatenate([delta_action_xyz, delta_action_rpy, gt_gripper])

    def step(self, action):
        """ if self._agent.is_opening_or_closing_gripper(): # TODO test
            last_action = self._agent.get_last_qpose()
            observation, reward, terminated, truncated, info = self.env.step(action)
            return observation, reward, terminated, truncated, info """
        
        if self._elapsed_steps >= self._T-1: # T-1 actions
            # running out of ground truth steps, keep it going without interupting.
            return self.env.step(action)
        
        last_obs = self._agent.get_last_observation()
        img_res = last_obs["front_stereo_left"].shape[:2][::-1]
        pred_action = self._agent.get_last_pred_action()

        gt_action = self._extract_gt_action(self._elapsed_steps)

        # loaded_image = self._steps[self._elapsed_steps]["observation"]["rgb_front_stereo_left"].numpy()
        loaded_image = self._episode.get_observation_image(self._elapsed_steps, "rgb_front_stereo_left")
        resized_loaded_image = np.array(Image.fromarray(loaded_image).convert("RGB").resize(img_res), dtype=np.uint8)
        self.loaded_video_writer.write(resized_loaded_image)

        fitted_action, tok_acc = self._agent.query_action({
            "front_stereo_left": resized_loaded_image,
            "front_stereo_right": last_obs["front_stereo_right"],
            "wrist": last_obs["wrist"]
        }, self._instruction, gt_action=[gt_action])
        pred_action_2, tok_acc2 = self._agent.query_action(last_obs, self._instruction, gt_action=[gt_action]) # last_obs["front_stereo_left"]
        # assert np.allclose(pred_action, pred_action_2), "inconsistent prediction, check if observations are same?" FIXME (could caused by GPT sampling?)

        if tok_acc is not None:
            self.action_token_accuracy.append([tok_acc,tok_acc2]) # fitted tok acc (gt obs),  sim tok acc (sim obs)

        if len(np.array(fitted_action).shape) == 2: # FIXME could be action chunking
            fitted_action = fitted_action[0]

        if not np.allclose(pred_action, fitted_action) and self._elapsed_steps == 0: # other timestep does not make sense as obs does not align
            diff_xyz = np.linalg.norm(np.array(pred_action[:3]) - np.array(fitted_action[:3]))
            diff_rpy = np.rad2deg(np.array(pred_action[3:6]) - np.array(fitted_action[3:6]))
            logging.warning(f"diff: xyz {diff_xyz}, rpy {diff_rpy}: either rendering mismatch or policy is too fragile to image noise ?!")

        gt_gripper = gt_action[-1]
        # error between predicted action and gt action
        error_action = np.zeros(7, dtype=np.float32)
        error_action[:3] = np.array(pred_action[:3]) - gt_action[:3]
        error_action[3:6] = np.rad2deg(np.array(pred_action[3:6])  - gt_action[3:6])
        error_action[6] = pred_action[-1] - gt_gripper
        self.err_action_per_step.append(error_action)

        # record image obs when gt closes gripper
        gt_gripper_action = GripperAction(gt_gripper)
        if gt_gripper_action == GripperAction.close and self.gt_close_step == 0:
            self.gt_close_step = self._elapsed_steps
            self.obs_gt_closing = resized_loaded_image

        def hack_gripper_action(gripper_act):
            return 0 if gripper_act < 0.7 else 1
        # record image obs when policy closes gripper
        pred_gripper_action = GripperAction(hack_gripper_action(pred_action[-1]))
        if pred_gripper_action == GripperAction.close and self.first_close_step == 0:
            self.first_close_step = self._elapsed_steps
            self.obs_sim_closing = last_obs["front_stereo_left"]
        
        logging.warning(f"step {self._elapsed_steps}, {np.linalg.norm(pred_action[:3])}, {pred_gripper_action.name}")

        err_gripper = pred_action[-1] - gt_gripper # check if close gripper at the right step
        gt_next_proprio_eef = self._episode.get_proprio_eef_pose(self._elapsed_steps+1)
        err_proprio = diff_eef(self._agent.get_last_pred_eef_pose(), gt_next_proprio_eef[-7:])
        self.err_eef_proprio_per_step.append(np.concatenate([err_proprio, [err_gripper]])) # error between current simu and dataset proprio

        if self._replay or (
                    self.REMEDY_ERR_BOUND_PQ is not None and \
                    (
                        np.linalg.norm(err_proprio[:3]) > self.REMEDY_ERR_BOUND_PQ[0] or \
                        np.max(np.rad2deg(err_proprio[3:])) > self.REMEDY_ERR_BOUND_PQ[1]
                    )
            ):
            # logging.warning(f"model derailed: {err_proprio} at step {self._elapsed_steps}, remedy now")
            # p1, q1 = np.split(gt_next_proprio_eef[-7:], [3])
            # p, q = get_hand_pose_from_eef_pose(p1, q1)
            # try:
            #     target_qpos = ik(self._agent.ik_solver, p, q, self._agent.get_last_qpose()[:7]) # loaded_proprio_qpos
            #     action[:7] = target_qpos
            #     action[7] = gt_gripper[0]
            #     """ if gt_gripper_action == GripperAction.close:
            #         self._agent._gripper_action_tick = 0
            #         self._agent._gripper_state = GripperState.closing
            #         self._agent._last_qpos = target_qpos[:7]
            #         self._agent._target_qpos = target_qpos[:7] """
            #     print(self._elapsed_steps, gt_gripper_action)
            # except:
            #     logging.warning("remedy failed at ik, fallback")

            # override
            next_proprio_qpos = self._episode.get_proprio_joint_position(self._elapsed_steps+1)
            action[:7] = next_proprio_qpos[:7]
            action[7] = 1. if next_proprio_qpos[8] > 0.03 else 0.

            p1, q1 = np.split(gt_next_proprio_eef, [3,])
        else:
            p1, q1 = np.split(self._agent.get_last_pred_eef_pose(), [3,])
            # pass # TODO
        
        observation, reward, terminated, truncated, info = self.env.step(action)
        # print(self._elapsed_steps, observation["agent"][-1])
        
        # current proprio from simulator
        simu_proprio_qpos = observation['agent']
        p, q = self.env.unwrapped.task.robot.fk(simu_proprio_qpos[:7]) # convert to eef pose
        p0, q0 = self.env.unwrapped.task.robot.get_eef_pose_from_hand_pose(p, q)
        simu_proprio_eef = np.concatenate([p0, q0])

        err_eef_ctrl = diff_eef(np.concatenate([p1, q1]), simu_proprio_eef)
        self.err_eef_ctrl_per_step.append(err_eef_ctrl)

        err_qpos = np.rad2deg(observation['agent'][:7] - action[:7])
        # tqdm.write(f"err_ctrl: {err_qpos}")
        self.err_qpos_ctrl.append(err_qpos)

        """ if self._elapsed_steps == self._T - 1: # because last action is meaningless in RLDS
            truncated = True
            self._success = False
        else:
            self._success = terminated """
        
        self._elapsed_steps += 1
        return observation, reward, terminated, truncated, info
    
    def close(self):
        self.env.close()

        success = self.unwrapped._success if hasattr(self.unwrapped, "_success") else False
        self.loaded_video_writer.release(success)

        if len(self.action_token_accuracy) > 0:
            logging.warning(f"Average action token accuracy: {np.mean(np.array(self.action_token_accuracy), axis=0)}") # N,2

        plot_err(self.err_eef_proprio_per_step, self.err_eef_ctrl_per_step, self.err_action_per_step, 
                 f"{self.work_dir}/{self._task_id}/chkplc_plot_err_{success}.png", 
                self.first_close_step, self.gt_close_step, 
                self.err_qpos_ctrl, 
                self.obs_gt_closing, 
                self.obs_sim_closing,
                self.action_token_accuracy)
