"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import os
import json
import numpy as np
import torch
import tyro
import cv2
from pathlib import Path
from queue import Queue
from typing import Optional, Dict, Any
import collections
import transforms3d as t3d

from simple.agents.base_agent import BaseAgent, GripperAction
from simple.baselines.dp import DiffusionPolicyModel
from simple.core.action import ActionCmd
from simple.robots.protocols import HasParallelGripper

def hack_gripper_action(gripper_act):
    return 0 if gripper_act < 0.7 else 1


def normalize_data(data: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Normalize data to [-1, 1] range using provided statistics."""
    # normalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Unnormalize data from [-1, 1] range back to original range."""
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


class DiffusionPolicyAgent(BaseAgent):
    def __init__(
        self, 
        robot,
        model_path: str = "checkpoints",
        obs_horizon: int = 2,
        action_horizon: int = 8,
        pred_horizon: int = 16,
        num_cameras: int = 3,
        action_dim: int = 7,
        share_vision_encoder: bool = True,
        device: Optional[str] = None,
        use_ema: bool = True,
    ):
        super().__init__(robot)
        
        self.model_path = Path(model_path)
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.pred_horizon = pred_horizon
        self.use_ema = use_ema
        self.num_cameras = num_cameras
        self.action_dim = action_dim
        self.share_vision_encoder = share_vision_encoder
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        config_path = os.path.join(model_path, "run_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, "r") as f:
            config_data = json.load(f)
        
        self.obs_horizon = config_data.get("data", {}).get("obs_horizon", 2)
        self.pred_horizon = config_data.get("model", {}).get("pred_horizon", 16)
        self.action_horizon = config_data.get("data", {}).get("action_horizon", 8)
        self.action_dim = config_data.get("data", {}).get("action_dim", 7)
        self.state_dim = config_data.get("data", {}).get("state_dim", 7)
        self.num_diffusion_iters = config_data.get("model", {}).get("num_diffusion_iters", 100)
        self.image_size = config_data.get("data", {}).get("transform", {}).get("model", {}).get("resize", {}).get("size", [256, 480])
        
        print(f"Model config loaded:")
        print(f"  obs_horizon: {self.obs_horizon}")
        print(f"  pred_horizon: {self.pred_horizon}")
        print(f"  action_horizon: {self.action_horizon}")
        print(f"  action_dim: {self.action_dim}")
        print(f"  state_dim: {self.state_dim}")
        print(f"  num_diffusion_iters: {self.num_diffusion_iters}")

        self._load_model()
        self.obs_deque = collections.deque(maxlen=self.obs_horizon)
        self.action_buffer = Queue()
        self.ik_failed_cnt = 0
        
        stats_path = os.path.join(model_path, "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                raw_stats = json.load(f)
                
                # Convert keys：observation.state -> agent_pos
                self.stats = {}
                for key in raw_stats:
                    if key == "observation.state" or key == "states":
                        self.stats["agent_pos"] = raw_stats[key]
                    else:
                        self.stats[key] = raw_stats[key]
                
                # Convert lists to numpy arrays
                for key in self.stats:
                    for stat_type in self.stats[key]:
                        self.stats[key][stat_type] = np.array(self.stats[key][stat_type])
                
                print(f"Stats loaded with keys: {list(self.stats.keys())}")
        else:
            print(f"Warning: Stats file not found at {stats_path}. You'll need to provide stats manually.")
            self.stats = None
            
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Model architecture:")
        print(f"  Input: obs_horizon={self.obs_horizon}, lowdim_obs_dim={self.state_dim}")
        print(f"  Output: pred_horizon={self.pred_horizon}, action_dim={self.action_dim}")
    
    def _load_model(self):
        self.model = DiffusionPolicyModel(
            lowdim_obs_dim=self.state_dim,
            obs_horizon=self.obs_horizon,
            action_dim=self.action_dim,
            num_diffusion_iters=self.num_diffusion_iters,
            num_cameras=self.num_cameras,
            share_vision_encoder=self.share_vision_encoder,
        )
        
        checkpoint_path = os.path.join(
            self.model_path, 
            f"ckpt_9899/ema_net.pth"
        )
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
    
    
    def _process_observation(self, observation: Dict[str, Any]) -> Dict[str, np.ndarray]:
        processed_obs = {}

        image_keys = ['front_stereo_right', 'wrist', 'side_left']
        image_list = []
        
        for k in image_keys:
            if k in observation:
                img = observation[k]
                img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                image_list.append(img)
        
        if image_list:
            processed_obs['image'] = np.concatenate(image_list, axis=0)

        agent_pos = None
        if 'eef_pose' in observation:
            agent_pos = np.array(observation['eef_pose'])
            if 'joint_qpos' in observation:
                # Concatenate gripper value (last dimension of joint_qpos)
                gripper = np.array(observation['joint_qpos'])[-1:]
                agent_pos = np.concatenate([agent_pos, gripper])
        elif 'joint_qpos' in observation:
            agent_pos = np.array(observation['joint_qpos'])
        
        if agent_pos is not None:
            if self.stats is not None and 'agent_pos' in self.stats:
                processed_obs['agent_pos'] = normalize_data(agent_pos, self.stats['agent_pos'])
            else:
                processed_obs['agent_pos'] = agent_pos
        
        return processed_obs

    def _stack_observations(self) -> Dict[str, np.ndarray]:
        """Stack observations from the deque"""
        stacked = {}

        obs_list = list(self.obs_deque)
        
        while len(obs_list) < self.obs_horizon:
            obs_list.insert(0, obs_list[0] if obs_list else {})
        
        for key in obs_list[0].keys():
            stacked[key] = np.stack([obs[key] for obs in obs_list])
        
        return stacked
    
    def get_action(
        self, 
        observation: Dict[str, Any], 
        instruction: Optional[str] = None,
        info: Optional[Dict] = None,
        **kwargs
    ) -> np.ndarray:
        self._last_observation = observation
        
        processed_obs = self._process_observation(observation)
        
        self.obs_deque.append(processed_obs)
        
        if self.action_buffer.qsize() == 0:
            stacked_obs = self._stack_observations()
            device = self.device
            nimages = torch.from_numpy(stacked_obs['image']).to(device, dtype=torch.float32)
            nagent_poses = torch.from_numpy(stacked_obs['agent_pos']).to(device, dtype=torch.float32)

            # (obs_horizon, num_cameras*C, H, W) -> (obs_horizon*num_cameras, C, H, W)
            nimages = nimages.view(-1, 3, nimages.shape[2], nimages.shape[3])

            with torch.no_grad():
                pred_actions = self.model.sample_actions(
                    nimages=nimages,
                    nagent_poses=nagent_poses,
                    device=self.device
                )
            
            if self.stats is not None and 'action' in self.stats:
                pred_actions = unnormalize_data(pred_actions[0], self.stats['action'])
            else:
                pred_actions = pred_actions[0]
                
            print(f"Generated {pred_actions.shape[0]} actions")
            
            start = self.obs_horizon - 1
            end = start + self.action_horizon
            actions_to_execute = pred_actions[start:end]
            
            for action in actions_to_execute:
                self.action_buffer.put(action)
        
        pred_action = self.action_buffer.get()
        self._last_pred_action = pred_action

        # get current qpos 
        curr_qpos = observation["agent"]

        assert isinstance(self.robot, HasParallelGripper), "Robot must have parallel gripper for OpenVLAAgent."

        # convert to eef pose
        p, q = self.robot.fk(curr_qpos[:7])
        p0, q0 = self.robot.get_eef_pose_from_hand_pose(p, q)

        # apply delta action
        delta_q = t3d.euler.euler2quat(*pred_action[3:6])
        q1 =  t3d.quaternions.qmult(delta_q, q0)
        p1 = p0 + pred_action[:3]
        self._last_pred_eef_pose = np.concatenate([p1, q1])

        p, q = self.robot.get_hand_pose_from_eef_pose(p1, q1)
        # ik to get target qpos
        try:
            # Ensure current_joint matches the DOF expected by the IK solver (usually 7 or 9)
            ik_dof = len(self.robot.kin_model.joint_names)
            target_qpos = self.robot.ik(p, q, curr_qpos[:ik_dof])
            gripper_state = "open_eef" if hack_gripper_action(pred_action[-1]) == 1 else "close_eef"
            self.ik_failed_cnt = 0 # reset
        except Exception as e:
            self.ik_failed_cnt += 1
            print(f"ik failed={self.ik_failed_cnt}, keep still, {e}")
            target_qpos = curr_qpos[:7]
            gripper_state = "keep"
            # return np.concatenate([curr_qpos, [GripperAction.keep]])

        self._last_qpos = curr_qpos

        # gripper_action = GripperAction(hack_gripper_action(pred_action[-1]))
        # return np.concatenate([target_qpos, [gripper_action]])
        arm_dim = len(target_qpos)
        return ActionCmd("move_qpos_with_eef", 
            target_qpos=dict(zip(self.robot.joint_names[:arm_dim], target_qpos)),
            eef_state=gripper_state
        )
    
    def reset(self):
        super().reset()
        self.obs_deque.clear()
        self.action_buffer = Queue()

