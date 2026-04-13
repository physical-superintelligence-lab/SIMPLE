"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import numpy as np

def invert_gripper_actions(actions):
    return 1. - actions

class EpisodeExtractor():
    """ extract episode data from datasets
        It's an  adapter for replay episode fron different data formats (h5py / rlds / etc.) 
    """
    def __init__(self, episode, format = "rlds"):
        self.format = format
        self.episode = episode
        assert self.format in ["h5py", "rlds", "rlds_numpy","lerobot", "rlds-legacy"], "unsupported format"

    def get_T(self) -> int:
        if self.format == "h5py":
            return self.episode["/action"].shape[0]
        elif self.format == "rlds" or self.format == "rlds-legacy":
            # exclude last step in RLDS, for unknown reason
            # rlds episode has one extra step with all zeros
            return len(list(self.episode["steps"])) -1 
        elif self.format == "rlds_numpy":
            return self.episode["action"].shape[0]
        elif self.format == "lerobot":
            return len(self.episode)

        return 0

    def get_proprio_actuators_action(self, idx):
        #FIXME Now it's only for g1
        if self.format == "h5py":
            return self.episode["/observations/actions"][idx]
        elif self.format == "rlds" or self.format == "rlds-legacy":
            return list(self.episode["steps"])[idx]["observation"]["proprio_actions"].numpy()
        elif self.format == "rlds_numpy":
            return self.episode["observation"]["proprio"][idx, 0, 8:]
        elif self.format == "lerobot":
            return self.episode[idx]["action"].numpy()
        raise ValueError
    
    def get_amo_policy_command(self, idx):
        if self.format == "h5py":
            return self.episode["/observations/amo_policy_command"][idx]
        elif self.format == "rlds" or self.format == "rlds-legacy":
            return list(self.episode["steps"])[idx]["observation"]["amo_policy_command"].numpy()
        elif self.format == "rlds_numpy":
            return self.episode["observation"]["amo_policy_command"][idx, 0, 8:]
        elif self.format == "lerobot":
            return self.episode[idx]["observation.amo_policy_command"].numpy()
        raise ValueError
    
    def get_amo_policy_dyaw(self, idx):
        if self.format == "h5py":
            return self.episode["/observations/amo_policy_obs_dyaw"][idx]
        elif self.format == "rlds" or self.format == "rlds-legacy":
            return list(self.episode["steps"])[idx]["observation"]["amo_policy_obs_dyaw"].numpy()
        elif self.format == "rlds_numpy":
            return self.episode["observation"]["amo_policy_obs_dyaw"][idx, 0, 8:]
        elif self.format == "lerobot":
            return self.episode[idx]["observation.amo_policy_dyaw"].numpy()
        raise ValueError
    def get_amo_policy_target_yaw(self, idx):
        if self.format == "h5py":
            return self.episode["/observations/amo_policy_obs_dyaw"][idx]
        elif self.format == "rlds" or self.format == "rlds-legacy":
            return list(self.episode["steps"])[idx]["observation"]["amo_policy_obs_target_yaw"].numpy()
        elif self.format == "rlds_numpy":
            return self.episode["observation"]["amo_policy_obs_dyaw"][idx, 0, 8:]
        elif self.format == "lerobot":
            try:
                return self.episode[idx]["observation.amo_policy_target_yaw"].numpy()
            except:
                return self.episode[idx]["action"][-1].numpy()
        raise ValueError



    def get_proprio_joint_position(self, idx):
        if self.format == "h5py":
            return self.episode["/observations/qpos"][idx]
        elif self.format == "rlds" or self.format == "rlds-legacy":
            return list(self.episode["steps"])[idx]["observation"]["proprio_joint_positions"].numpy()
        elif self.format == "rlds_numpy":
            qpos = self.episode["observation"]["proprio"][idx][0, :7]
            # this value is incorrect sometimes, due to the binarization
            gripper = 0.04 if self.episode["observation"]["proprio"][idx, 0, 7] == 1. else 0. 
            return np.concatenate([qpos, [gripper, gripper]])
        elif self.format == "lerobot":
            return self.episode[idx]["observation.proprio_joint_positions"].numpy()
        raise ValueError

    def get_observation_image(self, idx, cam=None):
        if self.format == "h5py":
            return self.episode["/observations/images"]["front-stereo-left"][idx]
        elif self.format == "rlds" or self.format == "rlds-legacy":
            return list(self.episode["steps"])[idx]["observation"]["rgb_front_stereo_left"].numpy()
        elif self.format == "rlds_numpy":
            return self.episode["observation"]["image_camera_0"][idx,0]
        elif self.format == "lerobot":
            return np.transpose((self.episode[idx]["observation.rgb_front_stereo_left"].numpy()*255).astype(np.uint8),(1,2,0))
        # return None
        raise ValueError
    
    def get_action_gripper(self, idx):
        if self.format == "h5py":
            return self.episode["/action"][idx][-1: ] # already inverted
        elif self.format == "rlds" or self.format == "rlds-legacy":
            return invert_gripper_actions(list(self.episode["steps"])[idx]["action"]["gripper"].numpy())
        elif self.format == "rlds_numpy":
            return self.episode["action"][idx, 0, 6:7]
        elif self.format == "lerobot":
            gripper_state = self.episode[idx]["action.gripper_state"].numpy()
            return invert_gripper_actions(gripper_state)
        raise ValueError
    
    def get_proprio_eef_pose(self, idx):
        if self.format == "h5py":
            return self.episode["/observations/eef_pose"][idx]
        elif self.format == "rlds" or self.format == "rlds-legacy":
            return list(self.episode["steps"])[idx]["observation"]["proprio_eef_pose"].numpy()
        elif self.format == "rlds_numpy":
            return self.episode["observation"]["proprio"][idx, 0, -7:]
        elif self.format == "lerobot":
            return self.episode[idx]["observation.proprio_eef_pose"].numpy()
        return None
    def get_prev_height(self, idx):
        if self.format == "h5py":
            return self.episode["/observations/prev_height"][idx]
        elif self.format == "rlds" or self.format == "rlds-legacy":
            return list(self.episode["steps"])[idx]["observation"]["prev_height"].numpy()
        elif self.format == "rlds_numpy":
            return self.episode["observation"]["prev_height"][idx, 0, 8:]
        elif self.format == "lerobot":
            return self.episode[idx]["observation.prev_height"].numpy()
    def get_prev_torso_rpy(self, idx):
        if self.format == "h5py":
            return self.episode["/observations/prev_torso_rpy"][idx]
        elif self.format == "rlds" or self.format == "rlds-legacy":
            return list(self.episode["steps"])[idx]["observation"]["prev_torso_rpy"].numpy()
        elif self.format == "rlds_numpy":
            return self.episode["observation"]["prev_torso_rpy"][idx, 0, 8:]
        elif self.format == "lerobot":
            return self.episode[idx]["observation.prev_torso_rpy"].numpy()

    def get_postprocess_action(self, idx):
        if self.format == "h5py":
            return self.episode["/action"][idx][-1: ] # already inverted
        elif self.format == "rlds" or self.format == "rlds-legacy":
            return list(self.episode["steps"])[idx]["action"]["postprocess"].numpy()
        elif self.format == "rlds_numpy":
            return self.episode["action"][idx, 0, 7:8]
        elif self.format == "lerobot":
            return self.episode[idx]["action"].numpy()
        raise ValueError
    def get_postprocess_states(self, idx):
        if self.format == "h5py":
            return self.episode["/observations/postprocess"][idx]
        elif self.format == "rlds" or self.format == "rlds-legacy":
            return list(self.episode["steps"])[idx]["observation"]["postprocess"].numpy()
        elif self.format == "rlds_numpy":
            return self.episode["observation"]["postprocess"][idx, 0, 8:]
        elif self.format == "lerobot":
            return self.episode[idx]["states"].numpy()
        raise ValueError
    def __len__(self) -> int:
        return self.get_T()