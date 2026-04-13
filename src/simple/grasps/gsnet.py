"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import numpy as np
import transforms3d as t3d
from typing import Optional
from simple.core.asset import Asset
from simple.core.object import SpatialAnnotated
from simple.core.types import Pose

class GSNet:

    def __init__(self):
        pass


    def detect_grasps(self, obj):
        pass

    @classmethod
    def load_cached_grasps(
        cls, 
        asset: SpatialAnnotated, 
        stable_idx: Optional[int] = None, 
        target_pose: Optional[Pose] = None,
        max_grasps: Optional[int] = None,
        bias: Optional[str] = None  # "outward" to prefer grasps far from object center
    ) -> list:
        """  Load cached grasps from the asset based on current object pose.
        Args:
            asset (SpatialAnnotated): The asset containing stable poses and grasps.

        Returns:
            List[Tuple[Pose, GraspPose]]: A list of tuples, each containing
        """
        # results = dict()
        stable_poses = asset.stable_poses
        grasps = asset.canonical_grasps
        
        if stable_idx is None:
            assert target_pose is not None, "Either stable_idx or stable_pose must be provided."
            for index in range(len(stable_poses)):
                pose = stable_poses[index]
                if np.allclose(
                    np.array(pose[2:3]), 
                    np.array(target_pose.position[2:3]), 
                    rtol=0, atol=1e-8
                ): # FIXME
                    stable_idx = index # type: ignore
                    break

        else:
            assert target_pose is not None, "Either stable_idx or stable_pose must be provided."

                # and np.allclose(
                #     np.array(pose.quaternion), 
                #     np.array(stable_pose.quaternion), 
                #     rtol=0, atol=1e-8
                # )
        assert stable_idx is not None, "No matching stable pose found."

        grasp_infos = grasps[stable_idx]
        if len(grasp_infos) == 0:
            return None

        # grasp filtering
        grasp_depths = np.array([grasp_info[1] for grasp_info in grasp_infos], dtype=np.float32) 
        T_object_grasp = np.stack([grasp_info[0] for grasp_info in grasp_infos], axis=0)

        T_world_object = np.eye(4)
        rand_target_ori_mat = t3d.quaternions.quat2mat(target_pose.quaternion)

        T_world_object[:3, :3] = rand_target_ori_mat
        T_world_object[:3, 3] = target_pose.position
        T_world_grasp = T_world_object @ T_object_grasp


        # the x-axis unit vector should roughly point to the ground
        x_rotated = T_world_grasp[:, :3, 0]
        # find the angle between x_rotated and the -z axis
        angles = np.arccos(np.dot(x_rotated, np.array([0, 0, -1])))
        valid_grasp_idxs = np.where(angles < 20 / 180 * np.pi)[0]
        
        if len(valid_grasp_idxs) == 0:
            # hack when no ideal valid poses are found
            valid_grasp_idxs = np.arange(len(T_world_grasp))

        if bias == "outward" and len(valid_grasp_idxs) > 0:

            T_object_grasp_valid = T_object_grasp[valid_grasp_idxs]
            grasp_positions_obj = T_object_grasp_valid[:, :3, 3]  # N x 3
            
            distances_from_center = np.linalg.norm(grasp_positions_obj, axis=1)
            sorted_by_distance = np.argsort(-distances_from_center)
            valid_grasp_idxs = valid_grasp_idxs[sorted_by_distance]

        # target_id=target_info["id"]
        # target_pose_a = {
            # "target_info": target_info,
            # grasp_poses = [{
            #     'position': T_world_grasp[idx, :3, 3],
            #     'orientation': t3d.quaternions.mat2quat(T_world_grasp[idx, :3, :3]),
            #     'width': grasp_infos[idx][1],
            #     'depth': grasp_depths[idx], #0.005,
            #     'score': 1,
            #     # 'object_id': target_id,
            #     'grasp_id': idx,
            # } for idx in valid_grasp_idxs]
        # }

        ### -debug- ###
        if max_grasps is not None:
            valid_grasp_idxs = valid_grasp_idxs[:max_grasps] 
        ###############

        grasp_poses = [{
            'position': T_world_grasp[idx, :3, 3], # FIXME 
            'orientation': t3d.quaternions.mat2quat(T_world_grasp[idx, :3, :3]),
            'width': grasp_infos[idx][1],
            'depth': grasp_depths[idx], #0.005,
            'score': 1,
            # 'object_id': target_id,
            'grasp_id': idx,
            'stable_idx': stable_idx,
        } for idx in valid_grasp_idxs]

        return grasp_poses
    
    @classmethod
    def load_cached_grasps_filter_by_negative_x(
        cls, 
        asset: SpatialAnnotated, 
        stable_idx: int = 0, 
        target_pose: Optional[Pose] = None,
        max_grasps: Optional[int] = None,
        angle_threshold: float = 20.0,
        reference_grasps: Optional[list] = None
    ) -> list:
        """Load cached grasps filtered by X-axis pointing towards -X direction (world frame).
        
        Args:
            reference_grasps: If provided, prefer grasps far from these reference grasps (in object frame)
        """
        grasps = asset.canonical_grasps

        grasp_infos = grasps[stable_idx]

        # grasp filtering
        grasp_depths = np.array([grasp_info[1] for grasp_info in grasp_infos], dtype=np.float32) 
        T_object_grasp = np.stack([grasp_info[0] for grasp_info in grasp_infos], axis=0)

        T_world_object = np.eye(4)
        rand_target_ori_mat = t3d.quaternions.quat2mat(target_pose.quaternion)
        T_world_object[:3, :3] = rand_target_ori_mat
        T_world_object[:3, 3] = target_pose.position
        T_world_grasp = T_world_object @ T_object_grasp

        # the x-axis unit vector should roughly point to the -x direction (world frame)
        x_rotated = T_world_grasp[:, :3, 0]
        # find the angle between x_rotated and the -x axis
        angles = np.arccos(np.clip(np.dot(x_rotated, np.array([-1, 0, 0])), -1.0, 1.0))
        valid_grasp_idxs = np.where(angles < angle_threshold / 180 * np.pi)[0]
        
        if len(valid_grasp_idxs) == 0:
            # If no grasps meet threshold, relax to 30 degrees
            valid_grasp_idxs = np.where(angles < 30 / 180 * np.pi)[0]
            print(f"[GSNet] No grasps meet {angle_threshold}° threshold, relaxed to 30°, found {len(valid_grasp_idxs)} grasps")
            
        if len(valid_grasp_idxs) == 0:
            # If still no valid grasps, take the best ones (smallest angles)
            valid_grasp_idxs = np.argsort(angles)[:max(5, len(angles) // 4)]
            print(f"still no valid grasps, take the best ones")

        if reference_grasps is not None and len(reference_grasps) > 0 and len(valid_grasp_idxs) > 0:

            T_world_object_mat = np.eye(4)
            T_world_object_mat[:3, :3] = rand_target_ori_mat
            T_world_object_mat[:3, 3] = target_pose.position
            T_object_world = np.linalg.inv(T_world_object_mat)
            
            reference_positions_obj = []
            for ref_grasp in reference_grasps:
                ref_pos_world = np.array(ref_grasp["position"])
                ref_pos_homogeneous = np.append(ref_pos_world, 1.0)
                ref_pos_obj = (T_object_world @ ref_pos_homogeneous)[:3]
                reference_positions_obj.append(ref_pos_obj)
            
            reference_positions_obj = np.array(reference_positions_obj)  # M x 3
            
            T_object_grasp_valid = T_object_grasp[valid_grasp_idxs]
            candidate_positions_obj = T_object_grasp_valid[:, :3, 3]  # N x 3
            
            min_distances = []
            for cand_pos in candidate_positions_obj:
                distances = np.linalg.norm(reference_positions_obj - cand_pos, axis=1)
                min_dist = np.min(distances)
                min_distances.append(min_dist)
            
            min_distances = np.array(min_distances)
            sorted_by_distance = np.argsort(-min_distances)
            valid_grasp_idxs = valid_grasp_idxs[sorted_by_distance]

        ### -debug- ###
        if max_grasps is not None:
            # Sort by angle (best alignment first) and take top max_grasps
            sorted_valid_idxs = valid_grasp_idxs[np.argsort(angles[valid_grasp_idxs])]
            valid_grasp_idxs = sorted_valid_idxs[:max_grasps]
        ###############

        grasp_poses = [{
            'position': T_world_grasp[idx, :3, 3], # FIXME 
            'orientation': t3d.quaternions.mat2quat(T_world_grasp[idx, :3, :3]),
            'width': grasp_infos[idx][1],
            'depth': grasp_depths[idx], #0.005,
            'score': 1,
            # 'object_id': target_id,
            'grasp_id': idx,
            'stable_idx': stable_idx,
        } for idx in valid_grasp_idxs]

        return grasp_poses