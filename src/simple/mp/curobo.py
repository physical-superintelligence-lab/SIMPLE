"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any

from simple.robots.protocols import BatchPlannable
if TYPE_CHECKING:
    from simple.core.robot import Robot
    from simple.core.task import Task
import os
import copy
import random
import torch
import numpy as np
import transforms3d as t3d

try: 
    import curobo
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.types.base import TensorDeviceType
    from curobo.geom.types import WorldConfig, Mesh, Cuboid
    from curobo.wrap.reacher.ik_solver import IKSolver
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, PoseCostMetric, MotionGenResult
    from curobo.types.robot import RobotConfig
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.util.logger import setup_curobo_logger
    from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml

    setup_curobo_logger("error")
except:
    raise ImportError("CuRobo is not installed properly. Please run `./scripts/install_curobo.sh` or follow the more detailed instructions in tutorials/installation")

from .planner import MotionPlanner

from simple.core.layout import Layout
from simple.core.actor import ObjectActor
from simple.utils import Timer
from simple.constants import *
from simple.assets.asset_manager import AssetManager
from simple.assets.primitive import Box as BoxActor

class CuRoboPlanner(MotionPlanner):
    plan_batch_size: int

    def __init__(
        self, 
        robot: BatchPlannable,
        plan_batch_size=40,
        plan_dt=0.10,
        approach_duration=5,
        pregrasp_duration=2,
        pregrasp_distance=[0.05, 0.08],
        lift_height=0.15,
        plan_per_traj = 4,
        # num_obstacles=3,
        easy_motion_gen=True,
        ignore_target_collisions=False,
    ):
        super().__init__()
        self.plan_batch_size = plan_batch_size
        # self.robot = robot

        self.plan_per_traj = plan_per_traj
        self.plan_dt = plan_dt
        self.approach_duration = approach_duration
        self.pregrasp_duration = pregrasp_duration
        self.pregrasp_distance = pregrasp_distance
        self.lift_height = lift_height
        # robot_cfg = self.robot_cfg = self.robot.robot_cfg
        self._init_robot_related(robot)
        self.easy_motion_gen = easy_motion_gen
        self.ignore_target_collisions = ignore_target_collisions
        
    def _init_robot_related(self, robot: BatchPlannable):
        self.robot = robot
        robot_cfg = self.robot_cfg = self.robot.robot_cfg

        self.init_js_ik_solver = IKSolver(IKSolver.load_from_robot_config(
            self.robot_cfg,
            # world_cfg,
            tensor_args = TensorDeviceType(),
        ))

        empty_world = WorldConfig()

        self.motion_gen = MotionGen(MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_model=empty_world, #world_cfg_table,
           
            interpolation_dt=self.plan_dt,
            interpolation_steps=1000,
            collision_cache={"obb": 10, "mesh": 10},

            collision_activation_distance=torch.tensor([0.01], device='cuda', dtype=torch.float32),
            ik_opt_iters=200,
            grad_trajopt_iters=200,
            use_cuda_graph=False,
        ))

        self.lift_ik_solver = IKSolver(IKSolver.load_from_robot_config(
            robot_cfg,
            # world_cfg, # no world, so it's easy for ik solver to find solution
            tensor_args=TensorDeviceType(),
            position_threshold=0.001,
            rotation_threshold=0.01,
            num_seeds=2,
            self_collision_check=False,
            self_collision_opt=False,
            use_cuda_graph=True,
            regularization=True
        ))
        
        if type(robot_cfg) == str: # a little hacky
            robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_cfg))["robot_cfg"] # FIX_ROBOT

        urdf_file = robot_cfg["kinematics"]["urdf_path"]  # Send global path starting with "/"
        base_link = robot_cfg["kinematics"]["base_link"]
        ee_link = robot_cfg["kinematics"]["ee_link"]
        if ("external_asset_path" in robot_cfg["kinematics"] and 
            robot_cfg["kinematics"]["external_asset_path"] is not None):
            urdf_file = join_path(robot_cfg["kinematics"]["external_asset_path"], urdf_file)
        robot_config = RobotConfig.from_basic(urdf_file, base_link, ee_link, TensorDeviceType())
        self.kin_model = CudaRobotModel(robot_config.kinematics)

    def reinit(self, robot: BatchPlannable):
        self._init_robot_related(robot)
 
    def batch_plan_for_approach_bodex(
        self, 
        goal_poses, 
        grasp_qpos, #: dict[str, float], 
        squeeze_qpos, #: dict[str, float], 
        lift_qpos, #: dict[str, float],
        **kwargs
    ) -> tuple[list, list]:
        timer = Timer()

        world_layout = kwargs.pop("world_layout", None)
        if not self.easy_motion_gen and world_layout is not None:
            world_cfg = self.create_collision_world_cfg(world_layout)
            self.motion_gen.update_world(world_cfg)

        debug_grasp = kwargs.pop("debug_grasp", None)
        # for i in range(num_waypoints):
        with timer(f"approach"):
            tensor_args = TensorDeviceType()

            goal_ee_pose = self._stack_curobo_type(
                Pose(
                    position=torch.tensor(p[:3], device='cuda', dtype=torch.float32), 
                    quaternion=torch.tensor(p[3:], device='cuda', dtype=torch.float32)
                ) 
                for p in goal_poses
            )

            link_poses= kwargs.pop("link_poses", None) 
            # TODO add link pose, make waist joint fixed
            current_qpos = kwargs.pop("current_joint_qpos", None)

            if current_qpos is None:
                init_qpos = torch.zeros(len(self.motion_gen.joint_names))
            else:
                joint_names = self.init_js_ik_solver.joint_names
                q_list = [current_qpos[name] for name in joint_names]  
                init_qpos = torch.tensor(q_list, dtype=torch.float32)
            lock_links_names = kwargs.pop("lock_links_names", None)
            if lock_links_names is not None:
                all_origin_link_poses = self.init_js_ik_solver.fk(init_qpos).link_poses
                lock_links_poses = {name: all_origin_link_poses[name] for name in lock_links_names}
                link_poses.update(lock_links_poses)

            # DEBUG 
            # init_qpos = torch.zeros(len(self.motion_gen.joint_names))


            batch_size=goal_ee_pose.batch
            init_state=JointState.from_position(
                tensor_args.to_device(init_qpos.view(1,-1)).repeat(batch_size, 1),
                joint_names=self.motion_gen.joint_names
            )

            mogen_result = self.motion_gen.plan_batch(
                init_state,
                goal_ee_pose,
                # debug_grasp["kin_state"].ee_pose,
                link_poses=link_poses,
                plan_config=MotionGenPlanConfig(
                    enable_finetune_trajopt=True, num_trajopt_seeds=12, max_attempts=100
                ),
            )

            if torch.sum(mogen_result.success) > 0:
                jNames = mogen_result.optimized_plan.joint_names

                if batch_size > 1:
                    ...
                else:
                    mogen_result_traj = mogen_result.interpolated_plan.trim_trajectory(
                        0, mogen_result.path_buffer_last_tstep[0]
                    ).position.cpu().numpy()
                    
                    ee_wrist_yaw_idx = next(
                        (i for i, n in enumerate(self.motion_gen.joint_names) if n.endswith("wrist_yaw_joint")),
                        None
                    )

                    if ee_wrist_yaw_idx is None:
                        raise ValueError("Cannot find ee_wrist_yaw_joint in joint_names!")

                    ee_hand_start_idx = ee_wrist_yaw_idx + 1
                    
                    hand_dof = self.robot.hand_dof

                    # mogen_result_traj[:,-hand_dof:] = init_qpos[-hand_dof:]

                    # squeeze_dir = grasp_qpos[0][0][-hand_dof:] - mogen_result_traj[-1][ee_hand_start_idx:ee_hand_start_idx+hand_dof]
                    # FIXME assume dex3-1 hand
                    # print(self.motion_gen.joint_names[10:17])
                    # 'right_hand_index_0_joint', 
                    # 'right_hand_index_1_joint', 
                    # 'right_hand_middle_0_joint', 
                    # 'right_hand_middle_1_joint', 
                    # 'right_hand_thumb_0_joint', 
                    # 'right_hand_thumb_1_joint', 
                    # 'right_hand_thumb_2_joint'
                    if "right" in self.kin_model.ee_link:
                        squeeze_dir = np.array([1,1,1,1,0,-1,-1], dtype=np.float32) # only close the fingers, not the wrist
                    
                    else:
                        squeeze_dir = np.array([-1,-1,-1,-1,0,1,1], dtype=np.float32) # only close the fingers, not the wrist
                    squeeze_dir[4] = (grasp_qpos[0][0][-hand_dof:] - mogen_result_traj[-1][ee_hand_start_idx:ee_hand_start_idx+hand_dof])[4]
                        

                    # squeeze_dir = np.sign(
                    #     grasp_qpos[0][0][-hand_dof:] - mogen_result_traj[-1][ee_hand_start_idx:ee_hand_start_idx+hand_dof]
                    # )
                    # squeeze_dir[4] = 0 # thumb does not move

                    squeeze_pose_qposs = []
                    for i in range(0, 20, 1):
                        squeeze_pose_qposs.append(np.concatenate(
                            [mogen_result_traj[-1][:ee_hand_start_idx],
                            mogen_result_traj[-1][ee_hand_start_idx:ee_hand_start_idx+hand_dof] + squeeze_dir * (i+1)/20 * 0.2,
                            mogen_result_traj[-1][ee_hand_start_idx+hand_dof:],
                            ], axis=-1
                        )[None, :])

                    squeeze_pose_qposs = np.concatenate(squeeze_pose_qposs, axis=0) 
                    trajs=[np.concatenate([mogen_result_traj, squeeze_pose_qposs,], axis=0)]
                
                return (trajs, jNames)
            else:
                raise ValueError(f"No valid trajectory found: {mogen_result.status}")


    def batch_plan_for_approach(self, goal_poses, **kwargs) -> tuple[list, list]:
        """ Batch plan the end-effector waypoints. 
        Args:
            eef_waypoints (list of Pose): list of end-effector waypoints (p,q) to plan to.
        """
        # num_waypoints = len(goal_poses[0])
        # assert num_waypoints <= 2, "not implemented"
        timer = Timer()

        world_layout = kwargs.pop("world_layout", None)
        if not self.easy_motion_gen and world_layout is not None:
            world_cfg = self.create_collision_world_cfg(world_layout)
            self.motion_gen.update_world(world_cfg)
            
        approach_axis = kwargs.pop("approach_axis", 'z')

        # for i in range(num_waypoints):
        with timer(f"approach"):
            # valid_mask = []
            # approach_trajectories = []
            # grasp_joint_states = []
            if isinstance(goal_poses[0], dict):
                grasp_poses = np.array([np.concatenate((p["position"], p["orientation"]), axis=0) for p in goal_poses], dtype=np.float32) # FIXME rename the variable (B,7)
            elif isinstance(goal_poses[0], (list, tuple)):
                grasp_poses = np.array([np.concatenate(p, dtype=np.float32) for p in goal_poses], dtype=np.float32) # FIXME rename the variable (B,7)
            else:
                grasp_poses = np.asarray(goal_poses, dtype=np.float32)

            batch_size = grasp_poses.shape[0]

            random_initial_states_for_batch_planning = self.robot.planning_init_joint_states(batch_size)
            random_initial_quats_for_batch_planning = self.robot.planning_init_quats(batch_size)
            random_idx = torch.randint(0, len(random_initial_states_for_batch_planning), (batch_size,))#shape[self.plan_per_traj]

            sampled_init_joint_states = random_initial_states_for_batch_planning[random_idx]
            sampled_init_quats = random_initial_quats_for_batch_planning[random_idx]
            if len(sampled_init_joint_states.shape) == 1:
                sampled_init_joint_states = sampled_init_joint_states[None,:].repeat(self.plan_per_traj, axis=0) #(B, joint_dim)
            if len(sampled_init_quats.shape) == 1:
                sampled_init_quats = sampled_init_quats[None,:] #(B, 4)

            tensor_args = TensorDeviceType()
            init_joint_states = JointState.from_position(tensor_args.to_device(sampled_init_joint_states))

            # _, grasp_pose = robot.get_grasp_pose_wrt_robot(grasp_info, T_world_robot=T_world_robot)  # self.__get_grasp_pose_wrt_robot(robot, grasp_info)
            # closer_grasp_quats = [self.__flip_by_z_if_closer(grasp_pose[1], init_quat) for init_quat in sampled_init_quats]
            closer_grasp_quats = []
            for (grasp_quat, init_quat) in zip(grasp_poses[:, 3:], sampled_init_quats):
                closer_grasp_quat = self.robot.rotate_round_approach_dir_if_needed(grasp_quat, init_quat, approach_axis=approach_axis)
                # closer_grasp_quat=grasp_quat
                closer_grasp_quats.append(closer_grasp_quat)

            # update the re-oriented grasp poses
            grasp_poses[:, 3:] = np.array(closer_grasp_quats, dtype=np.float32)

            curobo_goal_poses: Pose = self._stack_curobo_type(
                Pose(
                    position=torch.tensor(p[:3], device='cuda', dtype=torch.float32), 
                    quaternion=torch.tensor(p[3:], device='cuda', dtype=torch.float32)
                ) 
                for p in grasp_poses
            )
            if curobo_goal_poses.shape[0] != init_joint_states.shape[0]:
                # repeat the goal poses to match the initial joint states
                curobo_goal_poses = curobo_goal_poses.repeat(self.plan_per_traj) # self.plan_per_traj

            if "link_poses" in kwargs:
                for k, v in kwargs["link_poses"].items():
                    link_pose = Pose(
                        position=torch.tensor(v[:3], device='cuda', dtype=torch.float32), 
                        quaternion=torch.tensor(v[3:], device='cuda', dtype=torch.float32)
                    )
                    kwargs["link_poses"][k] = link_pose.repeat(self.plan_per_traj)

            mogen_result = self.motion_gen.plan_batch(
                init_joint_states, 
                curobo_goal_poses,
                plan_config=MotionGenPlanConfig(enable_finetune_trajopt=False, num_trajopt_seeds=12, max_attempts=10),
                **kwargs
            )
            if torch.sum(mogen_result.success) > 0:
                success_idx = torch.where(mogen_result.success)[0]
                traj_length = mogen_result.path_buffer_last_tstep
                # for each init joint state, find the shortest successful trajectory
                # temporarily, use for loop, debuggy!!!!!
                shorter_success_idx_list = []
                for init_js_idx in range(batch_size):
                    start_idx = init_js_idx * self.plan_per_traj
                    end_idx = (init_js_idx + 1) * self.plan_per_traj
                    js_traj_length = traj_length[start_idx:end_idx]
                    js_success_mask = mogen_result.success[start_idx:end_idx]
                    # find the shortest successful trajectory
                    success_idx = torch.where(js_success_mask)[0].cpu().numpy()
                    if len(success_idx) > 0:
                        js_traj_length = np.array([ length.item() for length in js_traj_length])
                        shortest_success_idx = success_idx[np.argmin(js_traj_length[success_idx])]
                        success_idx = start_idx + shortest_success_idx
                        shorter_success_idx_list.append(success_idx)
                
                jNames = mogen_result.interpolated_plan.joint_names
                # chosen_traj_idx = random.choice(shortest_success_idx_list)
                approach_trajectory = mogen_result.interpolated_plan[shorter_success_idx_list]
                trajs = []
                for idx in range(len(shorter_success_idx_list)):
                    trajs.append(
                        approach_trajectory[idx].trim_trajectory(
                            0, mogen_result.path_buffer_last_tstep[shorter_success_idx_list[idx]]
                        ).position.cpu().numpy()
                    )

                # TODO support pre-grasp
                return (trajs, jNames)
            
        #         print(f'[CUROBO] approach trajectory length: {apporach_trajectory.shape}')
        #         valid_mask.append(True)
        #         approach_trajectories.append(apporach_trajectory.position[:, :self.robot.dof].cpu().numpy())
        #         grasp_joint_states.append(mogen_result.optimized_plan.position[chosen_traj_idx, -1, :])
            else:
                raise ValueError(f"No valid trajectory found: {mogen_result.status}")
            
        #         valid_mask.append(False)
        #         approach_trajectories.append(None)
        #         grasp_joint_states.append(torch.zeros(self.robot.dof, device='cuda', dtype=torch.float32))

        #     grasp_joint_states = torch.stack(grasp_joint_states) # N, joint_dim
        #     approach_success_mask = torch.tensor(valid_mask, device='cuda', dtype=torch.bool)

        # if num_waypoints > 1:
        #     with timer("lift"):
        #         lift_trajectories, lift_success_mask = self.__batch_plan_for_lift(grasp_joint_states)
        #         print(f'[CUROBO] lift trajectories shape: {lift_trajectories.shape}, lift success mask shape: {lift_success_mask.shape}')

        # final_success_idx = torch.where(torch.logical_and(approach_success_mask, lift_success_mask))[0]
        # valid_ret = []
        # for idx in final_success_idx:
        #     # valid_ret.append((layouts[idx], selected_target_poses[idx], approach_trajectories[idx], lift_trajectories[:, idx]))
        #     valid_ret.append((approach_trajectories[idx], lift_trajectories[:, idx]))
        # return valid_ret

    def batch_plan_for_pregrasp(self, current_joint_states, grasp_distance, step_distance=0.01, **kwargs) -> list[dict[str, float]]:
        ...
        

    def batch_plan_for_lift(self, current_joint_states, lift_height=0.1, step_distance=0.01) -> list[dict[str, float]]:
        """  
        Batch plan the lift motion from current joint states.
        """
        # current_joint_states = torch.tensor(list(current_joint_states.values()), device='cuda', dtype=torch.float32)
        if isinstance(current_joint_states, dict):
            current_joint_states = torch.tensor(list(current_joint_states.values()), device='cuda', dtype=torch.float32)
        else:
            current_joint_states = torch.tensor(current_joint_states, device='cuda', dtype=torch.float32)

        if len(current_joint_states.shape) == 1:
            current_joint_states = current_joint_states.unsqueeze(0) # 1, dof
        batch_size = current_joint_states.shape[0]
        lift_steps = abs(int(lift_height / step_distance))
        current_hand_states = self.init_js_ik_solver.fk(current_joint_states)

        lift_trajectories = np.zeros((lift_steps, batch_size, self.robot.dof))#[steps, batch_size, dof]
        success_flag = torch.ones(batch_size, dtype=torch.bool, device='cuda')

        for step_idx in range(lift_steps):
            current_hand_states = self.init_js_ik_solver.fk(current_joint_states)
            target_hand_position = current_hand_states.ee_position
            target_hand_position[:, 2] += step_distance
            target_hand_orientation = current_hand_states.ee_quaternion
            target_poses = Pose(position=target_hand_position, quaternion=target_hand_orientation)

            retract_cfg = current_joint_states
            seed_cfg = current_joint_states.unsqueeze(1).repeat(1, 2, 1)
            
            ik_result = self.lift_ik_solver.solve_batch(target_poses, retract_cfg, seed_cfg)
            if torch.sum(ik_result.success) > 0:
                current_step_success = ik_result.success[:, 0] # N
                success_flag = torch.logical_and(success_flag, current_step_success)
                lift_trajectories[step_idx] = ik_result.solution[:, 0, :].cpu().numpy()
                current_joint_states = ik_result.solution[:, 0, :]
            else:
                raise ValueError(f"No valid lift trajectory found: {ik_result.status}")
        traj = lift_trajectories[:, success_flag.cpu().numpy(), :]
        jNames = self.lift_ik_solver.joint_names

        lift_trajs = []
        for b in range(traj.shape[1]):
            traj_b = []
            for t in range(traj.shape[0]):
                traj_b.append(dict(zip(jNames, traj[t, b, :])))
            lift_trajs.append(traj_b)
        if len(lift_trajs) == 1:
            return lift_trajs[0]
        elif len(lift_trajs) == 0:
            raise ValueError(f"No valid lift trajectory found!")
        else:
            return lift_trajs
    
    def batch_plan_for_lift_bodex(self, current_joint_states, lift_height=0.1, step_distance=0.01) -> list[dict[str, float]]:
        """  
        Batch plan the lift motion from current joint states.
        """
        # only need plan_dof not all dof
        
        joint_names = self.init_js_ik_solver.joint_names
        js_list = [current_joint_states[name] for name in joint_names]  
        plan_dof=len(self.init_js_ik_solver.joint_names)
        # current_joint_states = torch.tensor(list(current_joint_states.values()), device='cuda', dtype=torch.float32)[:plan_dof]
        current_joint_states = torch.tensor(js_list, device='cuda', dtype=torch.float32)[:plan_dof]

        if len(current_joint_states.shape) == 1:
            current_joint_states = current_joint_states.unsqueeze(0) # 1, dof
        batch_size = current_joint_states.shape[0]
        lift_steps = abs(int(lift_height / step_distance))
        current_hand_states = self.init_js_ik_solver.fk(current_joint_states)

        lift_trajectories = np.zeros((lift_steps, batch_size, plan_dof))#[steps, batch_size, dof]
        success_flag = torch.ones(batch_size, dtype=torch.bool, device='cuda')

        for step_idx in range(lift_steps):
            current_hand_states = self.init_js_ik_solver.fk(current_joint_states)
            target_hand_position = current_hand_states.ee_position
            target_hand_position[:, 2] += step_distance
            target_hand_orientation = current_hand_states.ee_quaternion
            target_poses = Pose(position=target_hand_position, quaternion=target_hand_orientation)

            retract_cfg = current_joint_states
            seed_cfg = current_joint_states.unsqueeze(1).repeat(1, 2, 1)

            ik_result = self.lift_ik_solver.solve_batch(target_poses, retract_cfg, seed_cfg)
            if torch.sum(ik_result.success) > 0:
                current_step_success = ik_result.success[:, 0] # N
                success_flag = torch.logical_and(success_flag, current_step_success)
                lift_trajectories[step_idx] = ik_result.solution[:, 0, :].cpu().numpy()
                current_joint_states = ik_result.solution[:, 0, :]
            else:
                raise ValueError(f"No valid lift trajectory found: {ik_result.status}")
        traj = lift_trajectories[:, success_flag.cpu().numpy(), :]
        jNames = self.lift_ik_solver.joint_names

        lift_trajs = []
        for b in range(traj.shape[1]):
            traj_b = []
            for t in range(traj.shape[0]):
                traj_b.append(dict(zip(jNames, traj[t, b, :])))
            lift_trajs.append(traj_b)
        if len(lift_trajs) == 1:
            return lift_trajs[0]
        return lift_trajs

    def batch_plan_for_move(
        self, 
        goal_poses,
        current_joint_states,
        **kwargs
    ) -> tuple[list, list]:
        """ 
        Batch plan to specified end-effector poses directly.
        """
        timer = Timer()
        
        # Update world collision model if needed
        world_layout = kwargs.pop("world_layout", None)
        if not self.easy_motion_gen and world_layout is not None:
            world_cfg = self.create_collision_world_cfg(world_layout)
            self.motion_gen.update_world(world_cfg)
        
        with timer("plan_for_move"):
            # ========== 1. Convert goal poses to numpy array ==========
            if isinstance(goal_poses[0], dict):
                goal_pose_array = np.array([
                    np.concatenate((p["position"], p["orientation"]), axis=0) 
                    for p in goal_poses
                ], dtype=np.float32)
            elif isinstance(goal_poses[0], (list, tuple)):
                goal_pose_array = np.array([
                    np.concatenate(p, dtype=np.float32) 
                    for p in goal_poses
                ], dtype=np.float32)
            else:
                goal_pose_array = np.asarray(goal_poses, dtype=np.float32)
            
            batch_size = goal_pose_array.shape[0]
            
            # ========== 2. Prepare initial joint states ==========
            if isinstance(current_joint_states, dict):
                joint_names = self.init_js_ik_solver.joint_names
                js_list = [current_joint_states[name] for name in joint_names]
                current_joint_states = np.array(js_list, dtype=np.float32)
            elif not isinstance(current_joint_states, np.ndarray):
                current_joint_states = np.array(current_joint_states, dtype=np.float32)
            
            if len(current_joint_states.shape) == 2:
                current_joint_states = current_joint_states.squeeze(0)
            
            # Expand to batch with plan_per_traj copies for each goal
            # This allows CuRobo to try multiple trajectory optimizations per goal
            sampled_init_joint_states = current_joint_states[None, :].repeat(
                batch_size * self.plan_per_traj, axis=0
            )  # Shape: (batch_size * plan_per_traj, joint_dim)
            
            # ========== 3. Get current end-effector orientation via FK ==========
            tensor_args = TensorDeviceType()
            current_joint_states_cuda = tensor_args.to_device(
                current_joint_states[None, :]
            )
            current_hand_state = self.init_js_ik_solver.fk(current_joint_states_cuda)
            current_quat = current_hand_state.ee_quaternion.cpu().numpy()[0]  # Shape: (4,)
            
            sampled_init_quats = np.tile(current_quat, (batch_size, 1))  # Shape: (batch_size, 4)
            
            # ========== 4. Adjust goal orientations to be closer to current orientation ==========
            closer_goal_quats = []
            for (goal_quat, init_quat) in zip(goal_pose_array[:, 3:], sampled_init_quats):
                closer_goal_quat = self.robot.rotate_round_approach_dir_if_needed(
                    goal_quat, init_quat
                )
                closer_goal_quats.append(closer_goal_quat)
            
            # Update goal poses with adjusted orientations
            goal_pose_array[:, 3:] = np.array(closer_goal_quats, dtype=np.float32)
            
            # ========== 5. Create CuRobo data structures ==========
            init_joint_states = JointState.from_position(
                tensor_args.to_device(sampled_init_joint_states)
            )
            
            curobo_goal_poses: Pose = self._stack_curobo_type(
                Pose(
                    position=torch.tensor(p[:3], device='cuda', dtype=torch.float32), 
                    quaternion=torch.tensor(p[3:], device='cuda', dtype=torch.float32)
                ) 
                for p in goal_pose_array
            )
            
            # Repeat goal poses to match init_joint_states batch size
            if curobo_goal_poses.shape[0] != init_joint_states.shape[0]:
                curobo_goal_poses = curobo_goal_poses.repeat(self.plan_per_traj)
            
            # ========== 6. Plan with motion generator ==========
            mogen_result = self.motion_gen.plan_batch(
                init_joint_states, 
                curobo_goal_poses,
                plan_config=MotionGenPlanConfig(
                    enable_finetune_trajopt=False,
                    num_trajopt_seeds=12,
                    max_attempts=1,
                ),
                **kwargs
            )
            
            # ========== 7. Extract successful trajectories ==========
            if torch.sum(mogen_result.success) > 0:
                # Find shortest successful trajectory for each goal
                shorter_success_idx_list = []
                for goal_idx in range(batch_size):
                    start_idx = goal_idx * self.plan_per_traj
                    end_idx = (goal_idx + 1) * self.plan_per_traj
                    
                    # Get trajectory lengths and success mask for this goal
                    traj_lengths = mogen_result.path_buffer_last_tstep[start_idx:end_idx]
                    success_mask = mogen_result.success[start_idx:end_idx]
                    
                    # Find successful trajectories
                    success_idx_local = torch.where(success_mask)[0].cpu().numpy()
                    
                    if len(success_idx_local) > 0:
                        # Choose shortest successful trajectory
                        traj_lengths_np = np.array([l.item() for l in traj_lengths])
                        shortest_idx = success_idx_local[
                            np.argmin(traj_lengths_np[success_idx_local])
                        ]
                        global_idx = start_idx + shortest_idx
                        shorter_success_idx_list.append(global_idx)
                
                if len(shorter_success_idx_list) == 0:
                    raise ValueError(
                        f"No valid move trajectory found after filtering: {mogen_result.status}"
                    )
                
                jNames = mogen_result.interpolated_plan.joint_names
                trajs = []
                for idx in shorter_success_idx_list:
                    traj = mogen_result.interpolated_plan[idx].trim_trajectory(
                        0, mogen_result.path_buffer_last_tstep[idx]
                    ).position.cpu().numpy()
                    trajs.append(traj)
                
                return (trajs, jNames)
            else:
                raise ValueError(f"No valid move trajectory found: {mogen_result.status}")
        
    def batch_plan_for_move_bodex(
        self, 
        position,
        orientation,
        current_joint_states,
        **kwargs
    ) -> tuple[list, list]:
        """ 
        Batch plan to specified end-effector poses directly.
        """
        timer = Timer()
        empty_world = WorldConfig()

        # create new motion gen, I don't know why self.motion_gen is always plan False 
        motion_gen = MotionGen(MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            world_model=empty_world, #world_cfg_table,

            interpolation_dt=self.plan_dt,
            interpolation_steps=1000,
            collision_cache={"obb": 10, "mesh": 10},

            collision_activation_distance=torch.tensor([0.01], device='cuda', dtype=torch.float32),
            ik_opt_iters=200,
            grad_trajopt_iters=200,
        ))
        
        #Update world collision model if needed
        # world_layout = kwargs.pop("world_layout", None)
        # if not self.easy_motion_gen and world_layout is not None:
        #     world_cfg = self.create_collision_world_cfg(world_layout)
        #     motion_gen.update_world(world_cfg)
        
        with timer("plan_for_move"):
            # ========== 2. Prepare initial joint states ==========
            if not isinstance(current_joint_states, np.ndarray):
                current_joint_states = np.array(current_joint_states, dtype=np.float32)
            
            # if len(current_joint_states.shape) == 2:
            #     current_joint_states = current_joint_states.squeeze(0)
            
            # Expand to batch with plan_per_traj copies for each goal
            # This allows CuRobo to try multiple trajectory optimizations per goal
            sampled_init_joint_states = current_joint_states[None, :].repeat(
                1 , axis=0
            )  # Shape: (batch_size , joint_dim)
            
            # ========== 3. Get current end-effector orientation via FK ==========
            tensor_args = TensorDeviceType()
            current_joint_states_cuda = tensor_args.to_device(
                current_joint_states[None, :]
            )
            current_hand_state = self.init_js_ik_solver.fk(current_joint_states_cuda).ee_pose
            current_hand_position = current_hand_state.position.cpu().numpy()[0]  # Shape: (3,)
            current_hand_quat = current_hand_state.quaternion.cpu().numpy()[0]  # Shape: (4,)
            
            #TODO batch size now does not support
            goal_pose_array = []

            if orientation is not None:
                goal_pose_array.append(np.concatenate((position, orientation), axis=0))
                # goal_pose_array[0][0] -= 0.1
                # goal_pose_array[0][1] -= 0.1#HACK
            
            else:
                goal_pose_array.append(np.concatenate((current_hand_position, current_hand_quat), axis=0))
                assert position.shape == (2,)
                goal_pose_array[0][0] = position[0]
                goal_pose_array[0][1] = position[1]
                
            # goal_pose_array[:, 3:] = np.array(current_quat, dtype=np.float32)
            
            # ========== 5. Create CuRobo data structures ==========
            init_joint_states = JointState.from_position(
                tensor_args.to_device(sampled_init_joint_states)
            )
            
            curobo_goal_poses: Pose = self._stack_curobo_type(
                Pose(
                    position=torch.tensor(p[:3], device='cuda', dtype=torch.float32), 
                    quaternion=torch.tensor(p[3:], device='cuda', dtype=torch.float32)
                ) 
                for p in goal_pose_array
            )
            
            # Repeat goal poses to match init_joint_states batch size
            if curobo_goal_poses.shape[0] != init_joint_states.shape[0]:
                curobo_goal_poses = curobo_goal_poses.repeat(init_joint_states.shape[0])

            link_poses = {}
            init_qpos = torch.tensor(sampled_init_joint_states[0])
            lock_links_names = kwargs.pop("lock_links_names", None)
            if lock_links_names is not None:
                # FIXME
                init_qpos = torch.zeros(len(self.motion_gen.joint_names))
                all_origin_link_poses = self.init_js_ik_solver.fk(init_qpos).link_poses
                lock_links_poses = {name: all_origin_link_poses[name] for name in lock_links_names}
                link_poses.update(lock_links_poses)
            
            # ========== 6. Plan with motion generator ==========
            mogen_result = motion_gen.plan_batch(
                init_joint_states, 
                curobo_goal_poses,
                link_poses=link_poses,
                plan_config=MotionGenPlanConfig(
                    enable_finetune_trajopt=False,
                    num_trajopt_seeds=12,
                    max_attempts=10,
                ),
            )

            ee_wrist_yaw_idx = next(
                (i for i, n in enumerate(self.motion_gen.joint_names) if n.endswith("wrist_yaw_joint")),
                None
            )

            if ee_wrist_yaw_idx is None:
                raise ValueError("Cannot find ee_wrist_yaw_joint in joint_names!")

            ee_hand_start_idx = ee_wrist_yaw_idx + 1
            hand_dof=self.robot.hand_dof
            
            # ========== 7. Extract successful trajectories ==========
            if torch.sum(mogen_result.success) > 0:
                jNames = mogen_result.interpolated_plan.joint_names
                trajs = []

                traj_result = mogen_result.interpolated_plan.trim_trajectory(
                    0, mogen_result.path_buffer_last_tstep[0]
                ).position.cpu().numpy()
                traj_result[:,ee_hand_start_idx:ee_hand_start_idx+hand_dof] = current_joint_states[ee_hand_start_idx:ee_hand_start_idx+hand_dof]
                traj_result[:,-hand_dof:] = 0
                trajs.append(traj_result)
                
                return (trajs, jNames)
            else:
                raise ValueError(f"No valid move trajectory found: {mogen_result.status}")

    def batch_plan_for_retreat_bodex(
            self, 
            current_joint_states,
            **kwargs
        ) -> tuple[list, list]:
            """ 
            Batch plan to specified end-effector poses directly.
            """
            timer = Timer()
            empty_world = WorldConfig()
            # create new motion gen, I don't know self.motion_gen is always plan False
            motion_gen = MotionGen(MotionGenConfig.load_from_robot_config(
                self.robot_cfg,
                world_model=empty_world, #world_cfg_table,
            
                interpolation_dt=self.plan_dt,
                interpolation_steps=1000,
                collision_cache={"obb": 10, "mesh": 10},
            
                collision_activation_distance=torch.tensor([0.01], device='cuda', dtype=torch.float32),
                ik_opt_iters=200,
                grad_trajopt_iters=200,
            ))
            
            # Update world collision model if needed
            world_layout = kwargs.pop("world_layout", None)
            if not self.easy_motion_gen and world_layout is not None:
                world_cfg = self.create_collision_world_cfg(world_layout)
                motion_gen.update_world(world_cfg)
            
            # FIXME batch_size now is fixed
            batch_size = 1
            tensor_args = TensorDeviceType()
            init_state=torch.zeros(len(self.motion_gen.joint_names))

            
            with timer("plan_for_move"):
                goal_pose = self.init_js_ik_solver.fk(init_state).ee_pose
                
                link_poses = {}
     
                lock_links_names = kwargs.pop("lock_links_names", None)
                if lock_links_names is not None:
                    if not isinstance(current_joint_states, torch.Tensor):
                        current_joint_states = torch.tensor(current_joint_states)
                    all_origin_link_poses = self.init_js_ik_solver.fk(current_joint_states).link_poses
                    lock_links_poses = {name: all_origin_link_poses[name] for name in lock_links_names}
                    link_poses.update(lock_links_poses)

                
                if not isinstance(current_joint_states, np.ndarray):
                    current_joint_states = np.array(current_joint_states, dtype=np.float32)
                    current_joint_states = np.round(current_joint_states,2)
                
                if len(current_joint_states.shape) == 2:
                    current_joint_states = current_joint_states.squeeze(0)
                
                # Expand to batch with plan_per_traj copies for each goal
                # This allows CuRobo to try multiple trajectory optimizations per goal
                sampled_init_joint_states = current_joint_states[None, :].repeat(
                    batch_size , axis=0
                )  # Shape: (batch_size * plan_per_traj, joint_dim)
                
                # ========== 3hand_uid. Get current end-effector orientation via FK ==========
                tensor_args = TensorDeviceType()

                # ========== 5. Create CuRobo data structures ==========
                init_joint_states = JointState.from_position(
                    tensor_args.to_device(sampled_init_joint_states)
                )

                # ========== 6. Plan with motion generator ==========
                mogen_result = motion_gen.plan_batch(
                    init_joint_states, 
                    goal_pose,
                    link_poses =link_poses,
                    plan_config=MotionGenPlanConfig(
                        enable_finetune_trajopt=False,
                        num_trajopt_seeds=12,
                        max_attempts=10,
                    ),
                )

                ee_wrist_yaw_idx = next(
                    (i for i, n in enumerate(self.motion_gen.joint_names) if n.endswith("wrist_yaw_joint")),
                    None
                )

                if ee_wrist_yaw_idx is None:
                    raise ValueError("Cannot find ee_wrist_yaw_joint in joint_names!")

                ee_hand_start_idx = ee_wrist_yaw_idx + 1
                hand_dof=self.robot.hand_dof
                # ========== 7. Extract successful trajectories ==========
                if torch.sum(mogen_result.success) > 0:
                    jNames = mogen_result.interpolated_plan.joint_names
                    trajs = []
               
                    tarj_result = mogen_result.interpolated_plan.trim_trajectory(
                        0, mogen_result.path_buffer_last_tstep[0]
                    ).position.cpu().numpy()
                    
                    tarj_result[:,ee_hand_start_idx:ee_hand_start_idx+hand_dof] = np.zeros(hand_dof)
                    trajs.append(tarj_result)
                    
                    return (trajs, jNames)
                else:
                    raise ValueError(f"No valid trajectory found: {mogen_result.status}")

    def create_collision_world_cfg(self, layout: Layout):
        """  create collision world (in robot frame) config from layout
        Args:
            layout (Layout): layout containing actors
        Returns:
            WorldConfig: world config for collision avoidance
        """

        robot_pose = layout.actors["robot"].pose.as_vec()
        def transform_to_robot_frame(object_pose: np.ndarray) -> list[float]:
            """ Transform world coordinates to robot frame """
            T_wr = np.eye(4)
            T_wr[:3, :3] = t3d.quaternions.quat2mat(robot_pose[3:])
            T_wr[:3, 3] = robot_pose[:3]
            
            T_wo = np.eye(4)
            T_wo[:3, :3] = t3d.quaternions.quat2mat(object_pose[3:])
            T_wo[:3, 3] = object_pose[:3]

            T_ro = np.linalg.inv(T_wr) @ T_wo
            return np.concatenate([T_ro[:3, 3], t3d.quaternions.mat2quat(T_ro[:3, :3])], axis=0).tolist()
        
        world_cfg = WorldConfig()
        for key, obj in layout.actors.items():
            if key == "robot": 
                continue

            if self.ignore_target_collisions and key == "target":
                continue

            if isinstance(obj, BoxActor):
                if layout.actors["robot"].robot.uid.startswith("franka") and key == "table":
                    # HACK franka can not has table
                    continue
                mesh_cfg = Cuboid(
                    name=key,
                    pose=transform_to_robot_frame(obj.pose.as_vec()),
                    dims=obj.size
                )
            else:
                assert isinstance(obj, ObjectActor)
                mesh_cfg = Mesh(
                    name=f"{key}_{obj.uid}",
                    pose=transform_to_robot_frame(obj.pose.as_vec()),
                    file_path=os.path.abspath(obj.asset.collision_mesh_curobo),
                    scale=[1.0, 1.0, 1.0],
                )
            world_cfg.add_obstacle(mesh_cfg)
        # world_cfg.save_world_as_mesh("test.obj")
        return world_cfg

    # def __plan_batch(self, robot, layouts, assets_info, target_poses, T_world_robot = None):
    #     timer = Timer()

    #     with timer("approach"):
    #         valid_mask = []
    #         approach_trajectories = []
    #         grasp_joint_states = []
    #         selected_target_poses = [] # SONGLIN: save selected grasping pose only
    #         for target_pose, layout in zip(target_poses, layouts):
    #             grasp_info = random.choice(target_pose["grasp_poses"])
    #             selected_target_poses.append({
    #                 "target_info": copy.deepcopy(target_pose['target_info']),
    #                 "grasp_poses": [grasp_info]
    #             })
    #             if not self.easy_motion_gen:
    #                 world_cfg = self.__update_world_config(layout, assets_info)
    #                 self.motion_gen.update_world(world_cfg)

    #             # randomly choose init joint states
    #             random_initial_states_for_batch_planning = robot.planning_init_joint_states(self.plan_batch_size)
    #             random_initial_quats_for_batch_planning = robot.planning_init_quats(self.plan_batch_size)
    #             random_idx = torch.randint(0, len(random_initial_states_for_batch_planning), (self.plan_per_traj,))#shape[self.plan_per_traj]
    #             sampled_init_joint_states = random_initial_states_for_batch_planning[random_idx]
    #             sampled_init_quats = random_initial_quats_for_batch_planning[random_idx]
    #             tensor_args = TensorDeviceType()
    #             init_joint_states = JointState.from_position(tensor_args.to_device(sampled_init_joint_states))


    #             # calibrate orientation to be nearest to current_hand_poses orientation, to avoid unnecessary flipping
    #             _, grasp_pose = robot.get_grasp_pose_wrt_robot(grasp_info, T_world_robot=T_world_robot)  # self.__get_grasp_pose_wrt_robot(robot, grasp_info)
    #             closer_grasp_quats = [self._flip_by_z_if_closer(grasp_pose[1], init_quat) for init_quat in sampled_init_quats]
                
    #             new_pregrasp_poses: Pose = self._stack_curobo_type(
    #                 Pose(
    #                     position=torch.tensor(grasp_pose[0], device='cuda', dtype=torch.float32), 
    #                     quaternion=torch.tensor(grasp_quat, device='cuda', dtype=torch.float32)
    #                 ) 
    #                 for grasp_quat in closer_grasp_quats
    #             )
    #             mogen_result = self.motion_gen.plan_batch(
    #                 init_joint_states, 
    #                 new_pregrasp_poses,
    #                 plan_config=MotionGenPlanConfig(enable_finetune_trajopt=False, num_trajopt_seeds=4, max_attempts=1)
    #             )
    #             if torch.sum(mogen_result.success) > 0:
    #                 success_idx = torch.where(mogen_result.success)[0]
    #                 traj_length = mogen_result.path_buffer_last_tstep
    #                 # for each init joint state, find the shortest successful trajectory
    #                 # temporarily, use for loop, debuggy!!!!!
    #                 shortest_success_idx_list = []
    #                 for init_js_idx in range(self.plan_per_traj):
    #                     start_idx = init_js_idx * 4
    #                     end_idx = (init_js_idx + 1) * 4
    #                     js_traj_length = traj_length[start_idx:end_idx]
    #                     js_success_mask = mogen_result.success[start_idx:end_idx]
    #                     # find the shortest successful trajectory
    #                     success_idx = torch.where(js_success_mask)[0].cpu().numpy()
    #                     if len(success_idx) > 0:
    #                         js_traj_length = np.array([ length.item() for length in js_traj_length])
    #                         shortest_success_idx = success_idx[np.argmin(js_traj_length[success_idx])]
    #                         success_idx = start_idx + shortest_success_idx
    #                         shortest_success_idx_list.append(success_idx)
                        
    #                 chosen_traj_idx = random.choice(shortest_success_idx_list)
    #                 apporach_trajectory = mogen_result.interpolated_plan[chosen_traj_idx].trim_trajectory(0, mogen_result.path_buffer_last_tstep[chosen_traj_idx])
    #                 print(f'[CUROBO] approach trajectory length: {apporach_trajectory.shape}')
    #                 valid_mask.append(True)
    #                 approach_trajectories.append(apporach_trajectory.position[:, :self.robot.dof].cpu().numpy())
    #                 grasp_joint_states.append(mogen_result.optimized_plan.position[chosen_traj_idx, -1, :])
    #             else: 
    #                 valid_mask.append(False)
    #                 approach_trajectories.append(None)
    #                 grasp_joint_states.append(torch.zeros(self.robot.dof, device='cuda', dtype=torch.float32))
    #         grasp_joint_states = torch.stack(grasp_joint_states) # N, joint_dim
    #         approach_success_mask = torch.tensor(valid_mask, device='cuda', dtype=torch.bool)
    #     # print(f'plan for {approach_success_mask.sum()} valid approach took {timer.get_time("approach")}s')
        
    #     with timer("lift"):
    #         lift_trajectories, lift_success_mask = self._batch_plan_for_lift(grasp_joint_states)
    #         print(f'[CUROBO] lift trajectories shape: {lift_trajectories.shape}, lift success mask shape: {lift_success_mask.shape}')
    #     # print(f'plan for {lift_success_mask.sum()} valid lift took {timer.get_time("lift")}s')

    #     final_success_idx = torch.where(torch.logical_and(approach_success_mask, lift_success_mask))[0]
    #     valid_ret = []
    #     for idx in final_success_idx:
    #         valid_ret.append((layouts[idx], selected_target_poses[idx], approach_trajectories[idx], lift_trajectories[:, idx]))
    #     return valid_ret

    # def __update_world_config(self, layout, assets_info):
    #     world_cfg = WorldConfig(cuboid=[Cuboid(
    #         'table', pose=[1., 0., 0.45912908 - 0.1/2, 1., 0., 0., 0.], scale=[1., 1., 1.], dims=[1.27, 0.7112, 0.1]
    #         )])
    #     for obj in layout:
    #         # change to make CubeLayout compatible
    #         if obj["name"].split("_")[0] in ["cube"]:
    #             mesh_cfg = Cuboid(
    #                 name=obj["name"],
    #                 pose=obj["position"].tolist() + obj["orientation"].tolist(),
    #                 dims=obj["dims"],
    #                 color=[1.0, 0.0, 0.0],
    #             )
    #         else:
    #             p = np.array(obj["position"], dtype=np.float32) + np.array([1.0, 0.0, 0.45912908])
    #             pose7 = (p.tolist() + obj["orientation"]) if isinstance(obj["position"], list) else obj["position"].tolist() + obj["orientation"].tolist()
    #             mesh_cfg = Mesh(
    #                 name=f'{obj["name"]}',
    #                 pose=pose7,
    #                 file_path=os.path.abspath(assets_info[obj['id']]["collision_mesh_curobo"]),
    #                 scale=[1.0, 1.0, 1.0],
    #             )
    #         world_cfg.add_obstacle(mesh_cfg)
        
    #     # # debug
    #     # world_cfg.add_obstacle(Cuboid(
    #     #     'debug_cuboid', pose=[0., 0., 0., 1, 0, 0, 0], scale=[1, 1,  1], dims=[0.05, 0.05, 0.05]
    #     # ))
    #     # world_cfg.add_obstacle(Cuboid(
    #     #     'debug_cuboid', pose=[0.1, 0., 0., 1, 0, 0, 0], scale=[1, 1,  1], dims=[0.05, 0.05, 0.05], color=[1,0,0]
    #     # ))
    #     # world_cfg.add_obstacle(Cuboid(
    #     #     'debug_cuboid', pose=[0., 0.1, 0., 1, 0, 0, 0], scale=[1, 1,  1], dims=[0.05, 0.05, 0.05], color=[0,1,0]
    #     # ))
    #     # world_cfg.save_world_as_mesh("test.obj")

    #     return world_cfg
    
    # def _flip_by_z_if_closer(self, target_quat: np.ndarray, ref_quat: np.ndarray):
    #     """
    #     Returns target_quat flipped by z axis if it is closer to ref_quat after flipping (180 degrees rotation around z axis).
    #     """
    #     ref_to_target_quat = t3d.quaternions.qmult(target_quat, t3d.quaternions.qinverse(ref_quat))
    #     _, ref_to_target_angle = t3d.quaternions.quat2axangle(ref_to_target_quat)
    #     # you can rotate inversely
    #     ref_to_target_angle = min(ref_to_target_angle, 2*np.pi - ref_to_target_angle)
    #     flipped_target_quat = t3d.quaternions.qmult(target_quat, t3d.quaternions.axangle2quat([0, 0, 1], np.pi))
    #     ref_to_flipped_target_quat = t3d.quaternions.qmult(flipped_target_quat, t3d.quaternions.qinverse(ref_quat))
    #     _, ref_to_flipped_target_angle = t3d.quaternions.quat2axangle(ref_to_flipped_target_quat)
    #     ref_to_flipped_target_angle = min(ref_to_flipped_target_angle, 2*np.pi - ref_to_flipped_target_angle)
    #     return target_quat if ref_to_target_angle < ref_to_flipped_target_angle else flipped_target_quat
    
    # def _batch_plan_for_lift(self, current_joint_states, step_distance=0.01):
    #     '''
    #         current_joint_states: CUDA tensor of N * 7
    #     '''
    #     lift_steps = int(self.lift_height / step_distance)

    #     current_hand_states = self.init_js_ik_solver.fk(current_joint_states)

    #     lift_trajectories = np.zeros((lift_steps, len(current_joint_states), self.robot.dof))#[steps, batch_size, dof]
    #     success_flag = torch.ones(len(current_joint_states), dtype=torch.bool, device='cuda')
    #     for step_idx in range(lift_steps):
    #         current_hand_states = self.init_js_ik_solver.fk(current_joint_states)
    #         target_hand_position = current_hand_states.ee_position
    #         target_hand_position[:, 2] += step_distance
    #         target_hand_orientation = current_hand_states.ee_quaternion
    #         target_poses = Pose(position=target_hand_position, quaternion=target_hand_orientation)

    #         retract_cfg = current_joint_states
    #         seed_cfg = current_joint_states.unsqueeze(1).repeat(1, 2, 1)

    #         ik_result = self.lift_ik_solver.solve_batch(target_poses, retract_cfg, seed_cfg)
    #         current_step_success = ik_result.success[:, 0] # N
    #         success_flag = torch.logical_and(success_flag, current_step_success)
    #         lift_trajectories[step_idx] = ik_result.solution[:, 0, :].cpu().numpy()
    #         current_joint_states = ik_result.solution[:, 0, :]
    #     return lift_trajectories, success_flag
    
    def _stack_curobo_type(self, args):
        iterator = iter(args)
        ret = next(iterator).clone()
        for a in iterator:
            ret = ret.stack(a)
        return ret