from queue import Queue
import torch
import numpy as np
from copy import deepcopy
from collections import deque
from simple.core.actor import ObjectActor
from simple.core.object import SpatialAnnotated
from simple.core.task import Task
from simple.mp.planner import MotionPlanner
from simple.robots.protocols import Graspable
from simple.core.asset import Asset
from simple.core.layout import Layout
from simple.robots.protocols import DualArm
from .base_agent import BaseAgent
from .primitive_agent import PrimitiveAgent
from queue import Queue
from simple.grasps.gsnet import GSNet
from simple.grasps.bodex import Bodex
from simple.datagen.subtask_spec import (
    OpenGripperSpec,
    MoveEEFToPoseSpec,
    CloseGripperSpec,
    GraspObjectSpec,
    LiftSpec,
    LowerSpec,
    RetreatSpec,
    PhaseBreakSpec,
    RetreatSpec,
    PhaseBreakSpec,
    StandSpec,
    WalkSpec,
    TurnSpec,
    HeightAdjustSpec,
)
import transforms3d as t3d
import math

def pad_to_full_qpos(qpos:dict, init_qpos:dict):
    qpos_full = deepcopy(qpos)
    for jname, val in init_qpos.items():
        if jname not in qpos:
            qpos_full[jname] = val
    return qpos_full

class MotionPlannerAgent(PrimitiveAgent):

    plan_batch_size = 1
    
    def __init__(self, task:Task, planner: MotionPlanner, debug: bool = False, plan_batch_size: int | None = None):
        super().__init__(task.robot)

        self.task = task
        self.planner = planner
        self.plan_batch_size = plan_batch_size if plan_batch_size is not None else self.plan_batch_size

        self.debug = debug
        self._subtask_index = 0
        self._last_traj_qpos = None

        self._stored_grasps = {}  # Store grasps by key for reference,only for aloha
        self.target_stable_index = 0
        self.last_target_yaw = 0

    def synthesize(self):
        """  Pre-generate the solution (trajectory) for the episode
        Returns:
            True if synthesis completed successfully
            False if planning failed
            "phase_break" if a phase break was encountered (multi-phase planning)
        """
        all_subtasks = self.task.decompose()

        # Skip already processed subtasks in multi-phase planning
        subtasks = all_subtasks[self._subtask_index:] if self._subtask_index > 0 else all_subtasks

        #for wholebody locomotion
        self.original_robot_pose = list(self.task.layout.actors["robot"].pose.position) + list(self.task.layout.actors["robot"].pose.quaternion) 
        self.desired_robot_pose = self.original_robot_pose.copy()

        for i, spec in enumerate(subtasks):
            self._subtask_index += 1
            # Phase break
            if isinstance(spec, PhaseBreakSpec):
                return "phase_break"
            
            # get hand uid and update ee_link for plan
            hand_uid = spec.meta.get("hand_uid",None)
            from simple.robots.protocols import Humanoid
            if isinstance(self.robot, Humanoid):
                if hand_uid is not None:
                    self.robot.update_ee_link(hand_uid) # type: ignore
                    self.planner.reinit(self.robot)# type: ignore
            else:
                if hasattr(self.robot, 'switch_arm'):
                    if self.robot.switch_arm(hand_uid): # type: ignore
                        self.planner.reinit(self.robot)# type: ignore
                print(self.robot.robot_cfg["kinematics"]["ee_link"])# type: ignore
                print(f"[MP Agent] Switched to {hand_uid} arm")

            if isinstance(spec, OpenGripperSpec):
                for _ in range(10):
                    self.queue_open_gripper(hand_uid=hand_uid)
            elif isinstance(spec, CloseGripperSpec):
                for _ in range(10):
                    self.queue_close_gripper(hand_uid=hand_uid)
            elif isinstance(spec, MoveEEFToPoseSpec):
                target_position = spec.meta.get("position", None)
                target_orientation = spec.meta.get("orientation", None)
                grasp_type = spec.meta.get("grasp_type", "parallel_gripper")
                eef_state = spec.meta.get("eef_state", "close_eef")

                if target_position is None:
                    raise ValueError("MoveEEFToPoseSpec requires 'position' in meta")
                
                if target_orientation is None:
                    try:
                        current_js = getattr(self, "_last_traj_qpos", None)
                        if current_js is not None:
                            # Convert dict to array if needed for FK
                            if isinstance(current_js, dict):
                                # Use motion_gen.joint_names to get correct order
                                joint_names = self.planner.motion_gen.joint_names
                                current_js_array = np.array(
                                    [current_js[name] for name in joint_names],
                                    dtype=np.float32
                                )
                                current_js_tensor = torch.tensor(
                                    current_js_array, device="cuda", dtype=torch.float32
                                ).unsqueeze(0)
                            else:
                                current_js_tensor = torch.tensor(
                                    current_js, device="cuda", dtype=torch.float32
                                ).unsqueeze(0)
                            hand_state = self.planner.init_js_ik_solver.fk(current_js_tensor)
                            target_orientation = (
                                hand_state.ee_quaternion.detach().cpu().numpy()[0]
                            )
                        else:
                            target_orientation = t3d.euler.euler2quat(np.pi, 0, 0)
                    except Exception as e:
                        print(
                            f"[MP Agent] Failed to infer current EE orientation, "
                            f"falling back to default. Error: {e}"
                        )
                        target_orientation = t3d.euler.euler2quat(np.pi, 0, 0)
                
                robot_actor = self.task.layout.actors["robot"]
                robot_pose = robot_actor.pose.as_matrix()
                
                T_world_goal = np.eye(4)
                if target_position.shape[0] == 2:
                    T_world_goal[:2, 3] = target_position
                else:
                    T_world_goal[:3, 3] = target_position
                if len(target_orientation) == 4:  # quaternion
                    T_world_goal[:3, :3] = t3d.quaternions.quat2mat(target_orientation)
                else:  # rotation matrix
                    T_world_goal[:3, :3] = target_orientation
                
                T_robot_goal = np.linalg.inv(robot_pose) @ T_world_goal
                
                goal_pose_robot_frame = {
                    "position": T_robot_goal[:3, 3],
                    "orientation": t3d.quaternions.mat2quat(T_robot_goal[:3, :3])
                }
                
                try:
                    # self._last_traj_qpos = np.array(list(self._last_traj_qpos.values()), dtype=np.float32)[:self.planner.init_js_ik_solver.dof]
                    if grasp_type == "bodex":
                        if isinstance(self._last_traj_qpos, dict):
                            joint_names = self.planner.init_js_ik_solver.joint_names
                            q_list = [self._last_traj_qpos[name] for name in joint_names]  
                            self._last_traj_qpos = np.array(q_list, dtype=np.float32)[:self.planner.init_js_ik_solver.dof]
                        lock_links_names = spec.meta.get("lock_links",None)
                        target_orientation = spec.meta.get("orientation", None)
                        if target_orientation is not None:
                            trajs, jnames = self.planner.batch_plan_for_move_bodex(
                            position=T_robot_goal[:3, 3],
                            orientation=t3d.quaternions.mat2quat(T_robot_goal[:3, :3]),
                            current_joint_states=self._last_traj_qpos,
                            world_layout=self.task.layout,
                            lock_links_names=lock_links_names

                        )
                        else:
                            trajs, jnames = self.planner.batch_plan_for_move_bodex(
                                position=T_robot_goal[:2, 3],
                                orientation=target_orientation,
                                current_joint_states=self._last_traj_qpos,
                                world_layout=self.task.layout,
                                lock_links_names=lock_links_names
                            )
                    else:
                        trajs, jnames = self.planner.batch_plan_for_move(
                            [goal_pose_robot_frame],
                            current_joint_states=self._last_traj_qpos,
                            world_layout=self.task.layout
                        )

                except Exception as e:
                    print(f"Planning to target pose failed with error: {e}")
                    return False
                
                traj = []
                for qpos in trajs[-1]:
                    from simple.robots.protocols import Humanoid
                    init_qpos = self.robot.init_joint_states
                    if isinstance(self.robot , Humanoid):
                        traj.append(pad_to_full_qpos(dict(zip(jnames, qpos)), init_qpos))
                    else:
                        traj.append(dict(zip(jnames, qpos)))
                
                self.queue_follow_path_with_eef(traj, eef_state, hand_uid=hand_uid)
                self._last_traj_qpos = traj[-1] # HACK store for retreat planning
                
                self._last_eef_orientation = target_orientation
                if self._last_eef_orientation is None:
                    self._last_eef_orientation = t3d.euler.euler2quat(np.pi, 0, 0)
                
            elif isinstance(spec, GraspObjectSpec):
                target_uid = spec.meta["target_uid"]
                target_actor = self.task.layout.actors["target"]
                robot_actor = self.task.layout.actors["robot"] 

                assert target_uid == target_actor.uid
                object_pose = target_actor.pose
                
                assert isinstance(target_actor, ObjectActor)
                assert isinstance(target_actor.asset, SpatialAnnotated)

                # grasp poses in SIMPLE world frame
                grasp_type = spec.meta.get("grasp_type", "parallel_gripper")
                grasp_bias = spec.meta.get("grasp_bias", None)
                store_grasp_key = spec.meta.get("store_grasp_key", None)
                use_negative_x_filter = spec.meta.get("use_negative_x_filter", False)
                approach_axis = spec.meta.get("approach_axis", 'z')
                
                if grasp_type == "parallel_gripper":

                    if use_negative_x_filter:
                        
                        reference_grasps = None
                        if "left_arm_grasps" in self._stored_grasps:
                            reference_grasps = self._stored_grasps["left_arm_grasps"]
                            print(f"[MP Agent] Using {len(reference_grasps)} stored left arm grasps as reference")
                        
                        grasps = GSNet.load_cached_grasps_filter_by_negative_x(
                            target_actor.asset,
                            stable_idx=self.target_stable_index,
                            target_pose=object_pose,
                            max_grasps=self.plan_batch_size,
                            angle_threshold=20.0,
                            reference_grasps=reference_grasps
                        )
                        print(f"[MP Agent] Found {len(grasps)} grasps approaching from -X direction")
                        self.target_stable_index = grasps[0]["stable_idx"] # save target stable_idx of env
                    else:
                        grasps = GSNet.load_cached_grasps(
                            target_actor.asset,
                            target_pose=object_pose,
                            max_grasps=self.plan_batch_size,
                            bias=grasp_bias
                        )
                        if grasps is None:
                            print(f"[MP Agent] No grasps found for object {target_uid}")
                            return False 
                        self.target_stable_index = grasps[0]["stable_idx"] # save target stable_idx of env
                        if store_grasp_key is not None:
                            self._stored_grasps[store_grasp_key] = grasps
                            print(f"[MP Agent] Stored {len(grasps)} grasps with key '{store_grasp_key}'")
                elif grasp_type == "bodex":
                    grasps = Bodex.load_cached_grasps(self.robot, target_actor.asset, target_pose=object_pose, max_grasps=self.plan_batch_size,hand_uid=hand_uid)
                else:
                    raise ValueError(f"Unsupported grasp type: {grasp_type}")

                # Pad more grasps if not enough
                object_grasp_poses = (
                    grasps[:self.plan_batch_size]
                    if len(grasps) >= self.plan_batch_size
                    else grasps + [grasps[0]]*(self.plan_batch_size - len(grasps))
                )
                
                plan_grasp_proses = deepcopy(object_grasp_poses)
                assert isinstance(self.robot, Graspable)
                for i in range(len(object_grasp_poses)):
                    # convert grasp to robot base frame (because curobo plans in robot base frame)
                    goal_pose = self.robot.get_grasp_pose_wrt_robot(
                        object_grasp_poses[i],
                        pregrasp=spec.meta.get("pregrasp", False),
                        robot_pose = robot_actor.pose.as_matrix()
                    )
                    # plan_grasp_proses.append(goal_pose)
                    plan_grasp_proses[i]["position"] = goal_pose[0]
                    plan_grasp_proses[i]["orientation"] = goal_pose[1]

                ### Visualization the first grasp for debugging
                ### convert from robot base frame to world frame for visualization
                if self.debug:
                    self.task.layout.add_visual_grasp("target", grasps[0])                       
                    
                    # Visualize planned grasp in robot frame
                    robot_pose = robot_actor.pose.as_matrix()
                    T_plan_grasp = np.eye(4)
                    T_plan_grasp[:3, 3] = plan_grasp_proses[0]["position"]
                    T_plan_grasp[:3, :3] = t3d.quaternions.quat2mat(plan_grasp_proses[0]["orientation"])
                    goal_in_world_frame = robot_pose @ T_plan_grasp
                    goal_pose_vec = np.concatenate([goal_in_world_frame[:3, 3], t3d.quaternions.mat2quat(goal_in_world_frame[:3, :3])])
                    self.task.layout.add_visual_frame("eef", goal_pose_vec.tolist())

                    # visualize bodex hand pose
                    if grasp_type == "bodex":
                        for key,pose in grasps[0]["link_poses"].items():
                            T_world_linkpose = np.eye(4)
                            T_world_linkpose[:3, 3] = pose.position.cpu().numpy().squeeze(0)
                            T_world_linkpose[:3, :3] = t3d.quaternions.quat2mat(pose.quaternion.cpu().numpy().squeeze(0))
                            # goal_in_world_frame = T_plan_bodexgrasp
                            bodex_goal_pose_vec = np.concatenate([T_world_linkpose[:3, 3], t3d.quaternions.mat2quat(T_world_linkpose[:3, :3])])
                            self.task.layout.add_visual_frame(f"bodex_{key}",  bodex_goal_pose_vec.tolist())
                            
                try:
                    if grasp_type == "parallel_gripper":
                        trajs, jnames = self.planner.batch_plan_for_approach(
                            plan_grasp_proses, # [np.concatenate([p["position"], p["orientation"]], axis=0) for p in plan_grasp_proses], 
                            world_layout=self.task.layout,
                            approach_axis=approach_axis
                        )
                    elif grasp_type == "bodex":
                        # FIXME
                        def transform_linkpose_to_robot(link_pose,robot_pose):
                            position = link_pose.position.cpu().numpy().squeeze(0)
                            quaternion = link_pose.quaternion.cpu().numpy().squeeze(0)

                            T_world_linkpose = np.eye(4)
                            T_world_linkpose[:3, 3] = position
                            T_world_linkpose[:3, :3] = t3d.quaternions.quat2mat(quaternion)

                            T_world_robot = robot_pose
                            T_robot_linkpose = np.linalg.inv(T_world_robot) @ T_world_linkpose

                            new_position = T_robot_linkpose[:3,3]

                            new_quaternion = t3d.quaternions.mat2quat(T_robot_linkpose[:3,:3])

                            link_pose.position = torch.from_numpy(new_position).float().to(link_pose.position.device).unsqueeze(0)

                            link_pose.quaternion = torch.from_numpy(new_quaternion).float().to(link_pose.position.device).unsqueeze(0)
                            return link_pose

                        for key,pose in grasps[0]["link_poses"].items():
                            grasps[0]["link_poses"][key] = transform_linkpose_to_robot(pose, robot_actor.pose.as_matrix())

                        #get lock_links
                        lock_links_names = spec.meta.get("lock_links",None)

                        current_joint_qpos = self.robot.get_robot_qpos()


                        
                        trajs, jnames = self.planner.batch_plan_for_approach_bodex(
                            # grasps[0],
                            [np.concatenate([p["position"], p["orientation"]], axis=0) for p in plan_grasp_proses], 
                            [g["grasp_qpos"] for g in plan_grasp_proses],
                            [g["squeeze_qpos"] for g in plan_grasp_proses],
                            [g["lift_qpos"] for g in plan_grasp_proses],
                            link_poses=grasps[0]["link_poses"], # remove this hack
                            debug_grasp=grasps[0],
                            world_layout=self.task.layout, # collision avoidance
                            lock_links_names=lock_links_names,
                            current_joint_qpos=current_joint_qpos
                        )
                    else:
                        raise ValueError(f"Unsupported grasp type: {grasp_type}")
                    
                except Exception as e:
                    print(f"Planning failed with error: {e}")
                    # raise e
                    return False
                
                init_qpos = self.robot.init_joint_states
                traj = []
                for qpos in trajs[-1]: # only taks the last successful one
                    from simple.robots.protocols import Humanoid
                    if isinstance(self.robot, Humanoid):
                        traj.append(pad_to_full_qpos(dict(zip(jnames, qpos)), init_qpos))
                    else:
                        # not support g1
                        traj.append(dict(zip(jnames, qpos)))
                # self.queue_follow_path_with_eef(traj, "open_eef") 
                
                #FIXME only for g1 handover
                keep_force = spec.meta.get("keep_force",False)
                self.queue_follow_path(traj, keep_force=keep_force)

                # self._grasp_qpos = traj[-1] # HACK store for lift planning
                self._last_traj_qpos = traj[-1] # HACK store for lift planning


            elif isinstance(spec, LiftSpec):
                grasp_type = spec.meta.get("grasp_type", "parallel_gripper")
                lift_height = spec.meta.get("up", 0.1)
                step_distance = spec.meta.get("step_distance", 0.01)
                eef_state = spec.meta.get("eef_state", "close_eef")
                
                try:
                    if grasp_type == "bodex":
                        traj = []
                        init_qpos = self.robot.init_joint_states
                        plan_traj = self.planner.batch_plan_for_lift_bodex(self._last_traj_qpos , lift_height, step_distance=0.002)
                        for qpos in plan_traj:
                            traj.append(pad_to_full_qpos(qpos, init_qpos))
                    else:
                        traj = self.planner.batch_plan_for_lift(self._last_traj_qpos , lift_height, step_distance=step_distance)
                except Exception as e:
                    # raise e
                    print(f"Lift planning failed with error: {e}")
                    return False
                
                # if isinstance(self.robot, Humanoid):
                #     traj
                self.queue_follow_path_with_eef(traj, eef_state, hand_uid=hand_uid)
                # self._lift_qpos_dict = traj[-1] # HACK store for move planning
                if grasp_type == "bodex":
                    self._last_traj_qpos = np.array(list(traj[-1].values()), dtype=np.float32)[:self.planner.init_js_ik_solver.dof]
                else:
                    self._last_traj_qpos = traj[-1]
            
            elif isinstance(spec, LowerSpec):
                grasp_type = spec.meta.get("grasp_type", "parallel_gripper")
                lower_height = spec.meta.get("down")
                try:
                    if grasp_type == "bodex":
                        traj = []
                        init_qpos = self.robot.init_joint_states
                        plan_traj = self.planner.batch_plan_for_lift_bodex(self._last_traj_qpos, lift_height=lower_height, step_distance=-0.002)
                        for qpos in plan_traj:
                            traj.append(pad_to_full_qpos(qpos, init_qpos))
                    else:
                        traj = self.planner.batch_plan_for_lift(self._last_traj_qpos, lift_height=lower_height, step_distance=-0.01) # type: ignore
                except Exception as e:
                    # raise e
                    print(f"Lower planning failed with error: {e}")
                    return False
                
                self.queue_follow_path_with_eef(traj, "close_eef")
                if grasp_type == "bodex":
                    self._last_traj_qpos = np.array(list(traj[-1].values()), dtype=np.float32)[:self.planner.init_js_ik_solver.dof]
                else:
                    self._last_traj_qpos = traj[-1] # HACK store for move planning
                # self._lower_qpos = np.array(list(traj[-1].values()), dtype=np.float32)
                
            elif isinstance(spec, RetreatSpec):
                if isinstance(self._last_traj_qpos, dict):
                        joint_names = self.planner.init_js_ik_solver.joint_names
                        q_list = [self._last_traj_qpos[name] for name in joint_names]  
                        self._last_traj_qpos = np.array(q_list, dtype=np.float32)[:self.planner.init_js_ik_solver.dof]
                grasp_type = spec.meta.get("grasp_type", "parallel_gripper")
                try:
                    if grasp_type == "bodex":
                        lock_links_names = spec.meta.get("lock_links",None)
                        trajs, jnames = self.planner.batch_plan_for_retreat_bodex(
                            current_joint_states=self._last_traj_qpos,
                            world_layout=self.task.layout,
                            lock_links_names=lock_links_names
                        )
                        traj = []
                        for qpos in trajs[-1]:
                            from simple.robots.protocols import Humanoid
                            init_qpos = self.robot.init_joint_states
                            if isinstance(self.robot , Humanoid):
                                traj.append(pad_to_full_qpos(dict(zip(jnames, qpos)), init_qpos))
                            else:
                                traj.append(dict(zip(jnames, qpos)))

                    else:
                        traj = self.planner.batch_plan_for_lift(self._last_traj_qpos)
                except Exception as e:
                    # raise e
                    print(f"Retreat planning failed with error: {e}")
                    return False
                self.queue_follow_path_with_eef(traj, "open_eef")
                self._last_traj_qpos = traj[-1] 

            # FOR WHOLEBODY LOCOMOTION 
            # command: [vx, yaw,vy,height,torso yaw, torso pitch,torso roll]      
            elif isinstance(spec, StandSpec):
                target_yaw = spec.meta.get("target_yaw", 0)
                command = [0,target_yaw,0,0,0,0,0,0]
                keep_waist_pose = spec.meta.get("keep_waist_pose", False)
                steps = spec.meta.get("steps", 60)
                for _ in range(steps):
                    self.queue_loco_command(
                        command=command.copy(),
                        motion_type="stand",
                        keep_waist_pose=keep_waist_pose
                    )

                """ # TODO DEBUG
                for i in range(20):
                    jnames = self.task.robot.joint_names
                    qpos = np.zeros((43,), dtype=np.float32)
                    qpos[-5] = -0.85/20 * (i+1) # right thumb 2
                    target_qpos = dict(zip(jnames, qpos))
                    self.queue_move_qpos(target_qpos) """
                    
            elif isinstance(spec, WalkSpec):
                keep_waist_pose = spec.meta.get("keep_waist_pose", False)
                walk_distance = spec.meta.get("target_distance", 0.5)
                target_yaw = spec.meta.get("target_yaw", 0)
                command = [0,0,0,0,0,0,0,0]
                vx = spec.meta.get("vx", 0.1)
                vy = spec.meta.get("vy", 0)
                command[0] = vx
                command[2] = vy
                command[1] = target_yaw
                command[3] = spec.meta.get("height", 0.0)
                command[4] = spec.meta.get("torso_yaw", 0)
                command[5] = spec.meta.get("torso_pitch", 0)
                command[6] = spec.meta.get("torso_roll", 0)

                x_origin = self.desired_robot_pose[0]
                y_origin = self.desired_robot_pose[1]
                #calculate the desired robot pose
                x_desired = x_origin+walk_distance*np.cos(target_yaw)
                y_desired = y_origin+walk_distance*np.sin(target_yaw)
                self.desired_robot_pose[0] = round(x_desired,3)
                self.desired_robot_pose[1] = round(y_desired,3)
                self.desired_robot_pose[3:] = t3d.euler.euler2quat(0,0,target_yaw)

                desired_robot_pose = self.desired_robot_pose.copy()
                # TODO
                if abs(vx) < abs(vy):
                    steps = int(walk_distance/10/0.002/(abs(vy)-0.1) + 30)
                else:
                    steps = int(walk_distance/10/0.002/(abs(vx)-0.1) + 30)
                
                for _ in range(steps):
                    self.queue_loco_command(
                        command=command.copy(),
                        desired_robot_pose=desired_robot_pose,
                        motion_type="walk",
                        keep_waist_pose=keep_waist_pose,
                        keep_waist_pitch=True
                    )

            elif isinstance(spec, TurnSpec):
                keep_waist_pose = spec.meta.get("keep_waist_pose", False)
                target_yaw = spec.meta.get("target_yaw", 0)
                command = [0, 0, 0, 0, 0, 0, 0, 1]
                command[1] = target_yaw
                command[0] = spec.meta.get("vx", 0.0)
                command[3] = spec.meta.get("height", 0.0)
                self.desired_robot_pose[3:] = t3d.euler.euler2quat(0,0,target_yaw)
                desired_robot_pose = self.desired_robot_pose.copy()
                
   
                total_steps = 100
                turn_ratio = 0.7  
                active_steps = int(total_steps * turn_ratio) 
                
                start_yaw = self.last_target_yaw
                yaw_diff = target_yaw - start_yaw
                
                for step in range(total_steps):

                    if step < active_steps:

                        ratio = step / (active_steps - 1)
                        curve_progress = ratio
                        yaw = start_yaw + yaw_diff * curve_progress
                    

                    else:
                        yaw = target_yaw
                    
                    command[1] = yaw
                        
                    self.queue_loco_command(
                        command=command.copy(),  
                        desired_robot_pose=desired_robot_pose,
                        motion_type="turn",
                        keep_waist_pose=keep_waist_pose
                    )


                self.last_target_yaw = target_yaw
            elif isinstance(spec, HeightAdjustSpec):
                keep_waist_pose = spec.meta.get("keep_waist_pose", False)
                target_yaw = spec.meta.get("target_yaw", 0)
                height = spec.meta.get("height", 0)
                vx = spec.meta.get("vx", 0)
                vy = spec.meta.get("vy", 0)
                command = [vx, target_yaw,vy,height,0,0,0,0]
                for _ in range(60):
                    self.queue_loco_command(command=command,desired_height=height,motion_type="height_adjust",keep_waist_pose=keep_waist_pose)
        
            else:
                raise ValueError(f"Unknown subtask spec: {type(spec)}")
            
        return True
    
    def reset(self):
        """Reset the agent state, including subtask index for multi-phase planning."""
        super().reset()
        Bodex.reset()
        self._subtask_index = 0
        self.last_target_yaw = 0

    