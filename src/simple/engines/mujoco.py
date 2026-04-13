"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Tuple, Any, List

if TYPE_CHECKING:
    # from simple.core.asset import Asset
    # from simple.core.actor import Actor
    from simple.core.task import Task
    from simple.core.actor import RobotActor, ObjectActor, CameraEntity, ArticulatedObjectActor
    from simple.assets.primitive import Primitive # , Box
    # from simple.sensors.config import CameraCfg
    

import numpy as np
from simple.core.simulator import Simulator

from simple.core.object import SemanticAnnotated # , Object, SpatialAnnotated
# from simple.core.robot import Robot
from simple.robots.protocols import Controllable
from simple.utils import resolve_data_path
# from simple.assets.objects import ObjectsAsset
# from simple.assets.graspnet import Graspnet1BAsset
# from dm_control import mjcf
import mujoco

from PIL import Image
import os
import cv2
import transforms3d as t3d

xyzw_to_wxyz = lambda q: np.array([q[3], q[0], q[1], q[2]])

class MujocoSimulator(Simulator):

    def __init__(self, task:Task, render_hz:int=30, physics_dt: float = 0.002, headless=True) -> None:
        self.task = task
        self.render_hz = task.metadata["render_hz"] if "render_hz" in self.task.metadata else render_hz
        self.physics_dt = task.metadata["physics_dt"] if "physics_dt" in self.task.metadata else physics_dt

        # TODO move to reset? 
        # self.mj_physics = None
        self.mj_worldbody = None
        self.robot_mjcf = None
        self.render_option = None
        self.renderers = {}
        self.default_camera_name = None
        self.render_step = 0
        self.last_action = [0 for _ in range(17)]#TODO

        self.viewer = None
        self.headless=headless
        
        self.need_gravity = self.task.metadata.get("need_gravity", False)
        self._is_sonic = None
        self.articulated_object_joints = None

    def update_layout(self, **kwargs) -> None:
        # FIXME
        self._is_sonic = ("sonic_config" in kwargs)
        self._setup_scene(**kwargs)

    def step(self, render=True, render_robot_mask=False, **kwargs) -> Dict[str, np.ndarray] | None:
        if self.need_gravity:
            if self.task.robot.command is None:# probably resetting?
                mujoco.mj_step(self.mjModel, self.mjData, nstep=1)
            else:
                self.task.robot.step(self.task.robot.command, replay=self.task.robot.is_replay ,eval = self.task.robot.is_eval)
        else:
            timestep = self.mjModel.opt.timestep
            current_physics_time = self.mjData.time
            num_physics_steps = int(((self.render_step + 1) / self.render_hz - current_physics_time) // timestep)
            assert num_physics_steps > 0 and num_physics_steps < 100000, "warning: why so many physics steps?"
            mujoco.mj_step(self.mjModel, self.mjData, nstep=num_physics_steps)

        self.render_step += 1
        if self.viewer is not None:
            self.viewer.sync()

        # updata self.task.layout.objects pose, at beginning few steps don't update
        if self.render_step > 5:
            for objtype , mj_obj in self.mj_objects.items():
                self.task.layout.actors[objtype].pose.position = list(mj_obj.xpos)
                self.task.layout.actors[objtype].pose.quaternion = list(mj_obj.xquat)
            self.task.layout.actors["robot"].pose.position = list(np.round(self.mjData.qpos[:3], 3))
            self.task.layout.actors["robot"].pose.quaternion = list(np.round(self.mjData.qpos[3:7], 3))

        if render:
            return self.render(render_robot_mask)
    
    def set_states(self, states: Dict[str, Any]) -> None: 
        raise NotImplementedError

    def get_states(self) -> Dict[str, Any]:
        obj_positions = [mj_obj.xpos for mj_obj in self.mj_objects.values()]
        obj_orientations = [mj_obj.xquat for mj_obj in self.mj_objects.values()]
        # joint_state = np.array([joint.qpos[0] for joint in self.joints])
        joint_state = self.task.robot.get_robot_qpos()
        robot_position = np.round(self.mjData.qpos[:7], 4)
        if self.articulated_object_joints is not None:
            articulated_joints_state = {}
            for articulate_joint in self.articulated_object_joints:
                # articulated_joints_state[articulate_joint] = self.mjData.joint(articulate_joint).qpos[0]
                raw_qpos = self.mjData.joint(articulate_joint).qpos[0]


                wrapped_qpos = (raw_qpos + np.pi) % (2 * np.pi) - np.pi

                articulated_joints_state[articulate_joint] = wrapped_qpos
            articulate_object_position = np.round(self.mjData.joint("articulate_floating_base").qpos, 4)
            return [self.render_step, joint_state, self.obj_names, obj_positions, obj_orientations, robot_position, articulated_joints_state, articulate_object_position]
       
            
        else:
            return [self.render_step, joint_state, self.obj_names, obj_positions, obj_orientations, robot_position]


    def _setup_scene(self, **kwargs):
        from simple.core.actor import RobotActor, ObjectActor, ArticulatedObjectActor, CameraEntity
        from simple.assets.primitive import Primitive

        # https://mujoco.readthedocs.io/en/stable/computation/index.html
        # https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip
        mjSpec = mujoco.MjSpec()
        mjSpec.option.timestep = self.physics_dt
        mjSpec.option.impratio = 10
        mjSpec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        mjSpec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        mjSpec.option.noslip_iterations = 2
        
        mj_worldbody = mjSpec.worldbody
        
        mj_worldbody.add_light(
            # type="directional_light",
            pos=[0, 0, 1.5], 
            dir=[0, 0, -1],
            # directional=True, 
            castshadow=False,
            # ambient=1.5
        )

        for objtype, actor in self.task.layout.actors.items():
            if isinstance(actor, ObjectActor):
                # This is a string, which means it's a name of an asset
                # asset = self.task.layout.assets[actor]
                # actor = Actor.from_asset(asset)
                self._build_object(mjSpec, mj_worldbody, actor)
            elif isinstance(actor, RobotActor):
                self._build_robot(mjSpec, mj_worldbody, actor) # HACK make sure called first
            elif isinstance(actor, Primitive):
                self._build_primitive(mjSpec, mj_worldbody, actor, table_name=objtype)
            elif isinstance(actor, ArticulatedObjectActor):
                self._build_articulated_object(mjSpec, mj_worldbody, actor)
            else:
                raise TypeError(f"Unsupported actor type: {type(actor)}")
            
        self.mj_worldbody = mj_worldbody

        for cname, camera in self.task.layout.cameras.items():
            self._build_camera(cname, camera)

        if not self._is_sonic:
            z_minus = self.task.layout.scene.table.pose.position[2] + 0.5 * self.task.layout.scene.table.size[2]
            if hasattr(self.task.robot, "z_offset"):
                # HACK for vega robot base height
                z_minus += self.task.robot.z_offset-self.robot_z
        else:
            z_minus = 0.0

        # add ground plane
        ground = mj_worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,  # type: ignore
            name="ground",
            size=[0, 0, 1],
            pos=[0, 0, -z_minus],
            material="groundplane"
        )
        # add some friction for contact stability
        ground.friction = [1.0, 0.005, 0.0001]  # [sliding, torsional, rolling]

        # 5. disable gravity (for better PID control of arms)
        self.mjModel=mjSpec.compile()
        self.mjData=mujoco.MjData(self.mjModel)
        self.mjSpec=mjSpec

        if not self.need_gravity:
            self.mjModel.opt.gravity= (0,0,0) # type: ignore
            # physics = mjcf.Physics.from_mjcf_model(mjcf_model)
            # physics.model.opt.gravity= (0,0,0) # type: ignore
            pseudo_gravity = np.zeros(6)
            pseudo_gravity[2] = -9.81 * 0.1#0.1 # 0.1 is the mass of the object. When changing the mass, this value should be changed accordingly!
        else:
            self.mjModel.opt.gravity = (0,0,-9.81)
            pseudo_gravity = np.zeros(6)
            pseudo_gravity[2] = -9.81 * 0.1

        mj_objects = {}
        obj_names = []
        for objtype, actor in self.task.layout.actors.items():
            if isinstance(actor, ObjectActor):
                label = actor.asset.uid
                name = actor.asset.uid
                uid = actor.asset.uid
                if isinstance(actor.asset, SemanticAnnotated):
                    label = actor.asset.label
                    name = actor.asset.name
                    uid = actor.asset.uid
                # mj_obj=physics.data.body(f"{label}")
                # mj_obj.xfrc_applied = pseudo_gravity
                mj_obj=self.mjData.body(f"{label}")
                if objtype == "container":
                    gravity= pseudo_gravity.copy()
                    gravity[2] = -9.81*1
                    mj_obj.xfrc_applied = gravity
                else:
                    mj_obj.xfrc_applied = pseudo_gravity
                mj_objects[objtype] = mj_obj
                obj_names.append(label)

        self.mj_objects = mj_objects
        self.obj_names = obj_names

        # if isinstance(self.task.robot, Controllable):
        #     self.task.robot.initialize_controller(data)

        # 6. setup control
        assert isinstance(self.task.robot, Controllable), "Task robot is None."
        self.joints, self.actuators=self.task.robot.setup_control(self.mjData, self.mjModel, mjSpec=self.mjSpec)

        

        # CHECK
        mujoco.mj_forward(self.mjModel, self.mjData)

        # 8. rendering
        self.render_option = mujoco.MjvOption() # type: ignore
        mujoco.mjv_defaultOption(self.render_option) # type: ignore
        # self.render_option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 1 # type: ignore

        # in case of forgetting to close the env before reset
        if (hasattr(self, "renderers") and len(self.renderers) > 0):
            self.close()

        self.renderers = {}
        for cname, camera in self.task.layout.cameras.items():
            self.renderers[cname] = mujoco.Renderer(
                self.mjModel,
                height=camera.resolution[1],
                width=camera.resolution[0]
            ) # type: ignore

        if not self._is_sonic:
            if self.viewer is not None:
                self.viewer.close()

            if not self.headless:
                # This will display the int running physics
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.mjModel, self.mjData)
            
        # ?. reset render step
        self.render_step = 0

    def _build_object(self, mjSpec, mjWorld, actor: ObjectActor):
        # asset_id = actor.asset.uid

        # TODO primitive types
        collision_meshes = actor.asset.collision_meshes_mujoco
        num_convex = len(collision_meshes)

        label = actor.asset.uid
        name = actor.asset.uid
        if isinstance(actor.asset, SemanticAnnotated):
            label = actor.asset.label
            name = actor.asset.name

        for i in range(num_convex):
            mjSpec.add_mesh(
                name=f'{label}_mesh_convex{i}', 
                file=collision_meshes[i],
            )

        mj_obj=mjWorld.add_body(
            name=label, 
            pos=actor.pose.position, 
            quat=actor.pose.quaternion)


        num_convex = len(collision_meshes)
        for i in range(num_convex):
            mj_obj.add_geom(
                name=f"{label}_convex_{i}", 
                meshname=f"{label}_mesh_convex{i}", 
                type=mujoco.mjtGeom.mjGEOM_MESH,
                # opposing slip in the tangent plane, rotation around the contact normal 
                # and rotation around the two axes of the tangent plane
                condim=4,  
                # total mass 0.1 helps preventing slipping
                mass=0.1/num_convex, 
                # rubber on rough ground: large static, sliding and torisonal friction
                friction=[0.8, 0.05, 0.005],  
                rgba=[1, 1, 1, 1],
                # stiff contact and no oscillation
                solref = [0.005, 2]
            )
        mj_obj.add_freejoint(name=f'{label}_joint')
    
    def _build_articulated_object(self, mjSpec, mjWorld, actor: ArticulatedObjectActor):
        """Build the articulated object in the Mujoco simulator."""
        articulated_object_mjcf = mujoco.MjSpec.from_file(resolve_data_path(actor.asset.mjcf_path, auto_download=True))
        # for geom in articulated_object_mjcf.geoms:
        #     geom.density = 30.0

        geoms = articulated_object_mjcf.geoms
        num_geoms = len(geoms)
        
        # if num_geoms > 0:
           
        #     target_total_mass = 2  
        #     mass_per_geom = target_total_mass / num_geoms
            
        #     for geom in geoms:
        #         geom.mass = mass_per_geom

        self.articulated_object_joints = []
        

        for joint in articulated_object_mjcf.joints:
            if joint.name and joint.name.startswith("articulate_joint"):
                self.articulated_object_joints.append(joint.name)


        frame = mjWorld.add_frame(pos=actor.pose.position, quat=actor.pose.quaternion)
        mjSpec.attach(articulated_object_mjcf, frame=frame)
        self.articulated_object_mjcf = articulated_object_mjcf

    # def _build_articulated_object(self, mjSpec, mjWorld, actor: ArticulatedObjectActor):
    #     """Build the articulated object in the Mujoco simulator with volume-proportional mass."""
    #     articulated_object_mjcf = mujoco.MjSpec.from_file(resolve_data_path(actor.asset.mjcf_path, auto_download=True))
        
    #     # --- Step 1: Perform a "dummy compile" with a base density to probe the total volume ---
    #     base_density = 1000.0  # Assume an initial density of 1000 kg/m^3 (density of water)
    #     for geom in articulated_object_mjcf.geoms:
    #         # CRITICAL: mass must be strictly 0.0 so MuJoCo calculates mass using density * volume.
    #         # Note: Ensure your XML has <compiler boundmass="0.001" boundinertia="0.000001" /> 
    #         # to prevent crashes on zero-volume geoms.
    #         geom.mass = 0.0  
    #         geom.density = base_density
            
    #     # Compile a temporary model. This is very fast and forces MuJoCo to compute 
    #     # the exact volumes, inertias, and masses under the hood.
    #     temp_model = articulated_object_mjcf.compile()
        
    #     # Get the current total mass calculated using the base_density
    #     current_total_mass = sum(temp_model.body_mass)
        
    #     # --- Step 2: Calculate the true density required to hit the target mass ---
    #     target_total_mass = 2.0  # The strict total mass you want for the object
        
    #     if current_total_mass > 0:
    #         # Scale factor = Target Mass / Current Mass
    #         density_scale = target_total_mass / current_total_mass
    #         target_density = base_density * density_scale
            
    #         # --- Step 3: Apply the correct density back to all geometries ---
    #         for geom in articulated_object_mjcf.geoms:
    #             geom.mass = 0.0             # Ensure mass is still 0.0 to force density usage
    #             geom.density = target_density

    #     # Finally, attach the perfectly balanced object to the main world frame
    #     frame = mjWorld.add_frame(pos=actor.pose.position, quat=actor.pose.quaternion)
    #     mjSpec.attach(articulated_object_mjcf, frame=frame)
        
    #     self.articulated_object_mjcf = articulated_object_mjcf


        
      





    def _build_robot(self, mjSpec, mjWorld, actor: RobotActor):
        """Build the robot in the Mujoco simulator."""
        # try: 
        # 1. resolve some local file path which is need to run the script
        robot_mjcf=mujoco.MjSpec.from_file(resolve_data_path(actor.robot.mjcf_path, auto_download=True))
        # except FileNotFoundError:
        #     # 2. catch file not found error if files are not auto-downloaded
        #     from huggingface_hub import snapshot_download
        #     local_data_dir =self.resolve_data_path()
        #     # robot=actor.robot.mjcf_path.split('/')[-2]
        #     print(f"Auto downloading assets to {local_data_dir} ...")
        #     # # 3. Now auto download it using huggingface-hub
        #     snapshot_download(
        #         repo_id="SIMPLE-org/SIMPLE",
        #         allow_patterns=["robots.zip"],
        #         local_dir=local_data_dir,
        #         repo_type="dataset",
        #         # resume_download=True,
        #         token="YOUR_HF_TOKEN",
        #     )
        #     # 4. unzip the downloaded zip file
        #     import zipfile
        #     zip_path = os.path.join(local_data_dir, "robots.zip")
        #     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        #         zip_ref.extractall(local_data_dir)
        #     if os.path.exists(zip_path):
        #         os.remove(zip_path)
        #         print(f"Deleted {zip_path}")

        #     robot_mjcf=mujoco.MjSpec.from_file(self.resolve_data_path(actor.robot.mjcf_path))

        for g in robot_mjcf.geoms:
            if "hand" in (g.name or g.meshname) or "gripper" in (g.name or g.meshname):
                # This solimp and solref comes from the Shadow Hand xml
                # They can generate larger force with smaller penetration
                # The body will be more "rigid" and less "soft"
                g.solimp[:3] = [0.9, 0.99, 0.0001]
                g.solref[:2] = [0.005, 1]

        self.robot_z = actor.pose.position[2]
        frame=mjWorld.add_frame(pos=actor.pose.position,quat=actor.pose.quaternion) 
        mjSpec.attach(robot_mjcf,frame=frame)

        self.robot_mjcf = robot_mjcf

    def _build_primitive(self, mjSpec, mjWorld, actor: Primitive, table_name: str = 'table'):
        from simple.assets.primitive import Box
        if isinstance(actor, Box):
            table_size = 0.5 * np.array(actor.size)
            table_position = actor.pose.position.copy()
            if not self._is_sonic and table_name == 'table': # allow personalization of other tables
                table_z = - 0.5 * actor.size[2] 
                table_position[2] = table_z 
            
            table=mjWorld.add_body(name=table_name,
                                   pos=table_position,
                                   quat=actor.pose.quaternion)
            # Use table_name as geom name to avoid conflicts when multiple tables exist
            table.add_geom(name=f"{table_name}_geom",
                           type=mujoco.mjtGeom.mjGEOM_BOX, 
                           size=table_size,
                           condim=6, 
                           friction=[2, 0.04, 0.0005], # 0.8
                           priority=10,)
        else:
            raise TypeError(f"Unsupported primitive type: {type(actor)}")

    def _build_camera(self, cname:str, camera: CameraEntity):
        q_isaac_mujoco = t3d.quaternions.mat2quat(np.array([
            [ 0,  0, -1],
            [-1,  0,  0],
            [ 0,  1,  0]
        ]))
        W, H = camera.resolution
        fovy = 2 * np.arctan(H / (2 * camera.fy)) * 180 / np.pi
        if camera.mount == "eye_on_base":
            # cam_pose = self.task.layout.actors["robot"].pose * camera.pose
            cam_pose = camera.pose
            cam_q = t3d.quaternions.qmult(cam_pose.quaternion, q_isaac_mujoco)

            self.mj_worldbody.add_camera(
                name=cname,
                pos=cam_pose.position,
                quat=cam_q,
                fovy=fovy,
            )
        elif camera.mount == "eye_in_hand":
            self.mj_worldbody.add_camera(
                name=cname,
                pos=[1.5, 0., 0.8],  #  FIXME
                xyaxes=[0,1,0,-0.5,0,1], 
                fovy=fovy
            )
        elif camera.mount == "eye_in_head":
            torso_body = None
            for body in self.mj_worldbody.find_all('body'):
                if body.name == "torso_link":
                    torso_body = body
                    break
            assert torso_body is not None
            """ 
            I know this numbers look crazy!
            I obtain the first coordinate using isaacsim (g1_29dof_wholebody_dex3.usd)
            and obtain the second coordinates using mujoco (g1_29dof_wholebody_dex3.xml)
            and then i add them up by LUCK and it works! 
            """
            DEFAULT_HEAD_CAM_POSITION = np.array([0.05366004+0.0039635, 0.01752999 + 0, 0.4738702 + -0.044], dtype=np.float32)
            DEFAULT_HEAD_CAM_ORIENTATION = np.array([0.91496, 0.0, 0.40355, 0.0], dtype=np.float32)
            q = np.asarray(camera.pose.quaternion, dtype=np.float32)
            is_identity_quat = np.allclose(q[1:], 0.0, atol=1e-6) and np.isclose(abs(float(q[0])), 1.0, atol=1e-6)
            assert is_identity_quat, f"Expected eye_in_head camera quaternion to be identity (wxyz)"

            torso_body.add_camera(
                name=cname,
                pos=DEFAULT_HEAD_CAM_POSITION + camera.pose.position,  # FIXME
                quat=t3d.quaternions.qmult(DEFAULT_HEAD_CAM_ORIENTATION, q_isaac_mujoco),
                fovy=fovy
            )
        else:
            raise ValueError(f"Unsupported camera mount: {camera.mount}")


    def get_robot_qpos(self) -> dict[str,float]:
        from simple.robots.protocols import Controllable
        if not isinstance(self.task.robot, Controllable):
            raise TypeError("The task robot is not a Robot instance.")
        return self.task.robot.get_robot_qpos()
    
    def get_actuators_action(self) -> dict[str,float]:
        from simple.robots.protocols import Controllable
        if not isinstance(self.task.robot, Controllable):
            raise TypeError("The task robot is not a Robot instance.")
        return self.task.robot.get_actuators_action()
    
    def apply_action(self, action_cmd) -> None: # target_qpos
        applied_action=self.task.robot.apply_action(action_cmd) # target_qpos[:self.task.robot.dof]
        self.last_action = applied_action
    
    # def apply_action_command(self, action_command):
    #     for act in action_command:
    #         self.actuators[act[0]].ctrl = act[1]
    #         self.last_action[act[0]] = act[1]

    def set_robot_qpos(self, qpos):
        joint_names = list(self.joints.keys())

        if len(qpos) > len(joint_names):
            qpos = qpos[:len(joint_names)]
        
        for joint_name, q in zip(joint_names, qpos):
            self.joints[joint_name].qpos = q
            self.joints[joint_name].qvel = 0
            self.joints[joint_name].qacc = 0

    def set_object_poses(self, obj_names, obj_positions, obj_orientations):
        for _, (name, p, q) in enumerate(zip(obj_names, obj_positions, obj_orientations)):
            self.mjData.joint(f"{name}_joint").qpos = np.concatenate([p, q], axis=0)
    
    def _robot_mask_geom_ids(self) -> set[int]:
        from simple.core.actor import ObjectActor
        from simple.assets.primitive import Primitive

        excluded_geom_names = {"ground"}
        excluded_body_names = {"world"}

        for actor_name, actor in self.task.layout.actors.items():
            if isinstance(actor, ObjectActor):
                excluded_body_names.add(actor.asset.uid)
            elif isinstance(actor, Primitive):
                excluded_body_names.add(actor_name)
                excluded_geom_names.add(f"{actor_name}_geom")

        robot_geom_ids: set[int] = set()
        for geom_id in range(self.mj_physics_model.ngeom):
            geom_name = mujoco.mj_id2name(self.mj_physics_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name in excluded_geom_names:
                continue
            body_id = int(self.mj_physics_model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(self.mj_physics_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name in excluded_body_names:
                continue
            robot_geom_ids.add(int(geom_id))
        return robot_geom_ids

    def _render_robot_mask(self, renderer, camera_name: str, color: np.ndarray, robot_geom_ids: set[int]) -> np.ndarray:
        original_rgba = self.mj_physics_model.geom_rgba[sorted(robot_geom_ids)].copy()
        try:
            self.mj_physics_model.geom_rgba[sorted(robot_geom_ids), 3] = 0.0
            renderer.update_scene(
                self.mj_physics_data,
                scene_option=self.render_option,
                camera=camera_name,
            )
            background = renderer.render()[..., :3].astype(np.uint8)
        finally:
            self.mj_physics_model.geom_rgba[sorted(robot_geom_ids)] = original_rgba
            renderer.update_scene(
                self.mj_physics_data,
                scene_option=self.render_option,
                camera=camera_name,
            )

        color_i16 = color.astype(np.int16)
        background_i16 = background.astype(np.int16)
        return np.any(np.abs(color_i16 - background_i16) > 2, axis=-1)

    def render(self, render_robot_mask: bool | str = False) -> Dict[str, np.ndarray]:
        image_observations = {}
        mask_camera_name = None
        if isinstance(render_robot_mask, str):
            mask_camera_name = render_robot_mask
        elif render_robot_mask:
            mask_camera_name = "front_stereo_left"

        robot_geom_ids = self._robot_mask_geom_ids() if mask_camera_name is not None else set()
        for mjCamera in self.mj_worldbody.find_all('camera'):
            renderer = self.renderers[mjCamera.name]
            if renderer is None:
                continue
            
            # with self._telemetry.timer(f"render.updatescene.{mjCamera.name}"):
            renderer.update_scene(
                self.mjData, 
                scene_option=self.render_option, 
                camera=mjCamera.name
            )
            # with self._telemetry.timer(f"render.render.{mjCamera.name}"):
            render_product = renderer.render()
            color = render_product[..., :3].astype(np.uint8) if render_product.dtype != np.uint8 else render_product[..., :3]
            image_observations[mjCamera.name] = color

            # with self._telemetry.timer(f"render.mask.{mjCamera.name}"):
            if render_robot_mask and mjCamera.name == "front_stereo_left": # FIXME:
                renderer.enable_segmentation_rendering()
                try:
                    out = renderer.render()[...,0]
                    panda_geom_ids = []
                    for geom_id in np.unique(out):
                        if "panda" in self.mjModel.id2name(geom_id, 'geom'): # FIXME
                            panda_geom_ids.append(geom_id)

                    panda_mask = np.zeros_like(out, dtype=bool)
                    for pgid in panda_geom_ids:
                        panda_mask = np.logical_or(panda_mask, out == pgid)
                except:
                    robot_mask = np.ones_like(color[...,0], dtype=bool) # not sure why render seg fails
                image_observations["robot_mask"] = robot_mask
                
        return image_observations
    
    def close(self):
        # print("Closing Mujoco simulator...")
        if hasattr(self, "renderers") and self.renderers:
            for renderer in self.renderers.values():
                renderer.close()
            self.renderers = {}
