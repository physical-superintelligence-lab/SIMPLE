"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

# from typing import TYPE_CHECKING

# if TYPE_CHECKING:
    # from simple.core import Randomizer
    # from simple.core import Layout
from copy import deepcopy
from typing import Protocol, Any, runtime_checkable
from abc import abstractmethod

from simple.core.randomizer import Randomizer, RandomizerCfg
from simple.core.layout import Layout
from simple.dr import TabletopSceneDR
from simple.core.actor import Light
from simple.core.types import Pose
from simple.assets.asset_manager import AssetManager
from simple.core.actor import ObjectActor

from simple.robots.protocols import WristCamMountable, HeadCamMountable
from simple.sensors.config import CameraCfg

class DRManager:
    
    # dr_level: int
    randomizers: dict[str, Randomizer]

    def __init__(self, level,  **kwargs) -> None:
        self._dr_level = level

        for key, rnd in kwargs.items():
            if isinstance(rnd, RandomizerCfg):
                self.randomizers[key] = rnd.build()
            else:
                raise TypeError(f"Expected Randomizer instance for {key}, got {type(rnd)}")
            
        self.set_level(level)
        
    @property
    def level(self):
        return self._dr_level

    @level.setter
    def level(self, value):
        if value != self._dr_level:
            self.set_level(value)
        self._dr_level = value

    
    def set_level(self, dr_level: int) -> None: 
        """Adjust domain randomization configurations based on DR level
        """
        self._dr_level = dr_level

    def get_randomizer(self, name: str) -> Randomizer | None:
        return self.randomizers.get(name, None)
    
    def state_dict(self) -> dict[str, Any]:
        inner_state_dict = {}
        for name, rnd in self.randomizers.items():
            inner_state_dict[name] = rnd.state_dict()
        return inner_state_dict
    
    def load_state_dict(self, state_dict: dict[str, Any],dr_level: int | None = None) -> None:
        """
          Depends on DR level to load state dict.If dr_level is None, load all state dict.
          If dr_level is 0, change distrcator,and table material.
          if dr_level is 1, change lighting,material,disctractors,
          if dr_level is 2, change spatial pose too.
        """
        if dr_level is None:
            dr_state_dict = state_dict["dr_state_dict"].copy()
        elif dr_level == 0:
            dr_state_dict = state_dict["dr_state_dict"].copy()
            dr_state_dict.pop("distractors", None)
            dr_state_dict.pop("material", None)
        elif dr_level == 1:
            dr_state_dict = state_dict["dr_state_dict"].copy()
            dr_state_dict.pop("distractors", None)
            dr_state_dict.pop("material", None)
            dr_state_dict.pop("lighting", None)
        elif dr_level == 2:
            dr_state_dict = state_dict["dr_state_dict"].copy()
            dr_state_dict.pop("distractors", None)
            dr_state_dict.pop("material", None)
            # dr_state_dict.pop("spatial", None)
            dr_state_dict.pop("lighting", None)
            
            if "g1_sonic" in dr_state_dict["spatial"]:
                dr_state_dict["spatial"] = {"g1_sonic": dr_state_dict["spatial"]["g1_sonic"]}
            elif "g1_wholebody" in dr_state_dict["spatial"]:
                dr_state_dict["spatial"] = {"g1_wholebody": dr_state_dict["spatial"]["g1_wholebody"]}
            else:
                dr_state_dict["spatial"] = None



        else:
            raise ValueError(f"Invalid DR level {dr_level}")

        for name, rnd_state in dr_state_dict.items():
            rnd =self.get_randomizer(name)
            assert rnd is not None, (
                "Cannot load DRManager state dict. "
                f"Because Randomizer {name} not found in DRManager."
                " Available randomizers: {list(self.randomizers.keys())}"
            )
            rnd.load_state_dict(rnd_state)

        # Reset randomizers that are not in state_dict
        for name, rnd in self.randomizers.items():
            if name not in dr_state_dict.keys():
                rnd._inner_state = None

    def reset(self, seed: int | None = None) -> None:
        for rnd in self.randomizers.values():
            rnd._inner_state = None

class TabletopGraspDRManager(DRManager):

    randomizers: dict[str, Randomizer] = {}

    def __init__(self, level: int, **kwargs) -> None:
        super().__init__(level, **kwargs)

        # self.dr_level = level
            
    def set_level(self, dr_level: int) -> None:
        """Adjust domain randomization configurations based on DR level"""
        if dr_level == 0:
            # keep original settings
            pass
        elif dr_level > 0:
            # dr_level=1
            lighting_dr = deepcopy(self.randomizers.get("lighting"))
            lighting_dr.cfg.light_mode = "fixed"
            material_dr = deepcopy(self.randomizers.get("material"))
            material_dr.cfg.material_mode = "fixed"
            scene_dr = deepcopy(self.randomizers.get("scene"))
            scene_dr.cfg.scene_mode = "fixed"
            scene_dr.cfg.room_choices = ["scene0"] # a specific scene "scene0, scene1, ..., scene5"

            self.randomizers["lighting"] = lighting_dr
            self.randomizers["material"] = material_dr
            self.randomizers["scene"] = scene_dr
            
            # dr_level=2
            if dr_level > 1: 
                distractors_dr = deepcopy(self.randomizers.get("distractors"))
                distractors_dr.cfg.number_of_distractors = 0
                self.randomizers["distractors"] = distractors_dr

                # dr_level=3
                if dr_level > 2:
                    spatial_dr = deepcopy(self.randomizers.get("spatial"))
                    spatial_dr.cfg.spatial_mode = "fixed"
                    spatial_dr.cfg.fixed_stable_pose_idx = 0 # default to first stable pose
                    self.randomizers["spatial"] = spatial_dr

    

    def random_layout(self, robot, sensor_cfgs, split) -> Layout:
        layout = Layout()

        layout.add_robot(robot)
        scene_info = {}

        target_dr = self.randomizers.get("target")
        if target_dr is not None:
            target_asset = target_dr(split)
            layout.add_object("target", target_asset)

        # Add distractors
        distractors_dr = self.randomizers.get("distractors")
        if distractors_dr is not None:
            distractors = distractors_dr(split)
            # print(f"Adding {len(distractors)} distractors to layout.")
            for obj_id, obj in distractors.items():
                layout.add_object(f"distractor_{obj_id}", obj)

        spatial_randomizer = self.randomizers.get("spatial")
        if spatial_randomizer is not None:
            # raise ValueError("Spatial randomizer not defined in DRManager.")
            spatial_randomizer.apply(split, layout)

        # add lights
        light_dr = self.randomizers.get("lighting")
        light_info = {}
        if light_dr is not None:
            for light in light_dr(split):
                layout.add_light(light)
        light_example = layout.lights[0] if len(layout.lights) > 0 else None
        light_info={
            "light_num":light_dr.cfg.light_num if light_dr is not None else 0,
            "light_color_temperature":light_example.light_color_temperature if light_example is not None else 6500,
            "light_intensity":light_example.light_intensity if light_example is not None else 100000,
            "light_radius":light_example.light_radius if light_example is not None else 0.1,
            "light_length":light_example.light_length if light_example is not None else 1.0,
            "light_spacing":light_example.pose.position,
            "light_position":light_example.center_light_postion if light_example is not None else [0.0,0.0,3.0],
            "light_orientation":light_example.center_light_orientation if light_example is not None else [0.0,0.0,0.0,1.0],

        }
        scene_info["light_info"] = light_info

        # add cameras
        camera_dr = self.randomizers.get("camera", None)
        for cam_id, cam_info in sensor_cfgs.items():
            if isinstance(cam_info, CameraCfg):
                cam_cfg = camera_dr(split, cam_info) if camera_dr else cam_info
                if (cam_cfg.quaternion is None and 
                    cam_cfg.mount == "eye_in_hand" and 
                    isinstance(robot, WristCamMountable)):
                    cam_cfg.pose["quaternion"] = robot.wrist_camera_orientation
                if (cam_cfg.quaternion is None and 
                    cam_cfg.mount == "eye_in_head" and 
                    isinstance(robot, HeadCamMountable)):
                    cam_cfg.pose["quaternion"] = robot.head_camera_orientation

                layout.add_camera(cam_id, cam_cfg)

        # add table
        scene_dr = self.randomizers.get("scene")
        if isinstance(scene_dr, TabletopSceneDR):
            scene = scene_dr(split)
            layout.scene = scene
            layout.add_primitive("table", scene.table)
            table_info = {
                "table_size": scene.table.size,
                "table_position": scene.table.pose.position,
                "table_orientation": scene.table.pose.quaternion,
            }
            scene_info["table_info"] = table_info

        # apply material randomization
        material = self.randomizers.get("material")
        if material is not None:
            material_info = material.apply(split, layout)
            scene_info["material_info"] = material_info
        layout.add_scene_info(scene_info)
        # layout.add_actor("robot", self.dr.robot, position["robot"])
        return layout
    
    def set_simple_layout(self, robot, sensor_cfgs, options: dict[str, any], split) -> Layout:
        object_info = options["object_info"]
        target_info = options["target_info"]

        layout = Layout()
        #add robot
        layout.add_robot(robot)
        #get scene info
        scene_info = {}

        for obj_info in object_info:
            obj_name = obj_info["name"]
            obj_id = obj_info["id"]
            obj_position = obj_info["position"]
            obj_orientation = obj_info["orientation"]
            is_target = obj_info["bTarget"]
            
            print(f"Adding object: {obj_name} (ID: {obj_id}) - Target: {is_target}")
            
            res_id = "graspnet1b"
            obj_asset = AssetManager.get(res_id).load(obj_id)
            layout_key = "target" if is_target else obj_name
            
            layout.add_object(layout_key, obj_asset)
            
            if layout_key in layout.actors:
                obj_actor = layout.actors[layout_key]
                obj_actor.pose.position = obj_position
                obj_actor.pose.quaternion = obj_orientation
                print(f"Set pose for {layout_key}: pos={obj_position}, quat={obj_orientation}")
            else:
                print(f"Warning: Object {layout_key} not found in layout actors after adding")

        # add cameras
        # camera_dr = self.randomizers.get("camera", None)
        # for cam_id, cam_info in sensor_cfgs.items():
        #     if isinstance(cam_info, CameraCfg):
        #         cam_cfg = camera_dr(split, cam_info) if camera_dr else cam_info
        #         layout.add_camera(cam_id, cam_cfg)

        camera_dr = self.randomizers.get("camera", None)
        for cam_id, cam_info in sensor_cfgs.items():
            if isinstance(cam_info, CameraCfg):
                cam_cfg = camera_dr(split, cam_info) if camera_dr else cam_info
                if (cam_cfg.quaternion is None and 
                    cam_cfg.mount == "eye_in_hand" and 
                    isinstance(robot, WristCamMountable)):
                    cam_cfg.pose["quaternion"] = robot.wrist_camera_orientation
                layout.add_camera(cam_id, cam_cfg)
                if (cam_cfg.quaternion is None and 
                    cam_cfg.mount == "eye_in_head" and 
                    isinstance(robot, HeadCamMountable)):
                    cam_cfg.pose["quaternion"] = robot.head_camera_orientation

        # add lights
        light_dr = self.randomizers.get("lighting")
        if light_dr is not None:
            for light in light_dr(split):
                layout.add_light(light)
        light_example = layout.lights[0] if len(layout.lights) > 0 else None
        light_info={
            "light_num":light_dr.cfg.light_num if light_dr is not None else 0,
            "light_color_temperature":light_example.light_color_temperature if light_example is not None else 6500,
            "light_intensity":light_example.light_intensity if light_example is not None else 100000,
            "light_radius":light_example.light_radius if light_example is not None else 0.1,
            "light_length":light_example.light_length if light_example is not None else 1.0,
            "light_spacing":light_example.pose.position,
            "light_position":light_example.center_light_postion if light_example is not None else [0.0,0.0,3.0],
            "light_orientation":light_example.center_light_orientation if light_example is not None else [0.0,0.0,0.0,1.0],

        }
        scene_info["light_info"] = light_info

         # add table
        scene = self.randomizers.get("scene")
        if isinstance(scene, TabletopSceneDR):
            scene = scene.apply(split)
            layout.scene = scene
            layout.add_primitive("table", scene.table)
            table_info = {
                "table_size": scene.table.size,
                "table_position": scene.table.pose.position,
                "table_orientation": scene.table.pose.quaternion,
              
            }
            scene_info["table_info"] = table_info

        # apply material randomization
        material = self.randomizers.get("material")
        if material is not None:
            material_info = material.apply(split, layout)
            scene_info["material_info"] = material_info
        layout.add_scene_info(scene_info)
        # layout.add_actor("robot", self.dr.robot, position["robot"])
        return layout
        


    def replicate_env(self, robot, sensor_cfgs, environment_config, split) -> Layout:

        object_info = environment_config["object_info"]
        scene_info = environment_config["scene_info"]
        material_info = scene_info["material_info"]
        # camera_info = scene_info["camera_info"]
        light_info = scene_info["light_info"]


        
        table_info = scene_info.get("table_info", None)

        if table_info is None:
            table_size = scene_info["robot_info"]["table_scale"]
            table_position = scene_info["robot_info"]["table_position"]
            table_orientation = scene_info["robot_info"]["robot_orientation"] # FIXME: TO BE CONSISTENT !!!
        else:
            table_size = table_info["table_size"] 
            table_position = table_info["table_position"]
            table_orientation = table_info["table_orientation"] # FIXME: TO BE CONSISTENT !!!

        layout = Layout()
        
        # add robot
        layout.add_robot(robot)
        # change robot pose by info
        # robot_actor = layout.actors["robot"]
        # robot_actor.pose.position = robot_info["robot_position"]
        # robot_actor.pose.quaternion = robot_info["robot_orientation"]
        
        # add objects
        for obj_info in object_info:
            obj_name = obj_info["name"]
            obj_id = obj_info["id"]
            obj_position = obj_info["position"]
            obj_orientation = obj_info["orientation"]
            is_target = obj_info["bTarget"]
            
            print(f"Adding object: {obj_name} (ID: {obj_id}) - Target: {is_target}")
            
            res_id = "graspnet1b"
            obj_asset = AssetManager.get(res_id).load(obj_id)
            
            layout_key = "target" if is_target else obj_name
            
            layout.add_object(layout_key, obj_asset)
            
            if layout_key in layout.actors:
                obj_actor = layout.actors[layout_key]
                obj_actor.pose.position = obj_position
                obj_actor.pose.quaternion = obj_orientation
                print(f"Set pose for {layout_key}: pos={obj_position}, quat={obj_orientation}")
            else:
                print(f"Warning: Object {layout_key} not found in layout actors after adding")
        
        # add cameras
        camera_dr = self.randomizers.get("camera", None)
        for cam_id, cam_info in sensor_cfgs.items():
            if isinstance(cam_info, CameraCfg):
                cam_cfg = camera_dr(split, cam_info) if camera_dr else cam_info
                if (cam_cfg.quaternion is None and 
                    cam_cfg.mount == "eye_in_hand" and 
                    isinstance(robot, WristCamMountable)):
                    cam_cfg.pose["quaternion"] = robot.wrist_camera_orientation
                if (cam_cfg.quaternion is None and 
                    cam_cfg.mount == "eye_in_head" and 
                    isinstance(robot, HeadCamMountable)):
                    cam_cfg.pose["quaternion"] = robot.head_camera_orientation
                layout.add_camera(cam_id, cam_cfg)
                
        # add table
        scene = self.randomizers.get("scene")
        if isinstance(scene, TabletopSceneDR):
            scene = scene.replicate_scene(split, table_size, table_position, table_orientation)
            layout.scene = scene
            layout.add_primitive("table", scene.table)

            
        # add lights
        light_num = light_info["light_num"]
        light_spacing = light_info["light_spacing"]  
        if isinstance(light_num, list) and len(light_num) == 2:
            for i in range(light_num[0]):
                for j in range(light_num[1]):
                    light = Light(
                        uid = f"Light_{i}_{j}",
                        type = "CylinderLight", 
                    )
                    light.pose = Pose(position=[
                        light_spacing[1] * (j - 0.5 * (light_num[1]-1)),
                        light_spacing[0] * (i - 0.5 * (light_num[0]-1)),
                        0.0
                    ])
                    # light.pose = Pose(position=light_spacing)
                    light.light_radius = light_info["light_radius"]
                    light.light_length = light_info["light_length"]
                    light.light_intensity = light_info["light_intensity"]
                    light.light_color_temperature= light_info["light_color_temperature"]
                    light.center_light_postion = light_info["light_position"]
                    light.center_light_orientation = light_info["light_orientation"]
                    layout.add_light(light)
        
        # add material
        object_shader_params = material_info["object_shader_params"] # FIXME
        # material_mode = "fixed"
        # import numpy as np
        for key, obj in layout.actors.items():
            if isinstance(obj, ObjectActor):
                # material = {
                #     'reflection_roughness_constant': np.random.uniform(0.0, 1.0 if material_mode != "fixed" else 0.5),
                #     'metallic_constant': np.random.uniform(0.0, 1.0) if material_mode != "fixed" else 0.,
                #     'specular_level': np.random.uniform(0.0, 1.0) if material_mode != "fixed" else 0.,
                # }
                material = object_shader_params.pop()
                obj.set_material(material)

        table = layout.actors.get("table", None)
        if table is not None:
            table.set_material(material_info["table_material"])
            
        robot = layout.actors.get("robot", None)
        if robot is not None:
            robot.set_shaders(material_info["robot_shader_params"])
        
        return layout
    
    def _get_object_info(self, obj: ObjectActor,is_target) -> dict:
        return {
            "name": obj.asset.label,
            "id": int(obj.asset.uid),
            "position": obj.pose.position,
            "orientation": obj.pose.quaternion,
            "bTarget": is_target
        }
    
    def _get_target_stable_poses_and_grasps(self, obj: ObjectActor):
        results = dict()
        stable_poses=obj.asset.stable_poses
        grasps=obj.asset.canonical_grasps
        pose_and_grasp = []
        for stable_idx in range(len(stable_poses)):
            grasp=grasps[stable_idx]
            pose_and_grasp.append((stable_poses[stable_idx],grasp))
        id=int(obj.asset.uid)
        results[id] = pose_and_grasp
        return pose_and_grasp

    def random_place_objects(self,  split: str) -> None:
        #add objects to layout and random place them
        layout = Layout()
        target_dr = self.randomizers.get("target")
        if target_dr is not None:
            target_asset = target_dr(split)
            layout.add_object("target", target_asset)

        # Add distractors
        distractors_dr = self.randomizers.get("distractors")
        if distractors_dr is not None:
            distractors = distractors_dr()
            # print(f"Adding {len(distractors)} distractors to layout.")
            for obj_id, obj in distractors.items():
                layout.add_object(f"distractor_{obj_id}", obj)

        spatial_randomizer = self.randomizers.get("spatial")
        if spatial_randomizer is not None:
            # raise ValueError("Spatial randomizer not defined in DRManager.")
            spatial_randomizer(split, layout)
        
        # get object info and target info
        target_info=None
        objects_info = []
        for key, obj in layout.actors.items():
            if isinstance(obj, ObjectActor):
                is_target = (key == "target")
                if is_target:
                    target_info = self._get_object_info(obj, is_target)
                    stable_poses_and_grasps = self._get_target_stable_poses_and_grasps(obj)

                obj_info = self._get_object_info(obj, is_target)
                objects_info.append(obj_info)
        
        #get target stable pose idx,in order to get grasps for this stable pose
        import numpy as np
        import transforms3d
        target_pos=layout.actors["target"].pose.position
        target_quat=layout.actors["target"].pose.quaternion
        target_stable_pose = np.array([*target_pos, *target_quat])
        target_stable_poses=layout.actors["target"].asset.stable_poses
        idx = np.where(np.isclose(target_stable_poses[:, 2], target_stable_pose[2], rtol=0, atol=1e-8))[0][0]
        stable_pose_and_grasp = stable_poses_and_grasps[idx]

        # grasp info
        grasp_infos=stable_pose_and_grasp[1]

        grasp_depths = np.array([grasp_info[1] for grasp_info in grasp_infos], dtype=np.float32) 
        T_object_grasp = np.stack([grasp_info[0] for grasp_info in grasp_infos], axis=0)
        T_world_object = np.eye(4)
        rand_target_ori_mat = transforms3d.quaternions.quat2mat(target_info['orientation'])
        T_world_object[:3, :3] = rand_target_ori_mat
        T_world_object[:3, 3] = target_info['position']
        T_world_grasp = T_world_object @ T_object_grasp

        # the x-axis unit vector should roughly point to the ground
        x_rotated = T_world_grasp[:, :3, 0]
        # find the angle between x_rotated and the -z axis
        angles = np.arccos(np.dot(x_rotated, np.array([0, 0, -1])))
        valid_grasp_idxs = np.where(angles < 20 / 180 * np.pi)[0]
        if len(valid_grasp_idxs) == 0:
            # hack when no ideal valid poses are found
            valid_grasp_idxs = np.arange(len(T_world_grasp))

        target_id=target_info["id"]
        target_pose = {
            "target_info": target_info,
            "grasp_poses": [{
                                'position': T_world_grasp[idx, :3, 3], # FIXME 
                                'orientation': transforms3d.quaternions.mat2quat(T_world_grasp[idx, :3, :3]),
                                'width': grasp_infos[idx][1],
                                'depth': grasp_depths[idx], #0.005,
                                'score': 1,
                                'object_id': target_id,
                                'grasp_id': idx,
                            } for idx in valid_grasp_idxs]
        }
        # print(f"Target ID: {target_id}, Stable Pose Index: {idx}, Total Grasps: {len(grasp_infos)}, Valid Grasps: {len(valid_grasp_idxs)}")

        return target_pose,objects_info
    
    def repeat_random_place_objects(self,split:str, num_repeats:int) :
        all_objects_info = []
        all_target_pose = []
        for _ in range(num_repeats):
            self.reset()
            target_pose, objects_info = self.random_place_objects(split)
            if target_pose is None:
                # let's ignore the little chance that the first layout if None
                all_target_pose.append(all_target_pose[-1])
                all_objects_info.append(all_objects_info[-1])
            else:
                all_target_pose.append(target_pose)
                all_objects_info.append(objects_info)
        return all_target_pose, all_objects_info



    



# class LayoutManager:

#     def __init__(self, DRManager: DRManager, split:str) -> None:
#         self.dr = DRManager
#         self.split = split
        
#     def random_layout(self) -> Layout:
#         spatial_randomizer = self.dr.randomizers.get("spatial")
#         if spatial_randomizer is None:
#             raise ValueError("Spatial randomizer not defined in DRManager.")
        
#         target_object = self.dr.randomizers.get("target")

#         position = spatial_randomizer.apply(self.split)

#         layout = Layout()
#         # layout.add_actor("robot", self.dr.robot, position["robot"])


#     def get_robot_position(self):
#         ...