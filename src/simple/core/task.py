"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, TYPE_CHECKING
# from simple.core.task import SubtaskSpec

if TYPE_CHECKING:
    from simple.sensors.config import SensorCfg
    from simple.core.randomizer import Randomizer, RandomizerCfg
    from simple.core.actor import Actor
    from simple.dr.manager import DRManager
    from simple.datagen.subtask_spec import SubtaskSpec
    # from simple.core.layout import Layout
    # from simple.core.robot import Robot
# from gymnasium.envs.registration import EnvSpec


from simple.core.layout import Layout
from simple.core.robot import Robot
from simple.robots.protocols import WristCamMountable,HeadCamMountable
from simple.sensors.config import CameraCfg
from simple.dr.scene import TabletopSceneDR

import numpy as np
from gymnasium import spaces

from abc import ABC, abstractmethod



class Task(ABC):

    metadata: dict[str, Any] = {
        "physics_dt": 0.002,
        "render_hz": 30,
        "split": "train",  # train, val, test
    }

    uid: str
    label: str
    description: str

    robot_cfg: dict[str, Any]

    sensor_cfgs: dict[str, SensorCfg]

    dr_cfgs: dict[str, RandomizerCfg]

    def __init__(
        self, 
        dr: DRManager,
        split: str | None = None, 
        render_hz: int | None = None,
        dr_level: int | None = None,
        physics_dt: float | None = None,
        *args, 
        **kwargs
    ) -> None:
        self.metadata.update({
            k: v for k, v in {
                "split": split,
                "render_hz": render_hz,
                "dr_level": dr_level,
                "physics_dt": physics_dt,
            }.items() if v is not None
        })
        self.dr = dr 
        self.split=split

    @property
    def render_hz(self) -> int:
        return self.metadata.get("render_hz", 30)

    @property
    # @abstractmethod
    def robot(self) -> Robot: 
        # TODO change to RobotConfig
        return self._robot

    @robot.setter
    def robot(self, value: "Robot"):
        print(f"Setting robot: {value}")
        self._robot = value

    @property
    @abstractmethod
    def layout(self) -> Layout: ...

    @property
    @abstractmethod
    def instruction(self) -> str:
        """ A natural language instruction describing the task. """
        # return self.description

    @property
    def observation_space(self) -> spaces.Space:
        """ default observation space, can be overridden by subclasses """
        obs = {}
        for key, sensorCfg in self.sensor_cfgs.items():
            sense_obs = sensorCfg.observation_space
            if isinstance(sense_obs, spaces.Dict):
                for sub_key, sub_space in sense_obs.spaces.items():
                    obs[f"{key}_{sub_key}"] = sub_space
            else:
                obs[key] = sense_obs
        return spaces.Dict(obs)
    
    @property
    @abstractmethod
    def action_space(self) -> spaces.Space:
        ...

    def save_config(self) -> dict[str, Any]:
        info = {
            "uid": self.uid,
            "label": self.label,
            "description": self.description,
            "metadata": self.metadata,
            "robot_cfg": self.robot_cfg,
            "sensor_cfgs": {k: v.__dict__ for k, v in self.sensor_cfgs.items()},
            "dr_cfgs": {k: v.to_dict() for k, v in self.dr_cfgs.items()},
        }
        return info
    
    def load_config(self, path: str) -> None:
        """ Load task config from a json file.
        """
        raise NotImplementedError

    def reset(self, seed:int|None=None, options: Optional[dict[str, Any]] = None) -> None:
        """ Resets the task state. 
        
            Reset the task to RANDOM initial state.
            if options is given it will DUPLICATE the states specified by the options. 
        
            This happens before reseting the environment, and can be used to set up the task-specific state, 
            such as randomizing the target object and its position.
            The reset of the environment will call the task's DR to randomize the scene, 
            and then call the task's reset() to set up the task-specific state based on the randomized scene.
        """    
        split = self.metadata.get("split", "train")

        if options is not None and "state_dict" in options and options["state_dict"] is not None:
            assert self.uid == options["state_dict"]["uid"], f"load the wrong state dict for the {self.uid} task ?!"
            dr_level = options.get("dr_level", None)
            self.dr.load_state_dict(options["state_dict"],dr_level=dr_level)
        else:
            self.dr.reset(seed=seed)

        self._layout = Layout()
        self._layout.add_robot(self.robot)

        # scene randomization
        scene_dr = self.dr.get_randomizer("scene")
        if isinstance(scene_dr, TabletopSceneDR):
            scene = scene_dr(split)
            self._layout.scene = scene
            self._layout.add_primitive("table", scene.table)
            
            if hasattr(scene, 'table2') and scene.table2 is not None:
                self._layout.add_primitive("table2", scene.table2)
            # table_info = {
            #     "table_size": scene.table.size,
            #     "table_position": scene.table.pose.position,
            #     "table_orientation": scene.table.pose.quaternion,
            # }
            # scene_info["table_info"] = table_info
            table_height = scene.table.pose.position[2] + 0.5 * scene.table.size[2]
        else:
            table_height = 0.0
            
        # container randomization
        container_dr = self.dr.get_randomizer("container")
        if container_dr is not None:
            container_asset = container_dr(split)
            self._layout.add_object("container", container_asset)

        # target randomization
        target_dr = self.dr.get_randomizer("target")
        if target_dr is not None:
            target_asset = target_dr(split)
            self._layout.add_object("target", target_asset)

        # distractor randomization
        distractors_dr = self.dr.get_randomizer("distractors")
        if distractors_dr is not None:
            distractors = distractors_dr(split)
            # print(f"Adding {len(distractors)} distractors to layout.")
            for idx, (obj_id, obj) in enumerate(distractors.items()):
                self._layout.add_object(f"distractor_{idx}", obj)
        
        articulated_dr = self.dr.get_randomizer("articulated")
        if articulated_dr is not None:
            articulated_asset = articulated_dr(split)
            self._layout.add_articulated_object("articulated", articulated_asset)
            
        
        # sptial randomization
        spatial_dr = self.dr.get_randomizer("spatial")
        if spatial_dr is not None:
            spatial_dr(split, self._layout, table_height=table_height)

        # lighting randomization
        light_dr = self.dr.get_randomizer("lighting")
        if light_dr is not None:
            for light in light_dr(split):
                self._layout.add_light(light)
        # light_example = self._layout.lights[0] if len(self._layout.lights) > 0 else None
        # light_info = {
        #     "light_num": light_dr.cfg.light_num if light_dr is not None else 0,
        #     "light_color_temperature":light_example.light_color_temperature if light_example is not None else 6500,
        #     "light_intensity":light_example.light_intensity if light_example is not None else 100000,
        #     "light_radius":light_example.light_radius if light_example is not None else 0.1,
        #     "light_length":light_example.light_length if light_example is not None else 1.0,
        #     "light_spacing":light_example.pose.position,
        #     "light_position":light_example.center_light_postion if light_example is not None else [0.0,0.0,3.0],
        #     "light_orientation":light_example.center_light_orientation if light_example is not None else [0.0,0.0,0.0,1.0],
        # }
        # scene_info["light_info"] = light_info

        # camera randomization
        camera_dr = self.dr.get_randomizer("camera")
        if camera_dr is not None:
            for cam_id, cam_info in self.sensor_cfgs.items():
                if isinstance(cam_info, CameraCfg):
                    cam_cfg = camera_dr(split, cam_info)
                    if (cam_cfg.quaternion is None and 
                        cam_cfg.mount == "eye_in_hand"):
                        assert isinstance(self.robot, WristCamMountable), "Robot does not support wrist camera mounting."
                        cam_cfg.pose["quaternion"] = self.robot.wrist_camera_orientation
                    if (cam_cfg.quaternion is None and 
                        cam_cfg.mount == "eye_in_head" and 
                        isinstance(self.robot, HeadCamMountable)):
                        cam_cfg.pose["quaternion"] = self.robot.head_camera_orientation

                    self._layout.add_camera(cam_id, cam_cfg)
                

        # material randomization
        material_dr = self.dr.get_randomizer("material")
        if material_dr is not None:
            material_info = material_dr(split, self._layout)
            # scene_info["material_info"] = material_info

        # self._layout.scene_info = scene_info # FIXME
    
    def state_dict(self) -> Dict[str, Any]:
        """ Dump the current state of the task.

        Returns:
            A dictionary containing the current state of the task.
        """
        rand_state_dict = self.dr.state_dict()
        return {
            "uid": self.uid,
            "label": self.label,
            "description": self.description,
            "metadata": self.metadata,
            "robot_cfg": self.robot_cfg,
            "sensor_cfgs": {k: v.__dict__ for k, v in self.sensor_cfgs.items()},
            "dr_cfgs": {k: v.to_dict() for k, v in self.dr_cfgs.items()},
            "dr_state_dict": rand_state_dict,
            "layout": self._layout.to_dict() if self._layout is not None else {},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """ Clone the current layout with given options.
        """
        return self.reset(options={"state_dict": state_dict})
    
    def check_success(self, *args, **kwargs) -> bool: 
        """ Check if the task is successfully completed.
        """
        raise NotImplementedError
    
    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        """ Compute the reward for the current state of the task.
        """
        return 0.0
    
    def preload_objects(self) -> list[Actor]:
        """ Preloads all assets required by the task. 
        
        This is needed for recurrent reset of the environment in IsaacSim
        if each episode might need different assets.


        Returns:
            list[Actor] a list of Actor instances that need to be preloaded.
        """
        raise NotImplementedError
    
    def decompose(self) -> list[SubtaskSpec]:
        """ Decompose the task into a sequence of subtasks.

        Returns:
            list[SubtaskSpec]: A list of SubtaskSpec instances representing the subtasks.
        """
        raise NotImplementedError