import numpy as np
import transforms3d as t3d
from typing import Optional
from simple.core.asset import Asset
from simple.core.object import SpatialAnnotated, SemanticAnnotated
from simple.core.types import Pose
import torch
import os
from simple.robots.protocols import HasDexterousHand
from simple.utils import resolve_data_path

class AnnoatedAsset(Asset, SpatialAnnotated, SemanticAnnotated):
    ...

class Bodex:
    stable_idx = None
    def __init__(self):
        pass


    def detect_grasps(self, obj):
        pass
    @classmethod
    def reset(cls):
        cls.stable_idx = None
    @staticmethod
    def create_hand_curobo_model(config_path:str):
        from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
        from curobo.types.base import TensorDeviceType
        from curobo.types.robot import RobotConfig
        from curobo.util.logger import setup_curobo_logger
        from curobo.util_file import get_robot_path, join_path, load_yaml
        from simple.utils import resolve_data_path
        import os

        tensor_args = TensorDeviceType()
        config_file=load_yaml(resolve_data_path(config_path))["robot_cfg"]
        config_file["kinematics"]["collision_spheres"] = os.path.join(os.getcwd(), "data",os.path.dirname(config_path), config_file["kinematics"]["collision_spheres"])
        
        if ("external_asset_path" in config_file["kinematics"] and config_file["kinematics"]["external_asset_path"] is not None):
                config_file["kinematics"]["urdf_path"] = join_path(config_file["kinematics"]["external_asset_path"], config_file["kinematics"]["urdf_path"])

        config_file["kinematics"]["urdf_path"] = os.path.join(os.getcwd(),"data", config_file["kinematics"]["urdf_path"].lstrip("/"))
        robot_config = RobotConfig.from_dict(config_file,tensor_args)

        hand_model = CudaRobotModel(robot_config.kinematics)
        return hand_model


    @staticmethod
    def transform_linkpose_to_world(link_pose , T_bodexworld_object: np.ndarray, T_world_object: np.ndarray):
        position = link_pose.position.cpu().numpy().squeeze(0)
        quaternion = link_pose.quaternion.cpu().numpy().squeeze(0)
        T_bodexworld_linkpose = np.eye(4)
        T_bodexworld_linkpose[:3, 3] = position
        T_bodexworld_linkpose[:3, :3] = t3d.quaternions.quat2mat(quaternion)

        T_object_linkpose =np.linalg.inv(T_bodexworld_object) @ T_bodexworld_linkpose

      
        T_world_linkpose = T_world_object @ T_object_linkpose

        new_position = T_world_linkpose[:3,3]

        new_quaternion = t3d.quaternions.mat2quat(T_world_linkpose[:3,:3])

        link_pose.position = torch.from_numpy(new_position).float().to(link_pose.position.device).unsqueeze(0)

        link_pose.quaternion = torch.from_numpy(new_quaternion).float().to(link_pose.position.device).unsqueeze(0)

        return link_pose


        
    
    @classmethod
    def load_cached_grasps_debug(
        cls, 
        robot: HasDexterousHand,
        asset: AnnoatedAsset, 
        stable_idx: Optional[str] = None, 
        target_pose: Optional[Pose] = None,
        max_grasps: Optional[int] = None,
        hand_uid: Optional[str] = None
    ) -> list:
        
        """  
        Load cached grasps for the given asset. transform grasps to world frame using target_pose.
        Args:
            asset (SpatialAnnotated): The asset to load grasps for.
            stable_idx (Optional[str], optional): The stable pose index. Defaults to None.
            target_pose (Optional[Pose], optional): The target pose of the asset in world frame
            max_grasps (Optional[int], optional): Maximum number of grasps to load. Defaults to None.
        Returns:
            list: A list of grasp dictionaries (in world frame) containing position, orientation, width, depth, score,
        """
        hand_model = cls.create_hand_curobo_model(
            resolve_data_path(robot.hand_yaml, auto_download=True)
        )
        hand_dof = robot.hand_dof
        object_stable_poses = asset.stable_poses[0] # np.load("data/assets/graspnet/stable/12_stable.npy", allow_pickle=True)[0]
        T_bodexworld_object = np.eye(4)
        T_bodexworld_object[:3, 3] = [0.4000000059604645, 0.0, object_stable_poses[2]]
        T_bodexworld_object[:3,:3] = t3d.quaternions.quat2mat([object_stable_poses[2]])

        bodex_asset_path = resolve_data_path(f"assets/graspnet/dex_grasp/{hand_uid}/12/12__1_pregrasp.pt", auto_download=True)
        bodex_asset_pregrasp_path = resolve_data_path(f"assets/graspnet/dex_grasp/{hand_uid}/12/12__1.pt", auto_download=True)

        dex_pregrasp_poses=torch.load(bodex_asset_path)
        dex_grasp_poses= torch.load(bodex_asset_pregrasp_path)
        
      
        kin_state= hand_model.get_state(dex_pregrasp_poses[:])
        link_poses = kin_state.link_poses

        dex_pregrasp_poses = dex_pregrasp_poses.cpu().numpy()
        # dex_grasp_poses = dex_grasp_poses.cpu().numpy()


        dex_grasp = np.load(f"data/assets/graspnet/dex_grasp/{hand_uid}/12/12__1.npy", allow_pickle=True).item()

        # get grasp pose in world frame
        T_bodexworld_grasp = np.eye(4)
        T_bodexworld_grasp[:3, 3] = dex_pregrasp_poses[:3]
        T_bodexworld_grasp[:3, :3] = t3d.quaternions.quat2mat(dex_pregrasp_poses[3:7])

        T_object_grasp = np.linalg.inv(T_bodexworld_object) @ T_bodexworld_grasp

        # FIXME rotation the success rate is very low , need to be fixed
        T_world_object = target_pose.as_matrix()
        T_world_object[:3, :3] = T_bodexworld_object[:3, :3]
        T_world_grasp = T_world_object @ T_object_grasp

        # get T_world_link_poses
        for key,pose in link_poses.items():
            link_poses[key] = cls.transform_linkpose_to_world(pose, T_bodexworld_object, T_world_object)


        position = T_world_grasp[:3,3]
        
        orientation = t3d.quaternions.mat2quat(T_world_grasp[:3,:3])

        #get grasp qpos,squeeze qpos,lift qpos
        grasp_qpos=dex_grasp["robot_pose"][:,-3,:]
        squeeze_qpos=dex_grasp["robot_pose"][:,-2,:]
        lift_qpos=dex_grasp["robot_pose"][:,-1,:]
        
        if "dex3" in robot.hand_yaml:
            grasp_qpos= np.concatenate([grasp_qpos[... , :-hand_dof] , grasp_qpos[..., -hand_dof:][..., [3,4,5,6,0,1,2]]],axis=1)
            squeeze_qpos= np.concatenate([squeeze_qpos[... , :-hand_dof] , squeeze_qpos[..., -hand_dof:][..., [3,4,5,6,0,1,2]]] , axis=1)
            lift_qpos = np.concatenate( [lift_qpos[... , :-hand_dof] , lift_qpos[..., -hand_dof:][..., [3,4,5,6,0,1,2]]] , axis=1 )

        grasp_poses = [{
            'position': position, 
            'orientation': orientation,
            'width': 0.04,
            'depth': 0, #0.005,
            'score': 1,
            # 'object_id': target_id,
            'grasp_id': 12,
            # "qpos": dex_grasp["pregrasp_qpos"][7:14]
            "link_poses": link_poses,
            "all_qpos" : dex_grasp["robot_pose"][0],
            "grasp_qpos": grasp_qpos,
            "squeeze_qpos": squeeze_qpos,
            "lift_qpos": lift_qpos,
            ## TODO pass links poses instead
        } ]

        return grasp_poses

    @classmethod
    def load_cached_grasps(
        cls, 
        robot: HasDexterousHand,
        asset: AnnoatedAsset,
        stable_idx: Optional[str] = None, 
        target_pose: Optional[Pose] = None,
        max_grasps: Optional[int] = None,
        hand_uid: Optional[str] = None
    ) -> list:
        
        """  
        Load cached grasps for the given asset. transform grasps to world frame using target_pose.
        Args:
            asset (SpatialAnnotated): The asset to load grasps for.
            stable_idx (Optional[str], optional): The stable pose index. Defaults to None.
            target_pose (Optional[Pose], optional): The target pose of the asset in world frame
            max_grasps (Optional[int], optional): Maximum number of grasps to load. Defaults to None.
        Returns:
            list: A list of grasp dictionaries (in world frame) containing position, orientation, width, depth, score,
        """
        hand_model = cls.create_hand_curobo_model(
            resolve_data_path(robot.hand_yaml, auto_download=True)
        )
        hand_dof = robot.hand_dof

        uid = asset.uid
        grasp_folder = resolve_data_path(f"assets/graspnet/dex_grasp/{hand_uid}/{uid}/", auto_download=True)
        num_grasp_poses = int(sum(1 for f in os.listdir(grasp_folder) if f.endswith("pt"))/ 2)
        # get stable pose
        stable_poses = asset.stable_poses
        
        if cls.stable_idx is None:
            assert target_pose is not None, "Either stable_idx or stable_pose must be provided."
            for index in range(len(stable_poses)):
                pose = stable_poses[index]
                if np.allclose(
                    np.array(pose[2:3]), 
                    np.array(target_pose.position[2:3]), 
                    rtol=0, atol=1e-2
                ): # FIXME
                    cls.stable_idx = index # type: ignore
                    break

        else:
            assert target_pose is not None, "Either stable_idx or stable_pose must be provided."
                
        # assert cls.stable_idx is not None, "No matching stable pose found."
        
        # Fallback: if no matching pose found, use the first stable pose
        if cls.stable_idx is None:
            print(f"[Bodex.load_cached_grasps] Warning: No exact stable pose match found. Using first stable pose (index 0).")
            cls.stable_idx = 0

        stable_position = stable_poses[cls.stable_idx][:3]
        stable_quaternion = stable_poses[cls.stable_idx][3:]

        # # What stable_idx is being selected?
        # print(f"[DEBUG] Object pose quaternion: {target_pose.quaternion}")
        # print(f"[DEBUG] Selected stable_idx: {cls.stable_idx}")  # or however it's determined
        # print(f"[DEBUG] Expected stable pose quat: {asset.stable_poses[cls.stable_idx][3:]}")
        # print(f"[DEBUG] Stable position: {stable_position}, Stable quaternion: {stable_quaternion}")

        # FIXME should directly load object from object frame
        T_bodexworld_object = np.eye(4)
        T_bodexworld_object[:3, 3] = [0.4, 0.0, stable_position[2]]
        T_bodexworld_object[:3,:3] = t3d.quaternions.quat2mat(stable_quaternion)

        # get random grasp pose
        random_grasp_idx = np.random.randint(0, num_grasp_poses)
        dex_pregrasp_poses=torch.load(os.path.join(grasp_folder, f"{uid}__{random_grasp_idx}_pregrasp.pt"))
        dex_grasp_poses= torch.load(os.path.join(grasp_folder, f"{uid}__{random_grasp_idx}.pt"))

        #get kin state and link poses
        kin_state= hand_model.get_state(dex_pregrasp_poses[:])
        link_poses = kin_state.link_poses

        dex_pregrasp_poses = dex_pregrasp_poses.cpu().numpy()
        dex_grasp_poses = dex_grasp_poses.cpu().numpy()


        dex_grasp = np.load(os.path.join(grasp_folder, f"{uid}__{random_grasp_idx}.npy"), allow_pickle=True).item()

        # get grasp pose in world frame
        T_bodexworld_grasp = np.eye(4)
        T_bodexworld_grasp[:3, 3] = dex_pregrasp_poses[:3]
        T_bodexworld_grasp[:3, :3] = t3d.quaternions.quat2mat(dex_pregrasp_poses[3:7])

        T_object_grasp = np.linalg.inv(T_bodexworld_object) @ T_bodexworld_grasp

        # FIXME rotation the success rate is very low , need to be fixed
        T_world_object = target_pose.as_matrix()
        # T_world_object[:3, :3] = T_bodexworld_object[:3, :3]
        T_world_grasp = T_world_object @ T_object_grasp

        # get T_world_link_poses
        for key,pose in link_poses.items():
            link_poses[key] = cls.transform_linkpose_to_world(pose, T_bodexworld_object, T_world_object)

        position = T_world_grasp[:3,3]
        
        orientation = t3d.quaternions.mat2quat(T_world_grasp[:3,:3])

        #get grasp qpos,squeeze qpos,lift qpos
        grasp_qpos=dex_grasp["robot_pose"][:,-3,:]

        squeeze_qpos=dex_grasp["robot_pose"][:,-2,:]
        lift_qpos=dex_grasp["robot_pose"][:,-1,:]

        
        if "dex3" in robot.hand_yaml:
            grasp_qpos= np.concatenate([grasp_qpos[... , :-hand_dof] , grasp_qpos[..., -hand_dof:][..., [3,4,5,6,0,1,2]]],axis=1)
            squeeze_qpos= np.concatenate([squeeze_qpos[... , :-hand_dof] , squeeze_qpos[..., -hand_dof:][..., [3,4,5,6,0,1,2]]] , axis=1)
            lift_qpos = np.concatenate( [lift_qpos[... , :-hand_dof] , lift_qpos[..., -hand_dof:][..., [3,4,5,6,0,1,2]]] , axis=1 )
        
        else:
            grasp_qpos= np.concatenate([grasp_qpos[... , :-hand_dof] , grasp_qpos[..., -hand_dof:]],axis=1)
            squeeze_qpos= np.concatenate([squeeze_qpos[... , :-hand_dof] , squeeze_qpos[..., -hand_dof:]] , axis=1)
            lift_qpos = np.concatenate( [lift_qpos[... , :-hand_dof] , lift_qpos[..., -hand_dof:]] , axis=1 )


        
       

        # hard code for debug
        # position=np.array([-0.5053, -0.1370,  0.1830])
        # orientation=np.array([1, 0, 0, 0])

        grasp_poses = [{
            'position': position, 
            'orientation': orientation,
            'width': 0.04,
            'depth': 0, #0.005,
            'score': 1,
            # 'object_id': target_id,
            'grasp_id': uid,
            # "qpos": dex_grasp["pregrasp_qpos"][7:14]
            "link_poses": link_poses,
            "all_qpos" : dex_grasp["robot_pose"][0],
            "grasp_qpos": grasp_qpos,
            "squeeze_qpos": squeeze_qpos,
            "lift_qpos": lift_qpos,
            ## TODO pass links poses instead
        } ]

        return grasp_poses

        


        
