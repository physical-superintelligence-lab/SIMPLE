"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""
import json

def get_episode_rlds(dataset, eps_idx, data_format):
    dataset = dataset.skip(eps_idx).take(1)
    episode = next(iter(dataset))
    # task_uuid = episode["uuid"].numpy().decode("utf-8")
    environment_config = json.loads(episode['environment_config'].numpy().decode("utf-8"))
    if data_format == "rlds-legacy":
        env_conf = convert_env_config_to_state_dict(environment_config)
    else:
        env_conf = environment_config
    return env_conf, episode

def convert_env_config_to_state_dict (env_config):
    state_dict={
        "uid": "franka_tabletop_grasp",
        "dr_state_dict":{}
    }
    # state_dict["state_dict"]["uid"] = "franka_tabletop_grasp"
    dr_state_dict = state_dict['dr_state_dict']
    # language instruction
    dr_state_dict["language"] = "Pick up {}."

    #for target
    target_info = dr_state_dict["target"]={}
    target_info['res_id'] = 'graspnet1b'
    target_info['uid'] = int(env_config['target_info']['id'])

    #for distractors
    distractors_info = dr_state_dict["distractors"]={}

    for obj_info in env_config["object_info"]:
        obj_id = obj_info["id"]
        res_id = "graspnet1b"

        is_target = obj_info["bTarget"]
        if not is_target:
            distractors_info[obj_id] = {
                "res_id": res_id,
                "uid": int(obj_id)
            }

    #for spatial
    spatials_info = dr_state_dict["spatial"]={}
    for obj_info in env_config["object_info"]:
        obj_id = str(obj_info["id"])
        position = obj_info["position"]
        quaternion = obj_info["orientation"]
        spatials_info[obj_id] = {
            "position": position,
            "quaternion": quaternion
        }
    
    if "scene_info" in env_config and "robot_info" in env_config["scene_info"]:
        robot_info = env_config["scene_info"]["robot_info"]
        robot_position = [0.0, 0.0, 0.0]
        robot_orientation = [1, 0, 0, 0]
        
        # robot_position = robot_info["robot_position"]

        spatials_info["franka_fr3"] = {
            "position": robot_position,
            "quaternion": robot_orientation
        }
    
    #for lighting
    if "scene_info" in env_config.keys():
        lighting_info = dr_state_dict["lighting"]={}

        light_info=env_config["scene_info"]["light_info"]

        light_num = light_info["light_num"]
        light_spacing = light_info["light_spacing"]  
        if isinstance(light_num, list) and len(light_num) == 2:
            for i in range(light_num[0]):
                for j in range(light_num[1]):
                    
                    uid = f"Light_{i}_{j}"
                    type = "CylinderLight"
                    
                    position=[
                        light_spacing[1] * (j - 0.5 * (light_num[1]-1)),
                        light_spacing[0] * (i - 0.5 * (light_num[0]-1)),
                        0.0
                    ]
                    
                    light_radius = light_info["light_radius"]
                    light_length = light_info["light_length"]
                    light_intensity = light_info["light_intensity"]
                    light_color_temperature = light_info["light_color_temperature"]
                    center_light_position = light_info["light_position"]
                    center_light_orientation = light_info["light_orientation"]

                    lighting_info[uid] = {'uid':uid,
                                        'type':type,
                                        'pose':{"position":position},
                                        'light_radius':light_radius,
                                        'light_length':light_length,
                                        'light_intensity':light_intensity,
                                        'light_color_temperature':light_color_temperature,
                                        'center_light_postion':center_light_position,
                                        'center_light_orientation':center_light_orientation
                                        }
                
    #for scene
    if "scene_info" in env_config.keys(): 
        scene_info = dr_state_dict["scene"]={}
        # scene_info['uid'] = f"hssd:{env_config["scene_info"]["hssd_scene"]['name']}"
        scene_uid = get_scene_uid(env_config["scene_info"]["hssd_scene"])
        scene_info['uid'] = f"hssd:{scene_uid}"

        scene_info["table"]=  {
            "size":env_config["scene_info"]["robot_info"]["table_scale"],
            "pose":{
                "position":env_config["scene_info"]["robot_info"]["table_position"],
                "quaternion":env_config["scene_info"]["robot_info"]["table_orientation"]
            }
        }
        
        scene_info['center_offset_limit_up'] = env_config["scene_info"]['hssd_scene']['center_offset_limit_up']
        scene_info['center_offset_limit_down'] = env_config["scene_info"]['hssd_scene']['center_offset_limit_down']
            
        # scene_info["surface"] = env_config["scene_info"]["surface"]
        scene_info['center_offset'] = env_config["scene_info"]['hssd_scene']["center_offset"]
        scene_info['center_orientation'] = env_config["scene_info"]['hssd_scene']["center_orientation"]

        dr_state_dict["material"] = env_config["scene_info"]["material_info"]

    return state_dict

def get_scene_uid(scene_info):
    for k, v in scene_info.items():
        if k.startswith("scene") and v is None:
            return k