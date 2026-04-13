"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import numpy as np
import transforms3d as t3d
from PIL import Image
import matplotlib.pyplot as plt
from queue import Queue

from simple.baselines.bresenham_line import bresenham_thick_line
from simple.agents.base_agent import BaseAgent, GripperAction, GripperState
from simple.baselines.client import HttpActionClient
from simple.core.action import ActionCmd
from simple.robots.protocols import HasParallelGripper

def hack_gripper_action(gripper_act):
    return 0 if gripper_act < 0.7 else 1    

def center_crop(im):
    width, height = im.size   # Get dimensions
    new_width, new_height = min(width, height), min(width, height)  # Set new dimensions to be the smaller of the two
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im

def project_2d_traj(camera_infos, eef_pose, cam_name="front_stereo_left"):
    fx, fy, cx, cy, cam_p, cam_q = None, None, None, None, None, None
    # camera_infos = env_config["scene_info"]["camera_info"]
    for cam_info in camera_infos:
        if cam_info["name"] == "front_stereo" and cam_name == "front_stereo_left":
            fx, fy, cx, cy = cam_info["fx"], cam_info["fy"], cam_info["cx"], cam_info["cy"]
            cam_p = np.array(cam_info["position"], dtype=np.float32)
            cam_q = np.array(cam_info["orientation"], dtype=np.float32)
            break
        elif cam_info["name"] == "front_stereo" and cam_name == "front_stereo_right":
            fx, fy, cx, cy = cam_info["fx"], cam_info["fy"], cam_info["cx"], cam_info["cy"]
            cam_p = np.array(cam_info["position"], dtype=np.float32)
            cam_q = np.array(cam_info["orientation"], dtype=np.float32)
            cam_p = cam_p + -t3d.quaternions.quat2mat(cam_q)[:3, 1] * 0.055 # -y axis (right)
            break
        elif cam_info["name"] == cam_name:
            fx, fy, cx, cy = cam_info["fx"], cam_info["fy"], cam_info["cx"], cam_info["cy"]
            cam_p = np.array(cam_info["position"], dtype=np.float32)
            cam_q = np.array(cam_info["orientation"], dtype=np.float32)
            break
    
    if fx is None:
        raise NotImplementedError(f"Camera name {cam_name} for trajectory projection is not implemented")
    
    K = np.array([[fx, 0, cx], 
                [0, fy, cy], 
                [0,  0,  1]], dtype=np.float32)
    
    T_robot_cam = np.eye(4)
    T_robot_cam[:3, :3] = t3d.quaternions.quat2mat(cam_q)
    T_robot_cam[:3,  3] = cam_p
    
    eef_traj_2d = []
    for step_idx in range(eef_pose.shape[0]):
        p = eef_pose[step_idx, :3]
        q = eef_pose[step_idx, 3:]
        
        T_robot_eef = np.eye(4)
        T_robot_eef[:3, :3] = t3d.quaternions.quat2mat(q)
        T_robot_eef[:3,  3] = p

        T_hand_eef = np.eye(4)
        # T_hand_eef[2, 3] = 0.126  # CHENGYANG: the offset has been included in the eef pose read from the dataset
        
        T_cam_hand = np.linalg.inv(T_robot_cam) @ T_robot_eef @ T_hand_eef
        switch_frame = np.array([ # isaac sim to openGL
            [0,-1, 0, 0],
            [0, 0,-1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ])
        T_cam_hand = switch_frame @ T_cam_hand # camera frame

        p_in_cam = T_cam_hand[:3, 3]
        p_in_2d = K @ p_in_cam / p_in_cam[2]
        eef_traj_2d.append(p_in_2d[:2])
    
    eef_traj_2d = np.array(eef_traj_2d, dtype=np.float32)
    return eef_traj_2d

def get_forward_traj_by_nearest_point(traj_3d, current_eef_pose, first_close_step_idx, first_close_point_reached, interpolate=False):
    curr_p = current_eef_pose[:3]
    p_traj = traj_3d[:, :3]
    dists = np.linalg.norm(p_traj - curr_p, axis=1)
    nearest_idx = np.argmin(dists)
    
    if not first_close_point_reached:
        nearest_idx = min(nearest_idx, first_close_step_idx)
    else:
        nearest_idx = p_traj.shape[0] - 1
    
    forward_traj = traj_3d[nearest_idx:].reshape(-1, 7)
    if interpolate:
        mean_dist = np.mean(np.linalg.norm(np.diff(p_traj, axis=0), axis=1))
        num_steps = int(np.linalg.norm(forward_traj[0, :3] - curr_p) / mean_dist)
        # interpolate a traj from current pose to the nearest point
        interp_p_traj = np.linspace(curr_p, forward_traj[0, :3], num_steps)
        interp_traj = np.hstack([interp_p_traj, np.tile(forward_traj[0, 3:], (num_steps, 1))])
        if len(forward_traj) > 1:
            forward_traj = np.vstack([interp_traj, forward_traj[1:].reshape(-1, 7)])
        else:
            forward_traj = interp_traj
    return forward_traj

def draw_2d_traj(img, traj_2d, thickness=5):
    # bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    traj_len = len(traj_2d)
    cmap = plt.get_cmap("Spectral")
    canvas = img.copy() 
    for i, (u,v) in enumerate(traj_2d):
        if i >= traj_len - 1:
            continue

        tint_start = (np.array(cmap(i/traj_len)[:3]) * 255).astype(np.uint8)
        tint_end = (np.array(cmap((i+1)/traj_len)[:3]) * 255).astype(np.uint8)
                                                                     
        start_point = traj_2d[i]
        end_point = traj_2d[i+1]

        bresenham_thick_line(canvas, 
                             round(start_point[0]), round(start_point[1]), 
                             round(end_point[0]), round(end_point[1]),
                             thickness, tint_start, tint_end)
    return canvas

def resize_traj2d(eef_traj_2d, H, W, newH, newW):
    assert H < W, "adjust some lines below"
    
    scale = H / newH

    eef_traj_2d_resized = []
    for p in eef_traj_2d:
        u, v = p
        u = np.clip((u - (W-H)/2) / scale, 0, newW-1)
        v = np.clip(v / scale,0, newH-1)
        
        eef_traj_2d_resized.append((u, v))
    return eef_traj_2d_resized

def draw_traj2d_resized(canvas, eef_traj_2d_resized, newH, newW, thickness=5):
    cmap = plt.get_cmap("Spectral")
    traj_len = len(eef_traj_2d_resized)
    for i in range(traj_len-1):

        tint_start = (np.array(cmap(i/traj_len)[:3]) * 255).astype(np.uint8)
        tint_end = (np.array(cmap((i+1)/traj_len)[:3]) * 255).astype(np.uint8)

        start_point = eef_traj_2d_resized[i]
        end_point = eef_traj_2d_resized[i+1]

        bresenham_thick_line(canvas, 
                             round(start_point[0]), round(start_point[1]), 
                             round(end_point[0]), round(end_point[1]),
                             thickness, tint_start, tint_end)
    return canvas

class VltAgent(BaseAgent):
    def __init__(self, robot, host: str, port: int):
        super().__init__(robot)
        
        self.server_ip = host # if access server host inside docker container
        self.server_port = port

        self.client = HttpActionClient(self.server_ip, self.server_port)
        self.action_buffer = Queue() # to support action chunking

        # self.step_idx = 0
        self.ik_failed_cnt = 0

        # self.reset(episode=episode, condition=condition, save_cond_images=save_cond_images)

    def get_action(self, observation, instruction=None, info=None, conditions=None, **kwargs):
        assert instruction is not None, "Instruction must be provided for DitPolicyAgent."

        self._last_observation = observation

        # if conditions is None:
        #     conditions = {}

        if np.linalg.norm(observation["eef_pose"][:3] - self.first_close_point) < 0.01:
            self.first_close_point_reached = True

        if self.condition == "full_traj":
            traj_3d = self.episode["additional"]["proprio_eef_pose"]
        
        elif self.condition.startswith("forward_"):
            # find the first close action index
            length = self.condition.split("_")[1]
            if length == "all":
                traj_3d = get_forward_traj_by_nearest_point(self.episode["additional"]["proprio_eef_pose"], 
                                                            observation["eef_pose"], 
                                                            self.first_close_step_idx, 
                                                            self.first_close_point_reached, 
                                                            interpolate=True)
            else:
                raise NotImplementedError(f"Forward condition with length {length} not implemented")
        
            traj_2d_front_stereo_left = project_2d_traj(self.camera_infos, traj_3d, cam_name="front_stereo_left")
            image_traj_front_stereo_left = draw_2d_traj(observation["front_stereo_left"], traj_2d_front_stereo_left)
            traj_2d_front_stereo_right = project_2d_traj(self.camera_infos, traj_3d, cam_name="front_stereo_right")
            image_traj_front_stereo_right = draw_2d_traj(observation["front_stereo_right"], traj_2d_front_stereo_right)
            traj_2d_side = project_2d_traj(self.camera_infos, traj_3d, cam_name="side_left")
            image_traj_side = draw_2d_traj(observation["side_left"], traj_2d_side)
            
            if self.save_cond_images:
                self.vis_cond_images[f"{self.condition}_front_stereo_left"].append(image_traj_front_stereo_left)
                self.vis_cond_images[f"{self.condition}_front_stereo_right"].append(image_traj_front_stereo_right)
                self.vis_cond_images[f"{self.condition}_side_left"].append(image_traj_side)

            H1, W1 = 360, 640
            newH, newW = (256, 256)
            eef_traj_2d_left = resize_traj2d(traj_2d_front_stereo_left, H1, W1, newH, newW)
            eef_traj_2d_right = resize_traj2d(traj_2d_front_stereo_right, H1, W1, newH, newW)
            eef_traj_2d_side = resize_traj2d(traj_2d_side, H1, W1, newH, newW)

            image_traj_front_stereo_left_cond = draw_traj2d_resized(np.zeros((newH, newW, 3), dtype=np.uint8), eef_traj_2d_left, newH, newW)
            image_traj_front_stereo_right_cond = draw_traj2d_resized(np.zeros((newH, newW, 3), dtype=np.uint8), eef_traj_2d_right, newH, newW)
            image_traj_side_cond = draw_traj2d_resized(np.zeros((newH, newW, 3), dtype=np.uint8), eef_traj_2d_side, newH, newW)
            
            self.conditions["image_left_future_traj_2d"] = image_traj_front_stereo_left_cond
            self.conditions["image_right_future_traj_2d"] = image_traj_front_stereo_right_cond
            self.conditions["image_side_future_traj_2d"] = image_traj_side_cond

        if False and step_cnt < len(raw_eps_data.keys()):
            unnorm_img = lambda x: ((0.5*x + 0.5) * 255).astype(np.uint8)
            sim_left = observation["front_stereo_left"]
            sim_cond = conditions["image_left_future_traj_2d"]

            from PIL import Image
            inspect = lambda x: (x.shape, x.max(), x.min(), x.mean(), x.std())
            pt_to_pil = lambda x: Image.fromarray(
                (((x.float() * 0.5 + 0.5).clamp(0, 1))*255.0).permute(1,2,0).cpu().numpy().astype(np.uint8)
            )

            frame = raw_eps_data[step_cnt]
            left = pt_to_pil(frame[0][0])
            cond = pt_to_pil(frame[1])
            obs = frame[2].cpu().numpy()
            action = frame[3].cpu().numpy()
            instruction = frame[4]

            left.save(f"{step_cnt}-left.png")
            cond.save(f"{step_cnt}-cond.png")

            sim_left = Image.fromarray(sim_left) # center_crop(Image.fromarray(sim_left)).resize((256, 256), Image.BILINEAR)
            sim_left.save(f"{step_cnt}-left_sim.png")

            sim_cond = Image.fromarray(sim_cond)
            sim_cond.save(f"{step_cnt}-cond_sim.png")

            observation["front_stereo_left"] = np.array(left)
            conditions["image_left_future_traj_2d"] = np.array(cond)
            observation["joint_qpos"] = obs[0, :8]
            observation["eef_pose"] = obs[0, -7:]

        if self.action_buffer.qsize() == 0:
            image_dict = {
                "front_stereo_left": observation["front_stereo_left"],
                "front_stereo_right": observation["front_stereo_right"],
                "wrist_mount_cam": observation["wrist"],
                "side": observation["side_left"],
            }
            state_dict = {
                "proprio_joint_positions": observation["joint_qpos"], # (9,)
                "proprio_eef_pose": observation["eef_pose"], # (7,)
            }
            history_dict = {k: [] for k in image_dict.keys()}
            condition_dict = conditions if conditions is not None else self.conditions
            self.condition_dict = condition_dict
            self.state_dict = state_dict
            self.history_dict = history_dict

            pred_actions, *_ = self.client.query_action(
                image_dict=image_dict, 
                instruction=instruction, 
                state_dict=state_dict, 
                condition_dict=condition_dict, 
                history=history_dict
            ) # delta: xyz, rpy, openness

            pred_actions = np.array(pred_actions, dtype=np.float32)
            print(f"received {pred_actions.shape[0]} actions") # for dit-policy-vl (7,)

            if len(pred_actions.shape) == 2: # N x 7 for action chunking
                for act in pred_actions:
                    self.action_buffer.put(act)
            else: # single action
                self.action_buffer.put(pred_actions)

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
            target_qpos = self.robot.ik(p, q, curr_qpos)  #[:7]
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
        return ActionCmd(
            "queue_move_qpos_with_eef", 
            target_qpos=dict(zip(self.robot.joint_names[:arm_dim], target_qpos)),
            eef_state=gripper_state
        )
    
    def reset(self, camera_infos=None, episode=None,  condition=None, save_cond_images=False, **kwargs):
        super().reset(**kwargs)

        self.ik_failed_cnt = 0
        self.action_buffer = Queue()

        if hasattr(self, "vis_cond_images"):
            import cv2,  os , time
            cond_video_path  = f"{time.time()}.mp4"
            for cond, cond_imgs in self.vis_cond_images.items():
                print(f"Saving {len(cond_imgs)} {cond} images")
                # cond_video_path = f"{env.work_dir}/{env.task_id}/{cond}.mp4"
                cond_video_path  = f"{cond}_{time.time()}.mp4"
                video_writer = cv2.VideoWriter(cond_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (cond_imgs[0].shape[1], cond_imgs[0].shape[0]))
                for img in cond_imgs:
                    video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                video_writer.release()
                # suffix = "success" if (not truncated) else "failed"
                # os.system(f"ffmpeg -i {cond_video_path} -vcodec libx264 {cond_video_path[:-4]}.mp4 > /dev/null 2>&1")
                # os.system(f"rm {cond_video_path}")


        self.conditions = {}
        self.save_cond_images = save_cond_images
        if episode is not None: # HACK
            self.first_close_step_idx = np.where(episode["action"][:, 0, -1] < 0.5)[0][0]
            self.first_close_point = episode["additional"]["proprio_eef_pose"][self.first_close_step_idx, :3]

        self.first_close_point_reached = False
        self.episode = episode
        self.camera_infos = camera_infos # episode["env_cfg"] if episode is not None else None

        if save_cond_images:
            self.vis_cond_images = {
                f"{condition}_front_stereo_left": [],
                f"{condition}_front_stereo_right": [],
                f"{condition}_side_left": []
            }
            
        self.condition = condition
