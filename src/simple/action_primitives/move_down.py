import numpy as np
import torch
import transforms3d as t3d


def move_down(robot, arm='left', distance_cm=3.0, frame='world', camera_pose=None):
    """
    Move the robot's end-effector down by a specified distance using curobo IK.
    
    This action primitive uses curobo's IK solver to compute the joint positions
    needed to move the end-effector down while maintaining its orientation.
    
    Args:
        robot: Robot instance with curobo IK support (e.g., Aloha, Franka)
        arm: Which arm to move - 'left' or 'right' (default: 'left')
        distance_cm: Distance to move in centimeters (default: 3.0 cm)
        frame: Reference frame - 'world' or 'camera' (default: 'world')
        camera_pose: Camera pose (position, quaternion) for camera frame movements.
                     Format: (position_xyz, quaternion_xyzw). Required if frame='camera'.
    
    Returns:
        target_qpos: Target joint positions (numpy array) or None if IK fails
    """
    distance_m = distance_cm / 100.0

    current_qpos = robot.get_robot_qpos()    
    left_joints = robot.extract_arm_joints(current_qpos, arm='left')
    right_joints = robot.extract_arm_joints(current_qpos, arm='right')
    

    left_ee_pos, left_ee_quat = robot.fk(left_joints, arm='left')
    right_ee_pos, right_ee_quat = robot.fk(right_joints, arm='right')

    if frame == 'world':
        # TODO
        pass
    elif frame == 'camera':
        if arm == 'left':
            target_ee_pos = left_ee_pos + np.array([0.0, 0.0, -distance_m])
            target_ee_quat = left_ee_quat
        else:
            target_ee_pos = right_ee_pos + np.array([0.0, 0.0, -distance_m])
            target_ee_quat = right_ee_quat
    
    try:
        if arm == 'left':
            target_arm_qpos = robot.ik(target_ee_pos, target_ee_quat, current_joint=left_joints, arm=arm)
            target_arm_qpos = np.concatenate([target_arm_qpos, right_joints])
        else:
            target_arm_qpos = robot.ik(target_ee_pos, target_ee_quat, current_joint=right_joints, arm=arm)
            target_arm_qpos = np.concatenate([left_joints, target_arm_qpos])
        
        # For now we assume grippers are open, we can use other primitives to modify the gripper values
        gripper_actions = np.array([0.04, 0.04])
        target_action = np.concatenate([target_arm_qpos, gripper_actions])
        return target_action
    except RuntimeError as e:
        print(f"[move_left] IK failed: {e}")
        return None
    except Exception as e:
        print(f"[move_left] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None