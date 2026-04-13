"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--sim_mode", type=str, default="mujoco_isaac", help="Simulation mode, e.g., mujoco, isaac, mujoco_isaac")
parser.add_argument("--headless", type=str, default="false", help="use lower init robot pose")
parser.add_argument("--max_episode_steps", type=int, default="50", help="max episode steps")
parser.add_argument("--save_dir", type=str, default="output", help="directory to save results")
parser.add_argument("--save_label", type=str, default="tmp", help="label for saving results")
parser.add_argument("--data_dir", type=str, default="data", help="path to meta data")
parser.add_argument("--num_poses", type=int, default=40, help="num poses to plan in a curobo batch plan run")
parser.add_argument("--render_hz", type=int, default=10, help="render update rate")

parser.add_argument("--task", type=str, default="franka_tabletop_grasp", help="task to run")
parser.add_argument("--pool_size", type=int, default=100, help="number of traj pool size, the larger the length of trajs in one bacth the similar")
args, unknown = parser.parse_known_args()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
args.headless = str2bool(args.headless) # HACK bool 