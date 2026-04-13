"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""
import os
os.environ["_TYPER_STANDARD_TRACEBACK"]="1"
from simple.mp.curobo import CuRoboPlanner
from simple.agents.mp import MotionPlannerAgent
from simple.tasks.registry import TaskRegistry
from simple.core.task import Task
from simple.core.robot import Robot
from simple.assets import AssetManager
import os
import pickle
from tqdm import tqdm
import numpy as np
from collections import deque 
import datetime
import transforms3d as t3d

from curobo.util.logger import setup_curobo_logger
setup_curobo_logger("error")

from tqdm.contrib import DummyTqdmFile
import contextlib
import traceback
import sys
import typer
from typing_extensions import Annotated
from typing import Optional
from simple.utils import dump_json




@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    """ redirect logging for tqdm """
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err



def forward_trajectory(kin_model, finger_length, fname, hz, trajectories): 
    """ for debug only """
    import torch
    from curobo.types.base import TensorDeviceType
    """ kin_model: forward kinematic model
        trajectories: List motion planned by curobo 
    """
    joint_traj = []
    for traj in trajectories:
        for t in range(len(traj)):
            joint_traj.append(torch.from_numpy(traj[t]))

    joint_traj = torch.stack(joint_traj)
    tensor_args = TensorDeviceType()
    joint_traj = tensor_args.to_device(joint_traj)
    out = kin_model.get_state(joint_traj)

    eef_poses = []
    with open(f'{fname}', 'w') as f:
        current_datetime = datetime.datetime.now()
        t = current_datetime.timestamp()
        for p, q in zip(out.ee_position, out.ee_quaternion):
            p = p.cpu().numpy()
            q = q.cpu().numpy()
            # hand to eef
            p += t3d.quaternions.rotate_vector([0,0,finger_length], q) 
            f.write(f"{t} {p[0]} {p[1]} {p[2]} {q[-1]} {q[0]} {q[1]} {q[2]}\n") # tum: timestamp x y z q_x q_y q_z q_w
            t = t + 1. / hz
            eef_poses.append(np.concatenate([p, q], axis=-1))

    return eef_poses

def dump_actions_to_save(mp_agent: MotionPlannerAgent):
    
    approach_traj = []
    lift_traj = []
    
    for action_cmd in mp_agent._action_queue:
        if action_cmd.type == "move_qpos_with_eef" and action_cmd.parameters.get('eef_state')== 'close_eef':
            qpos = np.array(list(action_cmd.parameters['target_qpos'].values()))
            lift_traj.append(qpos)
        elif action_cmd.type == "move_qpos":
            qpos = np.array(list(action_cmd.parameters['target_qpos'].values()))
            approach_traj.append(qpos)

    return [approach_traj, lift_traj]


def main(
    task: Annotated[str, typer.Option()] = "franka_tabletop_grasp",
    save_label: Annotated[str, typer.Option()] = "plan",
    save_dir: Annotated[str, typer.Option()] = "data/output",
    num_poses: Annotated[int, typer.Option()] = 40,
    render_hz: Annotated[int, typer.Option()] = 30,
    dr_level: Annotated[int, typer.Option()] = 0,
    pool_size: Annotated[int, typer.Option()] = 10,
    target_id: Annotated[int, typer.Option()] = 63,
):
    dump_trajectory = False # dump trajector for visualization
    max_try = 3 # max allowed consecutive failed curobo planning

    target_object = f"graspnet1b:{target_id:02d}"
    task = TaskRegistry.make(task, target_object=target_object, render_hz=render_hz, dr_level=dr_level)
    # task.reset()
    
    planner = CuRoboPlanner(
        robot=task.robot,
        plan_batch_size=num_poses,
        plan_dt=0.01,
    )
    
    save_dir = f"{save_dir}/{save_label}/lv{dr_level}"
    save_folder = f"{save_dir}/{target_id:02d}"
    os.makedirs(save_folder, exist_ok=True)

    # uid = int(task.layout.actors['target'].asset.uid)
    with std_out_err_redirect_tqdm() as orig_stdout:
        # for target_object_id in tqdm([uid], desc="Handling object", file=orig_stdout):
        # save_folder = f"{save_dir}/{target_id:02d}"
        with tqdm(total = pool_size, desc="Planning: ", file=orig_stdout, leave=False) as pbar:
            retry = 0
            plan_index = 0
            while plan_index < pool_size:
                task.reset()
                
                mp_agent = MotionPlannerAgent(task, planner)
                if mp_agent.synthesize() is False: # plan failed
                    print(f"Task synthesize failed for object {target_id}")
                    retry += 1
                    if retry > max_try:
                        print(f"reached maximum retry for object {target_id}")
                        break
                    continue   
                # plan: [approach_traj, lift_traj]
                plan = dump_actions_to_save(mp_agent)
                state_dict = task.state_dict()
                
                pbar.update(1)
                retry = 0
                pickle.dump([plan, state_dict], open(f"{save_folder}/p{plan_index:03d}.pkl", "wb"))
                plan_index += 1
    
def typer_main():
    typer.run(main)

if __name__ == "__main__":
    typer.run(main)

