"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""
import simple
import typer
from typing_extensions import Annotated
from simple.tasks.registry import TaskRegistry
from simple.utils import resolve_data_path, dump_json #, timestamp_str

def main(
    task_uid: Annotated[str, typer.Option()] = "franka_tabletop_grasp",
    robot_uid: Annotated[str, typer.Option()] = "franka_fr3",
    controller_uid: Annotated[str, typer.Option()] = "pd_joint_pos",
    target_object: Annotated[str, typer.Option()] = "graspnet1b:63",
):
    task = TaskRegistry.make(
        task_uid, 
        robot_uid=robot_uid, 
        controller_uid=controller_uid, 
        target_object=target_object
    )
    task_cfg = task.save_config()
    dump_path = resolve_data_path("output/tasks", create_if_not_exist=True)
    version_number = task.metadata.get("version", simple.__version__)
    output_file = f"{dump_path}/{task.uid}_v{version_number}.json"
    dump_json(task_cfg, open(output_file, "w"), indent=4)
    print(f"Task definition dumped to {output_file}")
    
if __name__ == "__main__":
    typer.run(main)