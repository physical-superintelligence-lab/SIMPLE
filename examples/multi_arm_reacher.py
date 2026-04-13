"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

try:
    # Third Party
    import isaacsim
except ImportError:
    pass


# Third Party
import torch

a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)

parser.add_argument(
    "--robot", type=str, default="dual_ur10e.yml", help="robot configuration to load"
)
args = parser.parse_args()

############################################################

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)
# Third Party
import carb
import numpy as np
from matplotlib import cm
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid
from omni.isaac.core.robots import Robot
from pxr import UsdPhysics

from curobo.util.logger import log_warn
from curobo.util.usd_helper import set_prim_transform

ISAAC_SIM_23 = False
ISAAC_SIM_45 = False
try:
    # Third Party
    from omni.isaac.urdf import _urdf  # isaacsim 2022.2
except ImportError:
    # Third Party
    try:
        from omni.importer.urdf import _urdf  # isaac sim 2023.1 or above
    except ImportError:
        from isaacsim.asset.importer.urdf import _urdf  # isaac sim 4.5+

        ISAAC_SIM_45 = True
    ISAAC_SIM_23 = True

try:
    # import for older isaacsim installations
    from omni.isaac.core.materials import OmniPBR
except ImportError:
    # import for isaac sim 4.5+
    from isaacsim.core.api.materials import OmniPBR


from typing import Optional, Dict
from omni.isaac.core.utils.extensions import enable_extension
from curobo.util_file import get_assets_path, get_filename, get_path_of_dir, join_path

def add_extensions(simulation_app, headless_mode: Optional[str] = None):
    ext_list = [
        "omni.kit.asset_converter",
        "omni.kit.tool.asset_importer",
        "omni.isaac.asset_browser",
    ]
    if headless_mode is not None:
        log_warn("Running in headless mode: " + headless_mode)
        ext_list += ["omni.kit.livestream." + headless_mode]
    [enable_extension(x) for x in ext_list]
    simulation_app.update()

    return True

def add_robot_to_scene(
    robot_config: Dict,
    my_world: World,
    load_from_usd: bool = False,
    subroot: str = "",
    robot_name: str = "robot",
    position: np.array = np.array([0, 0, 0]),
    initialize_world: bool = True,
):

    urdf_interface = _urdf.acquire_urdf_interface()
    # Set the settings in the import config
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = False
    import_config.default_drive_strength = 100 #1047.19751
    import_config.default_position_drive_damping = 20 # 52.35988
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0

    asset_path = get_assets_path()
    if (
        "external_asset_path" in robot_config["kinematics"]
        and robot_config["kinematics"]["external_asset_path"] is not None
    ):
        asset_path = robot_config["kinematics"]["external_asset_path"]

    # urdf_path:
    # meshes_path:
    # meshes path should be a subset of urdf_path
    full_path = join_path(asset_path, robot_config["kinematics"]["urdf_path"])
    # full path contains the path to urdf
    # Get meshes path
    robot_path = get_path_of_dir(full_path)
    filename = get_filename(full_path)
    if ISAAC_SIM_45:
        from isaacsim.core.utils.extensions import get_extension_path_from_name
        import omni.kit.commands
        import omni.usd

        # Retrieve the path of the URDF file from the extension
        extension_path = get_extension_path_from_name("isaacsim.asset.importer.urdf")
        root_path = robot_path
        file_name = filename

        # Parse the robot's URDF file to generate a robot model

        dest_path = join_path(
            root_path, get_filename(file_name, remove_extension=True) + "_temp.usd"
        )

        result, robot_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path="{}/{}".format(root_path, file_name),
            import_config=import_config,
            dest_path=dest_path,
        )
        prim_path = omni.usd.get_stage_next_free_path(
            my_world.scene.stage,
            str(my_world.scene.stage.GetDefaultPrim().GetPath()) + robot_path,
            False,
        )
        robot_prim = my_world.scene.stage.OverridePrim(prim_path)
        robot_prim.GetReferences().AddReference(dest_path)
        robot_path = prim_path
    else:

        imported_robot = urdf_interface.parse_urdf(robot_path, filename, import_config)
        dest_path = subroot

        robot_path = urdf_interface.import_robot(
            robot_path,
            filename,
            imported_robot,
            import_config,
            dest_path,
        )

    base_link_name = robot_config["kinematics"]["base_link"]

    robot_p = Robot(
        prim_path=robot_path + "/" + base_link_name,
        name=robot_name,
    )

    robot_prim = robot_p.prim
    stage = robot_prim.GetStage()
    linkp = stage.GetPrimAtPath(robot_path)
    set_prim_transform(linkp, [position[0], position[1], position[2], 1, 0, 0, 0])

    robot = my_world.scene.add(robot_p)
    if initialize_world:
        if ISAAC_SIM_45:
            my_world.initialize_physics()
            robot.initialize()

    # import omni.isaac.core.utils.stage as isaacsim_stage
    # isaacsim_stage.add_reference_to_stage(
    #     usd_path=robot_config["kinematics"]["usd_path"],
    #     prim_path=f"/World/aloha",
    # )
    # robot = Robot(prim_path=f"/World/aloha")

    # if initialize_world:
    #     if ISAAC_SIM_45:
    #         my_world.initialize_physics()
    #         robot.initialize()

    return robot, robot_path

# from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere

########### OV #################
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

############################################################


########### OV #################;;;;;


############################################################


def main():
    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
    stage = my_world.stage
    # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # Make a target to follow

    setup_curobo_logger("warn")
    past_pose = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    tensor_args = TensorDeviceType()

    if args.robot.endswith("yml"):
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

        robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
    else:
        import os
        import pathlib
        from importlib.resources import files
        
        src_path = files("simple").parent.parent 
        # src_path = pathlib.Path(__file__).resolve().parents[1] / "src" / "simple" / "resources" 
        robot_cfg = load_yaml(str(src_path / "data" / "robots" / args.robot / "curobo" / f"{args.robot}.yml"))["robot_cfg"]

        # fix: replace absolute path with data_dir
        for key in ["external_asset_path", "external_robot_configs_path", "usd_path"]:
            if robot_cfg["kinematics"][key] is not None:
                abs_path = robot_cfg["kinematics"][key].replace("<PROJECT_DIR>", str(src_path/""))
                if not os.path.isabs(abs_path):
                    abs_path = str(src_path / "data" / abs_path)
                robot_cfg["kinematics"][key] = abs_path

        robot_prim_path = f"/World/{args.robot}"
        import omni.isaac.core.utils.stage as isaacsim_stage
        isaacsim_stage.add_reference_to_stage(
            usd_path=robot_cfg["kinematics"]["external_asset_path"] + f"/{args.robot}.usd",
            prim_path=robot_prim_path,
        )
        
        robot = Robot(prim_path=robot_prim_path)
        if ISAAC_SIM_45:
            my_world.initialize_physics()
            robot.initialize()

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    articulation_controller = robot.get_articulation_controller()

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.02

    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        use_cuda_graph=True,
        interpolation_dt=0.03,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        collision_activation_distance=0.025,
        fixed_iters_trajopt=True,
        maximum_trajectory_dt=0.5,
        ik_opt_iters=500,
    )
    motion_gen = MotionGen(motion_gen_config)
    print("warming up...")
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

    print("Curobo is Ready")
    add_extensions(simulation_app, args.headless_mode)
    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=4,
        max_attempts=10,
        time_dilation_factor=0.5,
    )

    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    cmd_plan = None
    cmd_idx = 0
    my_world.scene.add_default_ground_plane()
    i = 0
    spheres = None

    # read number of targets in link names:
    link_names = motion_gen.kinematics.link_names
    ee_link_name = motion_gen.kinematics.ee_link
    # get link poses at retract configuration:

    kin_state = motion_gen.kinematics.get_state(motion_gen.get_retract_config().view(1, -1))

    link_retract_pose = kin_state.link_pose
    t_pos = np.ravel(kin_state.ee_pose.to_list())
    target = cuboid.VisualCuboid(
        "/World/target",
        position=t_pos[:3],
        orientation=t_pos[3:],
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

    # create new targets for new links:
    ee_idx = link_names.index(ee_link_name)
    target_links = {}
    names = []
    for i in link_names:
        if i != ee_link_name:
            k_pose = np.ravel(link_retract_pose[i].to_list())
            color = np.random.randn(3) * 0.2
            color[0] += 0.5
            color[1] = 0.5
            color[2] = 0.0
            target_links[i] = cuboid.VisualCuboid(
                "/World/target_" + i,
                position=np.array(k_pose[:3]),
                orientation=np.array(k_pose[3:]),
                color=color,
                size=0.05,
            )
            names.append("/World/target_" + i)
    i = 0
    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            # if step_index == 0:
            #    my_world.play()
            continue

        step_index = my_world.current_time_step_index
        # print(step_index)
        if step_index <= 10:
            # my_world.reset()
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([10000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
        if step_index < 20:
            continue

        if step_index == 50 or step_index % 1000 == 0.0:
            print("Updating world, reading w.r.t.", robot_prim_path)
            obstacles = usd_help.get_obstacles_from_stage(
                only_paths=["/World"],
                reference_prim_path=robot_prim_path,
                ignore_substring=[
                    robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ]
                + names,
            ).get_collision_check_world()

            motion_gen.update_world(obstacles)
            print("Updated World")
            carb.log_info("Synced CuRobo world from stage.")

        # position and orientation of target virtual cube:
        cube_position, cube_orientation = target.get_world_pose()

        if past_pose is None:
            past_pose = cube_position
        if target_pose is None:
            target_pose = cube_position
        sim_js = robot.get_joints_state()
        if sim_js is None:
            print("sim_js is None")
            continue
        sim_js_names = robot.dof_names
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

            if spheres is None:
                spheres = []
                # create spheres:

                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=np.array([0, 0.8, 0.2]),
                    )
                    spheres.append(sp)
            else:
                for si, s in enumerate(sph_list[0]):
                    spheres[si].set_world_pose(position=np.ravel(s.position))
                    spheres[si].set_radius(float(s.radius))
        if (
            np.linalg.norm(cube_position - target_pose) > 1e-3
            and np.linalg.norm(past_pose - cube_position) == 0.0
            # and np.max(np.abs(sim_js.velocities)) < 0.5
        ):
            # Set EE teleop goals, use cube for simple non-vr init:
            ee_translation_goal = cube_position
            ee_orientation_teleop_goal = cube_orientation

            # compute curobo solution:
            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )
            # add link poses:
            link_poses = {}
            for i in target_links.keys():
                c_p, c_rot = target_links[i].get_world_pose()
                link_poses[i] = Pose(
                    position=tensor_args.to_device(c_p),
                    quaternion=tensor_args.to_device(c_rot),
                )
            result = motion_gen.plan_single(
                cu_js.unsqueeze(0), ik_goal, plan_config.clone(), link_poses=link_poses
            )
            # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

            succ = result.success.item()  # ik_result.success.item()
            if succ:
                cmd_plan = result.get_interpolated_plan()
                cmd_plan = motion_gen.get_full_js(cmd_plan)
                # get only joint names that are in both:
                idx_list = []
                common_js_names = []
                for x in sim_js_names:
                    if x in cmd_plan.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)
                # idx_list = [robot.get_dof_index(x) for x in sim_js_names]

                cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)

                cmd_idx = 0

            else:
                carb.log_warn("Plan did not converge to a solution: " + str(result.status))
            target_pose = cube_position
        past_pose = cube_position
        if cmd_plan is not None:
            cmd_state = cmd_plan[cmd_idx]

            # get full dof state
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy(),
                joint_indices=idx_list,
            )
            # set desired joint angles obtained from IK:
            articulation_controller.apply_action(art_action)
            cmd_idx += 1
            for _ in range(2):
                my_world.step(render=False)
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None
    simulation_app.close()


if __name__ == "__main__":
    main()
