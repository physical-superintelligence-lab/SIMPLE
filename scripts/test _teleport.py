import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.core.api import World
import isaacsim.core.utils.stage as stage_utils
import isaacsim.core.utils.prims as prims_utils
import time

from omni.isaac.core.prims import RigidPrim, GeometryPrim, XFormPrim
import omni.isaac.core.utils.stage as isaacsim_stage

world = World()
# world.reset()

""" # Create a cube prim using stage utilities
cube_prim_path = "/World/cube"
stage_utils.get_current_stage().DefinePrim(cube_prim_path, "Cube")

# Wrap the prim as a SingleGeometryPrim (physics disabled by default)
obj = world.scene.add(
    SingleGeometryPrim(
        prim_path=cube_prim_path,
        name="cube",
        position=np.array([-2, -6, 9]),
        scale=np.array([1, 1, 1]),
        collision=False  # No physics/collision
    )
) """

object_usd_path = "/home/songlin/workspace/projects/SIMPLE/data/assets/graspnet/ruled_models/000_ruled.usd"
object_prim_path = "/World/object"
isaacsim_stage.add_reference_to_stage(usd_path=object_usd_path, prim_path=object_prim_path)


obj_xform = XFormPrim(prim_path=object_prim_path)
geom_prim_path = f'{object_prim_path}/Meshes'
obj_geom = GeometryPrim(prim_path=geom_prim_path)
obj = RigidPrim(prim_path=geom_prim_path)
obj.disable_rigid_body_physics()
obj_collision_geom = GeometryPrim(f"{geom_prim_path}/collision")
obj_collision_geom.set_collision_enabled(False)

print(f'Initial cube pose:')
print(f'cube.get_world_pose(): {obj.get_world_pose()}')
print(f'cube.get_local_pose(): {obj.get_local_pose()}')

# Set up circular motion parameters
radius = 1.0  # radius of circular path
center = np.array([0, 0, 0])  # center point of circle
angular_speed = 1.0  # radians per second

# Animate cube in circular motion
while simulation_app.is_running():
    # Calculate current time
    t = time.time()
    
    # Calculate position on circle
    x = center[0] + radius * np.cos(angular_speed * t)
    y = center[1] + radius * np.sin(angular_speed * t) 
    z = center[2]
    
    # Update cube position
    obj.set_local_pose(translation=np.array([x, y, z]))
    
    # Step simulation
    world.step(render=True)

print(f'After setting local pose:')
print(f'cube.get_world_pose(): {obj.get_world_pose()}')
print(f'cube.get_local_pose(): {obj.get_local_pose()}')

simulation_app.close()