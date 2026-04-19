import os
import sys
import argparse
import glob


# 1. 必须在导入任何 pxr 模块前启动 SimulationApp
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics, Sdf

def convert_obj_to_isaac_usd(obj_path, usd_output_path):
    stage = Usd.Stage.CreateNew(usd_output_path)
    
    # --- 1. World (Default Prim) ---
    world_path = Sdf.Path("/World")
    world_prim = UsdGeom.Xform.Define(stage, world_path)
    stage.SetDefaultPrim(world_prim.GetPrim())
    
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # --- 2. Meshes (Xform) ---
    meshes_path = world_path.AppendPath("Meshes")
    meshes_xform = UsdGeom.Xform.Define(stage, meshes_path)

    # --- 3. visual (作为容器引入 OBJ) ---
    visual_path = meshes_path.AppendPath("visual")
    # 关键修改：使用 Xform 而不是 Mesh，避免产生"没有顶点的空Mesh"
    visual_prim = UsdGeom.Xform.Define(stage, visual_path).GetPrim()
    visual_prim.GetReferences().AddReference(obj_path)

    # --- 4. collision (作为容器引入 OBJ) ---
    collision_path = meshes_path.AppendPath("collision")
    collision_prim = UsdGeom.Xform.Define(stage, collision_path).GetPrim()
    collision_prim.GetReferences().AddReference(obj_path)

    # --- 5. 给 collision 赋予物理属性 ---
    # 物理引擎会自动递归寻找 collision 容器内部的网格数据
    UsdPhysics.CollisionAPI.Apply(collision_prim)
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(collision_prim)
    # 对于 Objaverse 的复杂模型，使用凸包(convexHull)近似碰撞效果最好且不报错
    mesh_collision_api.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)

    # --- 6. 隐藏 collision 避免渲染重叠 ---
    UsdGeom.Imageable(collision_prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    # 保存并退出
    stage.GetRootLayer().Save()
    print(f"Success: Saved USD with correct hierarchy to {usd_output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert OBJ to Isaac USD with correct hierarchy.")
    parser.add_argument("--input_dir_template", type=str, help="Path to the input OBJ file.")
    parser.add_argument("--output_usd_template", type=str, help="Path to the output USD file.")
    args = parser.parse_args()

    input_dirs = glob.glob(args.input_dir_template)

    for input_dir in input_dirs:

        input_obj_dir = os.path.join(input_dir, "mesh", "normalized.obj")
        output_usd_dir = input_obj_dir.replace(".obj", "_isaac.usd")
        rel_obj_path = os.path.join(".", os.path.basename(input_obj_dir))
        convert_obj_to_isaac_usd(rel_obj_path, output_usd_dir)





    