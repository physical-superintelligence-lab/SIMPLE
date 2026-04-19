
# 0. 必须在所有 omni 和 pxr 导入之前启动 SimulationApp
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.kit.commands
import omni.kit.app
import os
# 👇 导入 USD 底层核心库
from pxr import Usd, Sdf, UsdGeom 
from pxr import UsdPhysics

# 启用 URDF 导入插件
manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)

# ── 路径配置 ────────────────────────────────────────────────
current_dir = os.getcwd()
urdf_file_path = os.path.join(current_dir, "data/assets/articulated/005/005_scaled_05.urdf")
usd_save_path  = os.path.join(current_dir, "data/assets/articulated/005/output_usd/005.usd")
# ─────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(usd_save_path), exist_ok=True)
print(f"正在导入: {urdf_file_path}")

# 创建导入配置
status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")

# ── 物理参数 ──
import_config.set_fix_base(True)               
import_config.set_import_inertia_tensor(True)   
import_config.set_convex_decomp(False)          
import_config.set_self_collision(False)         
import_config.set_default_drive_type(1)         
import_config.set_default_drive_strength(1e4)   
import_config.set_default_position_drive_damping(1e3)  
import_config.set_density(30.0)                 
import_config.set_distance_scale(1.0)           

# 执行转换
success, _ = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=urdf_file_path,
    import_config=import_config,
    dest_path=usd_save_path,
)

if success:
    print(f"✅ 初始转换成功: {usd_save_path}")
    print("🔧 正在重构 USD 层级树...")
    
    # 1. 打开刚才生成的 USD 文件
    stage = Usd.Stage.Open(usd_save_path)
    
    # 2. 抓取真实的机器人根节点
    root_children = stage.GetPseudoRoot().GetChildren()
    old_root_prim = None
    
    for child in root_children:
        if child.GetName() not in ["Looks", "Render", "OmniverseKit_Persp"]:
            old_root_prim = child
            break
    
    if old_root_prim:
        old_path = old_root_prim.GetPath()
        robot_name = old_root_prim.GetName()
        print(f"👉 成功抓取到真实的机器人根节点: {old_path}")
        
        # 3. 创建全新的 /root 节点 (Xform)
        UsdGeom.Xform.Define(stage, "/root")
        
        # 4. 将原来的节点移动到 /root 下面
        new_path = Sdf.Path(f"/root/{robot_name}")
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(old_path, new_path)
        stage.GetRootLayer().Apply(edit)
        
        # ==========================================
        # 👇 新增：批量重命名 joint 节点
        # ==========================================
        print("🔍 正在扫描并重命名关节 (joint) 节点...")
        joint_edit = Sdf.BatchNamespaceEdit()
        renamed_count = 0
        
        # 遍历舞台中的所有节点 (需要在上一步 Apply 之后进行，以获取最新的层级)
        for prim in stage.Traverse():
            prim_name = prim.GetName()
            # 如果名字以 joint_ 开头 (例如 joint_0, joint_1)
            if prim_name.startswith("joint_"):
                old_joint_path = prim.GetPath()
                # 替换名称规则，生成新名字
                new_joint_name = prim_name.replace("joint_", "articulate_joint_")
                # 生成新的 Sdf 路径
                new_joint_path = old_joint_path.ReplaceName(new_joint_name)
                
                # 添加到批量修改任务中
                joint_edit.Add(old_joint_path, new_joint_path)
                renamed_count += 1
            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Gprim):
                # 创建或获取碰撞开关属性
                # physics:collisionEnabled 是 PhysX 识别的标准属性
                collision_attr = prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool)
                collision_attr.Set(False)
                
                # 为了保险，如果导入器已经加上了 CollisionAPI，可以将其移除或设置
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    # 也可以通过 API 设置
                    coll_api = UsdPhysics.CollisionAPI(prim)
                    coll_api.CreateCollisionEnabledAttr(False)
                        
        # 执行批量重命名
        if renamed_count > 0:
            stage.GetRootLayer().Apply(joint_edit)
            print(f"✅ 成功将 {renamed_count} 个关节重命名为 articulate_joint_ 格式！")
        else:
            print("⚠️ 未找到任何以 'joint_' 开头的节点，请确认 URDF 中的原始命名。")
        # ==========================================
            
        # 5. 将新建的 /root 设为整个 USD 文件的默认主节点
        stage.SetDefaultPrim(stage.GetPrimAtPath("/root"))
        
        # 6. 保存修改
        stage.GetRootLayer().Save()
        
        print(f"🎉 结构重构和命名修改完美结束！")
    else:
        print("⚠️ 整个 USD 文件里找不到任何有效的实体节点。")
else:
    print("❌ 转换失败，请检查 URDF 文件路径和 mesh 文件是否存在")
    
simulation_app.close()