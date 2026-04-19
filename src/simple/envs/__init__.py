"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from enum import Enum
class SIM_MODE(Enum):
    MUJOCO = "mujoco"
    ISAAC = "isaac"
    MUJOCO_ISAAC = "mujoco_isaac"

from gymnasium.envs.registration import register

register(
    id="simple/FrankaTabletopGraspMP-v0",
    entry_point="simple.envs.tabletop_grasp:TabletopGraspEnv",
    kwargs={"task":"franka_tabletop_grasp_mp"},
)
register(
    id="simple/FrankaTabletopPickNPlaceMP-v0",
    entry_point="simple.envs.tabletop_grasp:TabletopGraspEnv",
    kwargs={"task":"franka_tabletop_pick_n_place_mp"},
)
register(
    id="simple/AlohaTabletopGraspMP-v0",
    entry_point="simple.envs.tabletop_grasp:TabletopGraspEnv",
    kwargs={"task":"aloha_tabletop_grasp_mp"},
)
register(
    id="simple/AlohaTabletopHandoverMP-v0",
    entry_point="simple.envs.tabletop_grasp:TabletopGraspEnv",
    kwargs={"task":"aloha_tabletop_handover_mp"},
)
register(
    id="simple/VegaTabletopGraspMP-v0",
    entry_point="simple.envs.tabletop_grasp:TabletopGraspEnv",
    kwargs={"task":"vega_tabletop_grasp_mp"},
)
register(
    id="simple/AlohaTabletopFindNGraspMP-v0",
    entry_point="simple.envs.tabletop_grasp:TabletopGraspEnv",
    kwargs={"task":"aloha_tabletop_find_n_grasp_mp"},
)
register(
    id="simple/VegaTabletopFindNGraspMP-v0",
    entry_point="simple.envs.tabletop_grasp:TabletopGraspEnv",
    kwargs={"task":"vega_tabletop_find_n_grasp_mp"},
)

register(
    id="simple/G1TabletopGraspMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_tabletop_grasp_mp"},
)
register(
    id="simple/G1TabletopPickNPlaceMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_tabletop_pick_n_place_mp"},

)

register(
    id="simple/G1InspireTabletopGraspMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_inspire_tabletop_grasp_mp"},
)
register(
    id="simple/G1TabletopHandoverMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_tabletop_handover_mp"},
)

register(
    id="simple/G1WholebodyLocomotionMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_locomotion_mp"},
)
register(
    id="simple/G1WholebodyPickNPlaceMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_pick_n_place_mp"},
)

register(
    id="simple/G1WholebodyLocomotionPickBetweenTablesMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_locomotion_pick_between_tables_mp"},
)
register(
    id="simple/G1WholebodySitMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_sit_mp"},
)
register(
    id="simple/G1WholebodyBendPickAndPlaceMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_bend_pick_and_place_mp"},
)
register(
    id="simple/G1WholebodyBendPickMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_bend_pick_mp"},
)
register(
    id="simple/G1WholebodyBendHandoverMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_bend_handover_mp"},
)
register(
    id="simple/G1WholebodyBendPickAndPlaceOnSofaMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_bend_pick_and_place_on_sofa_mp"},
)
register(
    id="simple/G1WholebodyTabletopGraspMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_tabletop_grasp_mp"},
)
register(
    id="simple/G1InspireWholebodyLocomotionMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_inspire_wholebody_locomotion_mp"},
)
register(
    id="simple/G1InspireWholebodyPickNPlaceMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_inspire_wholebody_pick_n_place_mp"},
)
register(
    id="simple/G1WholebodyLocomotionPickBetweenTablesVariant1MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_locomotion_pick_between_tables_variant1_mp"},
)
register(
    id="simple/G1WholebodyLocomotionPickBetweenTablesVariant2MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_locomotion_pick_between_tables_variant2_mp"},
)

register(
    id="simple/G1WholebodyPickNPlaceVariant1MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_pick_n_place_variant1_mp"},
)
register(
    id="simple/G1WholebodyLocomotionPickBetweenTablesVariant3MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_locomotion_pick_between_tables_variant3_mp"},
)

register(
    id="simple/G1WholebodyTurnPickMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_turn_pick_mp"},
)
register(
    id="simple/G1WholebodyLocomotionPickBetweenTablesVariant4MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_locomotion_pick_between_tables_variant4_mp"},
)

register(
    id="simple/G1WholebodyXMoveAndPickMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_x_move_and_pick_mp"},
)

register(
    id="simple/G1WholebodyXMoveAndPickNPlaceMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_x_move_and_pick_n_place_mp"},
)

register(
    id="simple/G1WholebodyXMoveBendPickMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_x_move_bend_pick_mp"},
)

register(

    id="simple/G1WholebodyXMoveBendPickNPlaceMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_x_move_bend_pick_n_place_mp"},
)
register(
    id="simple/G1WholebodyYMoveAndPickMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_y_move_and_pick_mp"},
)
register(
    id="simple/G1WholebodyXMoveAndPickVariant1MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_x_move_and_pick_variant1_mp"},
)
register(
    id="simple/G1WholebodyLocomotionPickBetweenTablesVariant5MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_locomotion_pick_between_tables_variant5_mp"},
)

register(
    id="simple/G1WholebodyXMoveAndHandoverMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_x_move_and_handover_mp"},
)
register(
    id="simple/G1WholebodyLocomotionPickBetweenTablesVariant6MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_locomotion_pick_between_tables_variant6_mp"},
)

register(
    id="simple/G1WholebodyYMoveAndHandoverMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_y_move_and_handover_mp"},
)

register(
    id="simple/G1WholebodyYMoveBendPickMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_y_move_bend_pick_mp"},
)

register(
    id="simple/G1WholebodyPickAndBendPlaceMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_pick_and_bend_place_mp"},
)

register(
    id="simple/G1WholebodyLocomotionPickBetweenTablesVariant7MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_locomotion_pick_between_tables_variant7_mp"},
)
register(
    id="simple/G1WholebodyXMoveAndPickVariant2MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_x_move_and_pick_variant2_mp"},
)
register(
    id="simple/G1WholebodyLocomotionPickBetweenTablesVariant8MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_locomotion_pick_between_tables_variant8_mp"},
)
register(
    id="simple/G1WholebodyBendPickVariant1MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_bend_pick_variant1_mp"},
)
register(
    id="simple/G1WholebodyLocomotionPickBetweenTablesVariant9MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_locomotion_pick_between_tables_variant9_mp"},
)
register(
    id="simple/G1WholebodyTurnPickVariant1MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_turn_pick_variant1_mp"},
)
register(
    id="simple/G1WholebodyTabletopGraspVariant1MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_tabletop_grasp_variant1_mp"},
)
register(
    id="simple/G1WholebodyBendPickAndPlaceOnSofaVariant1MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_bend_pick_and_place_on_sofa_variant1_mp"},
)

register(
    id="simple/G1WholebodyBendPickAndPlaceOnSofaVariant2MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_bend_pick_and_place_on_sofa_variant2_mp"},
)
register(
    id="simple/G1WholebodyYMoveAndPickVariant1MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_y_move_and_pick_variant1_mp"},
)
register(
    id="simple/G1WholebodyLocomotionPickBetweenTablesVariant10MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_locomotion_pick_between_tables_variant10_mp"},
)
register(
    id="simple/G1WholebodyLocomotionPickBetweenTablesVariant11MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_locomotion_pick_between_tables_variant11_mp"},
)
register(
    id="simple/G1WholebodyXMoveAndPickVariant3MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_x_move_and_pick_variant3_mp"},
)
register(
    id="simple/G1WholebodyLocomotionPickBetweenTablesVariant12MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_locomotion_pick_between_tables_variant12_mp"},
)
register(
    id="simple/G1WholebodyXMoveAndHandoverVariant1MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_x_move_and_handover_variant1_mp"},
)
register(
    id="simple/G1WholebodyYMoveAndPickVariant2MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_y_move_and_pick_variant2_mp"},
)
register(
    id="simple/G1WholebodyTabletopGraspVariant2MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_tabletop_grasp_variant2_mp"},
)
register(
    id="simple/G1WholebodyTurnXMoveAndPickMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_turn_x_move_and_pick_mp"},
)
register(
    id="simple/G1WholebodyTurnXMoveAndPickVariant1MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_turn_x_move_and_pick_variant1_mp"},
)
register(
    id="simple/G1WholebodyTurnXMoveAndPickVariant2MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_turn_x_move_and_pick_variant2_mp"},
)
register(
    id="simple/G1WholebodyXMoveBendHandoverMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_x_move_bend_handover_mp"},
)
register(
    id="simple/G1WholebodyTurnYMoveAndPickMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_turn_y_move_and_pick_mp"},
)
register(
    id="simple/G1WholebodyTurnYMoveAndPickVariant1MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_turn_y_move_and_pick_variant1_mp"},
)
register(
    id="simple/G1WholebodyTurnXMoveAndHandoverMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_turn_x_move_and_handover_mp"},
)

register(
    id="simple/G1WholebodyTurnXMoveAndHandoverVariant1MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_turn_x_move_and_handover_variant1_mp"},
)
register(
    id="simple/G1WholebodyTurnXMoveAndBendPickMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_turn_x_move_and_bend_pick_mp"},
)
register(
    id="simple/G1WholebodyTurnYMoveAndBendPickMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_turn_y_move_and_bend_pick_mp"},   
)
register(
    id="simple/G1WholebodyTurnXMoveAndBendHandoverMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_turn_x_move_and_bend_handover_mp"},
)
register(
    id="simple/G1WholebodyYMoveAndHandoverVariant1MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_y_move_and_handover_variant1_mp"},
)
register(
    id="simple/G1WholebodyTabletopGraspVariant3MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_tabletop_grasp_variant3_mp"},
)
register(
    id="simple/G1WholebodyTabletopHandoverMP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_tabletop_handover_mp"},
)
register(
    id="simple/G1WholebodyLocomotionPickBetweenTablesVariant13MP-v0",
    entry_point="simple.envs.loco_manipulation:LocoManipulationEnv",
    kwargs={"task":"g1_wholebody_locomotion_pick_between_tables_variant13_mp"},
)




register(
    id="simple/G1WholebodyXMovePickTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_xmove_pick_teleop"},
)

register(
    id="simple/G1WholebodyXMoveBendPickTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_xmove_bend_pick_teleop"},
)

register(
    id="simple/G1WholebodyPickAndPlaceAndHugContainerTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_pick_and_place_and_hug_container_teleop"},
)
register(
    id="simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_locomotion_pick_between_tables_teleop"},
)
register(
    id="simple/G1WholebodyHandoverTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_handover_teleop"},
)

register(
    id="simple/G1WholebodyCloseDoorTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_close_door_teleop"},
)
register(
    id="simple/G1WholebodyOpenOvenTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_open_oven_teleop"},    
)
register(
    id="simple/G1WholebodyOpenFaucetTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_open_faucet_teleop"},
)
register(
    id="simple/G1WholebodyPushOfficeChairTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_push_office_chair_teleop"},
)
register(
    id="simple/G1WholebodyOpenTrashCanTeleop-v0",
    entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
    kwargs={"task":"g1_wholebody_open_trash_can_teleop"},
)


# def task_register(task):
#     register(
#         id="FrankaTabletopGrasp-v0",
#         entry_point="simple.envs.tabletop_grasp:TabletopGraspEnv",
#     )


# task_register()
from .base_dual_env import BaseDualSim
from .tabletop_grasp import TabletopGraspEnv
from .loco_manipulation import LocoManipulationEnv
