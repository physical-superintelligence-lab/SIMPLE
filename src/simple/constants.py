"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

# GraspNet_1B_Object_Names = {
#     0: "cracker box",
#     1: "sugar box",
#     2: "tomato soup can",
#     3: "mustard bottle",
#     4: "potted meat can",
#     5: "banana",
#     6: "bowl",
#     7: "mug", # "red mug", # 
#     8: "power drill",
#     9: "scissors",
#     10: "chips can", # "red chips can", #
#     11: "strawberry", 
#     12: "apple",
#     13: "lemon",
#     14: "peach",
#     15: "pear",
#     16: "orange",
#     17: "plum",
#     18: "knife", 
#     19: "blue screwdriver", #
#     20: "red screwdriver", #
#     21: "racquetball", 
#     22: "blue cup", #
#     23: "yellow cup", #
#     24: "airplane", # "toy airplane", 
#     25: "toy gun",  # 
#     26: "blue toy part", # workpiece
#     27: "metal screw", # 
#     28: "yellow propeller", # "yellow propeller", # 
#     29: "blue toy part a", #
#     30: "blue toy part b", #
#     31: "yellow toy part", # 
#     32: "padlock",
#     33: "toy dragon", # 
#     34: "small green bottle", # 
#     35: "cleansing foam",
#     36: "dabao wash soup",
#     37: "mouth rinse",
#     38: "dabao sod",
#     39: "soap box",
#     40: "kispa cleanser",
#     41: "darlie toothpaste",
#     42: "men oil control",
#     43: "marker",
#     44: "hosjam toothpaste",
#     45: "pitcher cap",
#     46: "green dish",
#     47: "white mouse",
#     48: "toy model", # 
#     49: "toy deer", # 
#     50: "toy zebra", # 
#     51: "toy large elephant", # 
#     52: "toy rhinocero", #
#     53: "toy small elephant", #
#     54: "toy monkey", #
#     55: "toy giraffe", #
#     56: "toy gorilla", #
#     57: "yellow snack box", #
#     58: "toothpaste box", #
#     59: "soap", 
#     60: "mouse", 
#     61: "dabao facewash", 
#     62: "pantene facewash", # "pantene facewash", #
#     63: "head shoulders supreme",
#     64: "thera med",
#     65: "dove", 
#     66: "head shoulder care",
#     67: "toy lion", # 
#     68: "coconut juice box", 
#     69: "toy hippo", # 
#     70: "tape",
#     71: "rubiks cube", 
#     72: "peeler cover",
#     73: "peeler",
#     74: "ice cube mould"
# }

Franka_Init_QPos = [0, -1.3, 0, -2.5, 0, 1, 0, 0.04, 0.04]
# FRANKA_REST_TRANS = [ 1.0939e-01, -5.2536e-12,  5.8483e-01]
# FRANKA_REST_QUAT = [0.0382, -0.9193, -0.3808,  0.0922]

# Franka_Init_QPos_LOWER = [0, -0.5585, 0, -2.3038, 0, 1.6580, 0, 0.04, 0.04]
# FRANKA_REST_LOWER_TRANS = [ 3.7282e-01, -4.8804e-12,  5.3904e-01]
# FRANKA_REST_LOWER_QUAT = [ 0.0167, -0.9230, -0.3823,  0.0403]

FRANKA_FINGER_LENGTH = 0.126 # 0.1034 # # FIX_ROBOT

from enum import IntEnum
class GripperAction(IntEnum):
    open = 1
    close = 0
    keep = -1

class GripperState(IntEnum):
    closed = 0
    open = 1
    opening = 2
    closing = 3


Supported_HSSD_Scenes = [
    {}
]