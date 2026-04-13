"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import os
import tempfile
import gymnasium as gym
import zmq

from gymnasium import spaces
from typing import Any, Dict, Optional
import mujoco
import numpy as np
import xml.etree.ElementTree as ET
import transforms3d as t3d

from simple.core.task import Task
from simple.envs.base_dual_env import BaseDualSim
from simple.robots.protocols import Humanoid
from threading import Lock, Thread

import tyro
from simple.envs.base_dual_env import BaseDualSim
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.data.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from gear_sonic.utils.mujoco_sim.simulator_factory import init_channel
from gear_sonic.utils.mujoco_sim.robot import Robot as SonicRobot
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge

from simple.teleop.pico.tcp_server import TCPControlServer
from simple.teleop.pico.tcp_video_sender import TCPVideoSender
from simple.teleop.pico.streaming import FrameBuffer, StreamingThread

from .wbc_agent import WholeBodyControlAgent
from simple.robots.g1_sonic import G1Sonic

class SonicWbcAgent(WholeBodyControlAgent):
    
    def __init__(self, robot: G1Sonic):
        self.robot = robot

        # # adapted from SimWrapper.__init__
        # try:
        #     init_channel(config=robot.sonic_config)
        # except Exception as e:
        #     print(f"Note: Channel factory initialization attempt: {e}")

    