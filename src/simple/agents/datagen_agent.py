from queue import Queue
import torch
import numpy as np
import transforms3d as t3d
from collections import deque

from .base_agent import BaseAgent
from .primitive_agent import PrimitiveAgent
from simple.core.task import Task
from simple.datagen.solver import TaskSolver
# from simple.datagen.solver import MotionPlannerSolver
# from simple.mp.planner import MotionPlanner

from simple.datagen.subtask_spec import (
    OpenGripperSpec,
    MoveEEFToPoseSpec,
    CloseGripperSpec,
    GraspObjectSpec
)

class DataGenAgent(PrimitiveAgent):

    def __init__(self, task:Task, solver: TaskSolver):
        super().__init__(task.robot)
        self.task = task
        self.solver = solver
        # self.max_try = 3

    def synthesize(self):
        """  Pre-generate all the solutions for the episode
        """
        # retry = 0
        # while retry < self.max_try:
        # self.solver.solve(self.task)
        # self.task.solve()

        subtasks = self.task.decompose()
        for spec in subtasks:
            if isinstance(spec, OpenGripperSpec):
                self.queue_open_gripper()
            elif isinstance(spec, CloseGripperSpec):
                self.queue_close_gripper()
            elif isinstance(spec, GraspObjectSpec):
                traj = self.solver.plan_to_grasp(spec)
                self.queue_follow_path(traj)

            elif isinstance(spec, MoveEEFToPoseSpec):
                traj = self.solver.plan_to_move_eef(spec)
                self.queue_follow_path(traj)

    def get_action(self, observation, instruction=None, **kwargs):
        raise NotImplementedError