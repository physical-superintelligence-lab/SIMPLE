"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import os

import numpy as np

from simple.agents.primitive_agent import PrimitiveAgent
from simple.baselines.client import HttpActionClient
from simple.core.action import ActionCmd


class ReplayPolicyAgent(PrimitiveAgent):
    def __init__(self, robot, host: str, port: int, **kwargs):
        super().__init__(robot, **kwargs)
        self.client = HttpActionClient(host, port)
        self._global_step_idx = 0
        self._session_idx = 0
        self._session_id = self._make_session_id()
        self._reset_history = True

    def _make_session_id(self) -> str:
        return f"replay-policy-{os.getpid()}-{self._session_idx}"

    def get_action(
        self,
        observation,
        instruction=None,
        info=None,
        conditions=None,
        **kwargs,
    ):
        self._last_observation = observation
        self._last_qpos = observation.get("joint_qpos")

        if len(self._action_queue) == 0:
            history = {
                "session_id": self._session_id,
                "episode_index": int(info.get("episode_index", -1)) if info is not None else -1,
                "step_index": int(self._global_step_idx),
            }
            if self._reset_history:
                history["reset"] = True
                self._reset_history = False

            pred_actions, *_ = self.client.query_action(
                image_dict={},
                instruction=instruction or "replay recorded actuator actions",
                state_dict={"joint_qpos": observation.get("joint_qpos")},
                condition_dict=conditions or {},
                history=history,
                dataset="simple_replay",
            )

            pred_actions = np.asarray(pred_actions, dtype=np.float32)
            if pred_actions.size == 0:
                raise StopIteration("Replay server returned no more recorded actions.")
            if pred_actions.ndim == 1:
                pred_actions = pred_actions[None, :]

            expected_dim = len(self.robot.joint_names)
            if pred_actions.shape[1] != expected_dim:
                raise ValueError(
                    f"Expected replay action dim {expected_dim}, got {pred_actions.shape[1]}"
                )

            for action_vec in pred_actions:
                self.queue_action(
                    ActionCmd(
                        "replay_move_actuators",
                        target_qpos=dict(zip(self.robot.joint_names, action_vec, strict=True)),
                    )
                )

        self._last_pred_action = super().get_action(observation, instruction, **kwargs)
        self._global_step_idx += 1
        return self._last_pred_action

    def reset(self):
        super().reset()
        self._global_step_idx = 0
        self._session_idx += 1
        self._session_id = self._make_session_id()
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None
        self._reset_history = True
