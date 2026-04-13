"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""


from simple.core.controller import Controller, ControllerCfg


class BaseController(Controller):
    def __init__(self, controller_cfg):
        super().__init__(controller_cfg)


class BaseControllerCfg(ControllerCfg):
    clazz: type[Controller] = BaseController

    