from collections import deque
import numpy as np
from .base_agent import BaseAgent, ActionCommand
from collections import deque
import numpy as np
from .base_agent import BaseAgent, ActionCommand

class PlanAgent(BaseAgent):
    '''
    This agent is for data generation. It takes a plan containing states (approach, grasp, lift, wait), 
    and joint positions. Supports both Franka and ALOHA robots.
    '''
    def __init__(self, robot, plan, robot_type="franka", ee_link="right_gripper_site", ik_solver=None):
        """
        Args:
            robot: Robot instance
            plan: [(state, action), ...]
            robot_type: "franka" or "aloha"
            ee_link: For ALOHA, "right_gripper_site" or "left_gripper_site"
            ik_solver: IK solver instance (optional)
        """
        super().__init__(robot, ik_solver, robot_type, ee_link)
        self.state = 'rest'
        self.actions = self._load_actions(plan)

    def _load_actions(self, plan):
        """Load actions from plan based on robot type."""
        actions = deque()
        prev_arm_joints = None
        
        for state, joints in plan:
            # Determine openness based on state
            if state in ['grasp', 'lift', 'wait']:
                openness = -1.
            elif state in ['approach']:
                openness = 1.
            else:
                raise NotImplementedError(f'unknown state: {state}')
            
            # Handle different joint input formats
            if len(joints) in [0, 1, 2]:  # Empty or minimal joint info
                assert prev_arm_joints is not None, "No previous arm joints available"
                arm_joints = prev_arm_joints
            else:
                # Extract arm joints based on robot type
                if "franka" in self.robot_type:
                    arm_joints = [j[1] for j in joints[:7]]
                elif self.robot_type == "aloha":
                    arm_joints = [j[1] for j in joints[:13]]
                else:
                    # Default fallback for unknown robot types
                    joint_slice = min(len(joints), 13)  # Assume max 13 joints
                    arm_joints = [j[1] for j in joints[:joint_slice]]
            
            prev_arm_joints = arm_joints
            params = np.concatenate([arm_joints, [openness]])
            actions.append((state, ActionCommand('move_joints_with_openness', params=params)))
        
        return actions

    def _step(self):
        """Get next action command."""
        if len(self.actions) > 0:
            state, command = self.actions.popleft()
            self.state = state
            return command
        else:
            raise StopIteration()

# Backward compatibility aliases
class AlohaPlanAgent(PlanAgent):
    """Backward compatibility for ALOHA robot."""
    def __init__(self, robot, plan, ee_link="right_gripper_site"):
        super().__init__(robot, plan, robot_type="aloha", ee_link=ee_link)

class FrankaPlanAgent(PlanAgent):
    """Backward compatibility for Franka robot."""  
    def __init__(self, robot, plan):
        super().__init__(robot, plan, robot_type="franka")