# robot_state_controller.py
# RobotStateController class

import time
from enums import SystemState
from position_history import PositionHistory
from config import HAND_HOLD_TIME, ROBOT_BASE_X, ROBOT_BASE_Y, ROBOT_BASE_Z, COOLDOWN_TIME

class RobotStateController:
    """
    Five-state machine: IDLE → TRACKING → POSITION_LOCKED → EXECUTING → COOLDOWN.

    The `robot_detector` argument (RobotPositionDetector) is optional.
    When provided its live pose estimation is used for coordinate conversion;
    otherwise the fallback ROBOT_BASE_X/Y/Z constants are used.
    """

    def __init__(self, robot_manager, robot_detector=None, robot_base_position=None):
        self.state             = SystemState.IDLE
        self.state_enter_time  = time.time()
        self.robot_manager     = robot_manager
        self.robot_detector    = robot_detector   # ArUco detector (can be None)
        self.robot_base_position = robot_base_position  # Manual base position (can be None)
        self.position_history  = PositionHistory()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_state_name(self):
        return self.state.name

    def get_time_in_state(self):
        return time.time() - self.state_enter_time

    def _enter(self, new_state):
        self.state            = new_state
        self.state_enter_time = time.time()
        print(f'\n>>> STATE: {new_state.name}')

    def _camera_to_robot_coords(self, cam_x, cam_y, cam_z):
        """
        Convert triangulated camera-world coords → robot-relative coords.

        Priority:
          1. ArUco live detection  (accurate, dynamic)
          2. Manual base position   (user-provided, static)
          3. Hardcoded fallback    (ROBOT_BASE_X/Y/Z)

        Returns (rel_x, rel_y, rel_z) or (None, None, None) if ArUco
        is the chosen method but robot is not currently visible.
        """
        if self.robot_detector is not None:
            if self.robot_detector.is_detected:
                rel = self.robot_detector.get_robot_to_hand_vector(
                    (cam_x, cam_y, cam_z))
                return rel[0], rel[1], rel[2]
            else:
                # Robot markers not visible — unsafe to move
                print('[WARNING] ArUco markers not visible — movement blocked')
                return None, None, None
        elif self.robot_base_position is not None:
            # Use manual base position
            return cam_x - self.robot_base_position[0], cam_y - self.robot_base_position[1], cam_z - self.robot_base_position[2]
        else:
            # Fallback: static offsets
            return cam_x - ROBOT_BASE_X, cam_y - ROBOT_BASE_Y, cam_z - ROBOT_BASE_Z

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self, hand_detected, world_x=None, world_y=None, world_z=None):
        """Call once per frame. Returns current SystemState."""
        time_in_state = self.get_time_in_state()

        if self.state == SystemState.IDLE:
            if self.robot_manager.is_connected:
                self.robot_manager.controller.move_joints(
                    A1=0, A2=0, A3=0, A4=0, A5=0, A6=0,
                    E1=0, E2=0, E3=0, velocity=50, wait_move_finished=False
                )
            if hand_detected:
                self.position_history.clear()
                self.position_history.add(world_x, world_y, world_z, time.time())
                self._enter(SystemState.TRACKING)

        elif self.state == SystemState.TRACKING:
            if not hand_detected:
                self._enter(SystemState.IDLE)
                self.position_history.clear()
            else:
                self.position_history.add(world_x, world_y, world_z, time.time())
                if time_in_state >= HAND_HOLD_TIME and self.position_history.is_stable():
                    self._enter(SystemState.POSITION_LOCKED)

        elif self.state == SystemState.POSITION_LOCKED:
            if not hand_detected:
                self._enter(SystemState.IDLE)
                self.position_history.clear()
            else:
                avg_x, avg_y, avg_z = self.position_history.get_average()
                rel_x, rel_y, rel_z = self._camera_to_robot_coords(avg_x, avg_y, avg_z)

                if rel_x is None:
                    # ArUco not visible — go back to IDLE
                    self._enter(SystemState.IDLE)
                    self.position_history.clear()
                    return self.state

                print(f'[POSITION_LOCKED] Camera: ({avg_x:.4f}, {avg_y:.4f}, {avg_z:.4f})')
                print(f'[POSITION_LOCKED] Robot:  ({rel_x:.4f}, {rel_y:.4f}, {rel_z:.4f})')

                if self.robot_manager.move_to_target(rel_x, rel_y, rel_z, velocity=75):
                    self._enter(SystemState.EXECUTING)
                else:
                    self._enter(SystemState.IDLE)
                    self.position_history.clear()

        elif self.state == SystemState.EXECUTING:
            if time_in_state > 10.0:
                self._enter(SystemState.COOLDOWN)

        elif self.state == SystemState.COOLDOWN:
            if time_in_state > COOLDOWN_TIME:
                self._enter(SystemState.IDLE)
                self.position_history.clear()

        return self.state

print('✓ RobotStateController class defined')