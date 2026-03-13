# enums.py
# System state enum

from enum import Enum

class SystemState(Enum):
    """Operating states of the robot control state machine."""
    IDLE            = 1   # Waiting for a hand to appear
    TRACKING        = 2   # Hand detected, accumulating position samples
    POSITION_LOCKED = 3   # Position stable — ready to send IK command
    EXECUTING       = 4   # Robot is moving
    COOLDOWN        = 5   # Waiting before accepting next command

print('[OK] SystemState enum defined')