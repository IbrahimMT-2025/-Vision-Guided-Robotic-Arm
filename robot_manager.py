# robot_manager.py
# RobotManager class

import time
import math

# cri_lib may not be installed in this environment
try:
    from cri_lib import CRIController
except ImportError:
    CRIController = None
    print('⚠ cri_lib not installed; robot functions will be no-ops')

from config import L1, L2, L3

class RobotManager:
    """Connect to, control, and disconnect from the CRI robot."""

    def __init__(self, ip, port):
        self.ip               = ip
        self.port             = port
        self.is_connected     = False
        self.current_position = {k: 0 for k in ['A1','A2','A3','A4','A5','A6','E1','E2','E3']}
        if CRIController is not None:
            self.controller = CRIController()
        else:
            self.controller = None

    # ── Connection lifecycle ──────────────────────────────────────────────────

    def connect(self):
        if self.controller is None:
            print('⚠ Robot controller not available; skipping connection')
            return False

        print('Connecting to robot...')
        if not self.controller.connect(self.ip, self.port):
            print('✗ Connection failed')
            return False
        print('✓ Connected')
        self.controller.set_active_control(True)
        self.controller.enable()
        self.controller.wait_for_kinematics_ready(3000)
        self.controller.set_override(100.0)
        print('Referencing all joints...')
        self.controller.reference_all_joints()
        time.sleep(3.0)
        self.controller.enable()
        self.controller.wait_for_kinematics_ready(3000)
        self.controller.set_override(100.0)
        print('✓ Robot ready\n')
        self.is_connected = True
        return True

    def disconnect(self):
        if self.is_connected and self.controller is not None:
            print('\nShutting down robot...')
            self.controller.disable()
            self.controller.close()
            self.is_connected = False
            print('✓ Done')

    # ── Motion ────────────────────────────────────────────────────────────────

    def move_to_target(self, target_x, target_y, target_z, velocity=50):
        if self.controller is None or not self.is_connected:
            print('⚠  Robot not connected or controller unavailable')
            return False
        try:
            A1, A2, A3, A4, A5, A6 = self.calculate_inverse_kinematics(
                target_x, target_y, target_z)
            if A1 is None:
                return False
            self.controller.move_joints(
                A1=A1, A2=A2, A3=A3, A4=A4, A5=A5, A6=A6,
                E1=0,  E2=0,  E3=0,
                velocity=velocity, wait_move_finished=False
            )
            self.current_position.update(
                A1=A1, A2=A2, A3=A3, A4=A4, A5=A5, A6=A6)
            print(f'  IK: A1={A1:6.1f}° A2={A2:6.1f}° A3={A3:6.1f}°')
            return True
        except Exception as e:
            print(f'⚠  Move error: {e}')
            return False

    def calculate_inverse_kinematics(self, x, y, z):
        """2-link planar IK using L1/L2/L3 from Configuration cell."""
        try:
            l2, l3  = L2, L3
            dist_xy = math.sqrt(x**2 + y**2)
            A1      = 0.0 if dist_xy < 0.001 else math.degrees(math.atan2(y, x))
            d       = math.sqrt(dist_xy**2 + z**2)

            if d > l2 + l3:
                print(f'  ⚠  Unreachable: {d:.4f} m  (max {l2+l3:.4f} m)')
                return None, None, None, None, None, None

            cos_A3  = (l2**2 + l3**2 - d**2) / (2 * l2 * l3)
            cos_A3  = max(-1.0, min(1.0, cos_A3))
            A3_rad  = math.acos(cos_A3)
            A3      = math.degrees(A3_rad)

            alpha   = math.atan2(z, dist_xy)
            beta    = math.atan2(l3 * math.sin(A3_rad), l2 + l3 * math.cos(A3_rad))
            A2      = math.degrees(alpha - beta)

            A1 = max(-90, min(90,  A1))
            A2 = max(-90, min(90,  A2))
            A3 = max(-120, min(120, A3))

            return A1, A2, A3, 0, 0, 0
        except Exception as e:
            print(f'  ⚠  IK error: {e}')
            return None, None, None, None, None, None

print('✓ RobotManager class defined')