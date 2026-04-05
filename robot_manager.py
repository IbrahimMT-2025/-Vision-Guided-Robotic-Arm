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

from config import L1, L2, L3, L4, JOINT_LIMITS

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
        """
        Move end-effector to (target_x, target_y, target_z) in robot frame [metres].
        target_z < 0 means in front of robot (toward camera).
        """
        if self.controller is None or not self.is_connected:
            print('⚠  Robot not connected or controller unavailable')
            return False
        try:
            angles = self.calculate_inverse_kinematics(target_x, target_y, target_z)
            if angles is None or angles[0] is None:
                return False
            
            # Verify FK before sending move command (safety check)
            if not self.forward_kinematics_check(angles, target_x, target_y, target_z):
                print('  ⚠  FK verification failed — aborting for safety')
                return False
            
            A1, A2, A3, A4, A5, A6 = angles
            self.controller.move_joints(
                A1=A1, A2=A2, A3=A3, A4=A4, A5=A5, A6=A6,
                E1=0,  E2=0,  E3=0,
                velocity=velocity, wait_move_finished=False
            )
            self.current_position.update(
                A1=A1, A2=A2, A3=A3, A4=A4, A5=A5, A6=A6)
            print(f'  ✓ Move sent  A1={A1:.1f}°  A2={A2:.1f}°  A3={A3:.1f}°  '
                  f'A4={A4:.1f}°  A5={A5:.1f}°  A6={A6:.1f}°')
            return True
        except Exception as e:
            print(f'⚠  Move error: {e}')
            return False

    def calculate_inverse_kinematics(self, rx, ry, rz):
        """
        Full 6-DOF IK for the CRI 6-axis robot arm.

        Inputs (robot frame, metres):
          rx : + to the right   (same axis as camera X)
          ry : + upward         (same axis as camera Y)
          rz : + away from camera; NEGATIVE means "in front of robot"

        Arm plane convention:
          • A1 rotates around the vertical Y axis.
            A1 = atan2(wx, −wz)  →  0 when arm is along −Z (facing camera).
          • A2 is shoulder pitch from VERTICAL   (0 = arm straight UP, +° tilts toward camera).
          • A3 is elbow angle from A2 direction (standard 2-link IK).
          • A5 is wrist pitch so the EE points along the base-to-target direction.
          • A4 = A6 = 0  (roll axes — no effect on position).

        Returns tuple (A1, A2, A3, A4, A5, A6) in degrees or (None, None, ...) if unreachable.
        """
        try:
            print(f'\n  IK target  rx={rx:.3f}  ry={ry:.3f}  rz={rz:.3f}')

            # ── 1. Wrist centre (offset from target by L4 toward base) ───────
            dist = math.sqrt(rx**2 + ry**2 + rz**2)
            if dist < 0.01:
                print('  IK: target too close to base')
                return None, None, None, None, None, None

            ux, uy, uz = rx / dist, ry / dist, rz / dist   # unit vec: base → target
            wx = rx - L4 * ux
            wy = ry - L4 * uy
            wz = rz - L4 * uz

            # ── 2. A1 — base rotation around vertical Y axis ──────────────────
            #   atan2(wx, −wz) :  0 when wrist is along −Z (in front)
            #                    +° when wrist is to the +X side (anticlockwise from above)
            r_horiz = math.sqrt(wx**2 + wz**2)
            A1_rad = math.atan2(-wx, wz) if r_horiz > 1e-4 else 0.0
            A1      = math.degrees(A1_rad)
            
            # ── 3. Arm-plane coordinates (vertical plane through reach direction) ─
            #   Shoulder (A2) is on the A1 axis at height L1 above the base.
            #   dr = horizontal distance from shoulder to wrist in arm plane
            #   dy = vertical distance from shoulder to wrist
            dr = r_horiz         # wrist's horizontal distance from A1 axis
            dy = wy - L1        # wrist height above shoulder

            d_sw = math.sqrt(dr**2 + dy**2)   # shoulder → wrist distance
            print(f'  IK: A1={A1:.1f}°  r_horiz={r_horiz:.3f}  '
                  f'd_shoulder_wrist={d_sw:.3f}  max={L2+L3:.3f}')

            if d_sw > L2 + L3:
                print(f'  IK fail: wrist out of reach ({d_sw:.3f} > {L2+L3:.3f})')
                return None, None, None, None, None, None
            if d_sw < abs(L2 - L3):
                print(f'  IK fail: wrist too close ({d_sw:.3f} < {abs(L2-L3):.3f})')
                return None, None, None, None, None, None

            # ── 4. Standard 2-link IK for A3 and A2 ──────────────────────────
            cos_q3 = (dr**2 + dy**2 - L2**2 - L3**2) / (2.0 * L2 * L3)
            cos_q3 = max(-1.0, min(1.0, cos_q3))

            best = None
            for sign in (1, -1):          # +1 = elbow-up first, −1 = elbow-down
                sin_q3 = sign * math.sqrt(max(0.0, 1.0 - cos_q3**2))
                A3_rad = math.atan2(sin_q3, cos_q3)

                # gamma: angle from VERTICAL to the shoulder→wrist direction.
                # atan2(horizontal, vertical) gives angle-from-vertical (not from horizontal).
                #   wrist straight up   → gamma=0   → A2=0  ✓
                #   wrist horizontal    → gamma=90° → A2=90° ✓
                gamma  = math.atan2(dr, dy)                            # angle from vertical
                delta  = math.atan2(L3 * sin_q3, L2 + L3 * cos_q3)   # triangle offset
                A2_rad = gamma - delta

                A2 = math.degrees(A2_rad)
                A3 = math.degrees(A3_rad)

                # ── 5. A5 — wrist pitch to align EE toward target ─────────────
                #   L4 must point from wrist → target.
                #   Compute wrist position in arm plane (from shoulder), then
                #   find the wrist→target direction angle from vertical.
                r_wrist = L2*math.sin(A2_rad) + L3*math.sin(A2_rad + A3_rad)
                v_wrist = L2*math.cos(A2_rad) + L3*math.cos(A2_rad + A3_rad)
                r_h    = math.sqrt(rx**2 + rz**2)   # target horizontal reach
                d_r    = r_h    - r_wrist             # horizontal step wrist→target
                d_v    = (ry - L1) - v_wrist          # vertical   step wrist→target
                A5_rad = math.atan2(d_r, d_v) - (A2_rad + A3_rad)
                A5 = math.degrees(A5_rad)

                A4 = 0.0   # forearm roll — does not affect position
                A6 = 0.0   # end-effector roll

                # ── 6. Joint limit check ──────────────────────────────────────
                in_limits = (
                    JOINT_LIMITS['A1'][0] <= A1 <= JOINT_LIMITS['A1'][1] and
                    JOINT_LIMITS['A2'][0] <= A2 <= JOINT_LIMITS['A2'][1] and
                    JOINT_LIMITS['A3'][0] <= A3 <= JOINT_LIMITS['A3'][1] and
                    JOINT_LIMITS['A5'][0] <= A5 <= JOINT_LIMITS['A5'][1]
                )
                if in_limits:
                    best = [A1, A2, A3, A4, A5, A6]
                    break

            if best is None:
                last_vals = f"A1={A1:.1f} A2={A2:.1f} A3={A3:.1f} A5={A5:.1f}"
                print(f'  IK fail: joint limits exceeded  ({last_vals})')
                return None, None, None, None, None, None

            print(f'  IK ✓  A1={best[0]:.1f}°  A2={best[1]:.1f}°  A3={best[2]:.1f}°  '
                  f'A5={best[4]:.1f}°')
            return tuple(best)

        except Exception as e:
            print(f'  IK error: {e}')
            return None, None, None, None, None, None

    # ── Forward kinematics ────────────────────────────────────────────────────
    
    def forward_kinematics(self, A1_d, A2_d, A3_d, A5_d):
        """
        Returns end-effector (x, y, z) in robot frame given joint angles.

        Convention matching calculate_inverse_kinematics above:
          A1  rotation around Y,    0 → arm along −Z (toward camera)
          A2  shoulder pitch,       0 → arm VERTICAL (straight up)
          A3  elbow relative angle, 0 → arm straight
          A5  wrist pitch relative to A3
          (A4, A6 are rolls — no effect on EE position)
        """
        A1 = math.radians(A1_d)
        A2 = math.radians(A2_d)
        A3 = math.radians(A3_d)
        A5 = math.radians(A5_d)

        # Horizontal unit vector of arm plane in XZ after A1 rotation:
        #   at A1=0  →  (0, 0, −1)  i.e. toward camera
        hr_x = -math.sin(A1)
        hr_z =  math.cos(A1)

        # Shoulder is on the A1 axis at height L1; not offset horizontally.
        sh_y = L1

        # A2=0 means arm is VERTICAL (straight up).
        # sin(A2) gives horizontal reach from vertical; cos(A2) gives vertical rise.
        #   A2=0:   r=0, v=L2  → arm points straight up          ✓
        #   A2=90°: r=L2, v=0  → arm is horizontal               ✓
        r = (L2 * math.sin(A2) +
             L3 * math.sin(A2 + A3) +
             L4 * math.sin(A2 + A3 + A5))

        v = (L2 * math.cos(A2) +
             L3 * math.cos(A2 + A3) +
             L4 * math.cos(A2 + A3 + A5))

        # Convert to 3-D robot frame
        ee_x = r * hr_x
        ee_y = sh_y + v
        ee_z = r * hr_z
        return ee_x, ee_y, ee_z

    def forward_kinematics_check(self, angles, tx, ty, tz, tol=0.05):
        """Returns True if FK of the IK solution is within tol metres of target."""
        A1, A2, A3, _, A5, _ = angles
        fx, fy, fz = self.forward_kinematics(A1, A2, A3, A5)
        err = math.sqrt((fx - tx)**2 + (fy - ty)**2 + (fz - tz)**2)
        print(f'  FK check: target({tx:.3f},{ty:.3f},{tz:.3f}) '
              f'→ FK({fx:.3f},{fy:.3f},{fz:.3f})  err={err:.4f}')
        return err < tol

print('✓ RobotManager class defined')