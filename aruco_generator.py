# aruco_generator.py
# Generate ArUco markers for robot base

import cv2.aruco as aruco
import os

def generate_robot_markers(output_dir='.'):
    """
    Generate 4 ArUco markers (IDs 0-3) for the robot base faces.
    Saves PNG files ready for printing at 10 cm x 10 cm.
    """
    aruco_dict       = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    marker_size_px   = 400
    face_labels      = ['Front', 'Right', 'Back', 'Left']

    for marker_id in range(4):
        img = aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)

        # White border so the detector can see all four edges
        bordered = cv2.copyMakeBorder(
            img,
            top=40, bottom=40, left=40, right=40,
            borderType=cv2.BORDER_CONSTANT,
            value=255
        )

        filename = os.path.join(output_dir, f'robot_marker_face_{marker_id}.png')
        cv2.imwrite(filename, bordered)
        print(f'  ✓ Saved {filename}  →  attach to {face_labels[marker_id]} face')

    print('\n⚠  Print each file at EXACTLY 10 cm x 10 cm.')
    print('   After attaching, verify MARKER_OFFSETS in config.py match physical placement.')

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_robot_markers()