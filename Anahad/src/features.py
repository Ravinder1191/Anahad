import math
import numpy as np

def calculate_wrist_mcp_angle(wrist_x, wrist_y, mcp_x, mcp_y):
    radians = math.atan2(wrist_y - mcp_y, mcp_x - wrist_x)
    degrees = math.degrees(radians)
    return degrees

def joint_angle(a, b, c):
    ba = a - b
    bc = c - b
    mag = np.linalg.norm(ba) * np.linalg.norm(bc)
    if mag < 1e-6:
        return 0.0
    cos_theta = np.clip(np.dot(ba, bc) / mag, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))

def fingers_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def palm_size(pts):
    size = np.linalg.norm(pts[9] - pts[0])
    return size if size > 1e-6 else 1.0

def is_palm_showing(hand_landmarks, handedness):

    p0 = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
    p5 = np.array([hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y])
    p17 = np.array([hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y])

    v1 = p5 - p0
    v2 = p17 - p0

    cp = v1[0] * v2[1] - v1[1] * v2[0]

    if handedness == "Right":
        return cp > 0
    else:
        return cp < 0

def extract_features(mean_pts):
    angles = {
        "thumb":       joint_angle(mean_pts[4],  mean_pts[3],  mean_pts[2]),
        "thumb mcp":   joint_angle(mean_pts[3],  mean_pts[2],  mean_pts[2]),
        "index":       joint_angle(mean_pts[8],  mean_pts[7],  mean_pts[6]),
        "index pip":   joint_angle(mean_pts[7],  mean_pts[6],  mean_pts[5]),
        "middle":      joint_angle(mean_pts[12], mean_pts[11], mean_pts[10]),
        "ring":        joint_angle(mean_pts[16], mean_pts[15], mean_pts[14]),
        "pinky":       joint_angle(mean_pts[20], mean_pts[19], mean_pts[18]),
        "wrist angle": calculate_wrist_mcp_angle(mean_pts[0][0], mean_pts[0][1],
                                                mean_pts[9][0], mean_pts[9][1])
    }

    dists = {
        "thumb_index": fingers_distance(mean_pts[4], mean_pts[8]),
        "index_middle": fingers_distance(mean_pts[6], mean_pts[10]),
        "middle_ring":  fingers_distance(mean_pts[10], mean_pts[14]),
        "ring_pinky":   fingers_distance(mean_pts[14], mean_pts[18]),
    }
    wrist_vec = mean_pts[9] - mean_pts[0]
    wrist_angle = math.degrees(math.atan2(wrist_vec[1], wrist_vec[0]))

    return angles, dists, wrist_angle
