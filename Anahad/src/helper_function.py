import cv2
import numpy as np
import mediapipe as mp
from scipy.interpolate import interp1d


mp_holistic = mp.solutions.holistic

BASE_FEATURE_SIZE  = 150
TOTAL_FEATURE_SIZE = 300

BODY_ORDER = [
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_hip",      "right_hip",
    "nose"
]


def init_mediapipe():
    return mp_holistic.Holistic(model_complexity=0, smooth_landmarks=True)


def get_hand_points(result):
    zero_hand = np.zeros((21, 3), dtype=np.float32)

    right = (np.array([[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks.landmark],
                       dtype=np.float32)
             if result.right_hand_landmarks else zero_hand.copy())

    left  = (np.array([[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks.landmark],
                       dtype=np.float32)
             if result.left_hand_landmarks else zero_hand.copy())

    return right, left


def get_pose_points(result):
    if not result.pose_landmarks:
        return None

    lm = result.pose_landmarks.landmark

    if lm[11].visibility < 0.4 and lm[12].visibility < 0.4:
        return None

    return {
        "left_shoulder":  np.array([lm[11].x, lm[11].y, lm[11].z]),
        "right_shoulder": np.array([lm[12].x, lm[12].y, lm[12].z]),
        "left_elbow":     np.array([lm[13].x, lm[13].y, lm[13].z]),
        "right_elbow":    np.array([lm[14].x, lm[14].y, lm[14].z]),
        "left_hip":       np.array([lm[23].x, lm[23].y, lm[23].z]),
        "right_hip":      np.array([lm[24].x, lm[24].y, lm[24].z]),
        "nose":           np.array([lm[0].x,  lm[0].y,  lm[0].z]),
    }


def normalize_hand(hand_pts):
    if not np.any(hand_pts):
        return hand_pts

    hand_pts = hand_pts - hand_pts[0]
    scale    = np.linalg.norm(hand_pts[9][:2] - hand_pts[0][:2])
    scale    = scale if scale > 1e-6 else 1.0

    return hand_pts / scale


def normalize_body(pose_pts):
    left_sh  = pose_pts["left_shoulder"]
    right_sh = pose_pts["right_shoulder"]

    center = (left_sh + right_sh) / 2
    scale  = np.linalg.norm((left_sh - right_sh)[:2])
    scale  = scale if scale > 1e-6 else 1.0

    for key in pose_pts:
        pose_pts[key] = (pose_pts[key] - center) / scale

    return pose_pts


def extract_features(right_pts, left_pts, body_pts):
    features = []

    features.extend(right_pts.flatten())
    features.extend(left_pts.flatten())

    if body_pts is not None:
        for key in BODY_ORDER:
            features.extend(body_pts[key])
    else:
        features.extend(np.zeros(len(BODY_ORDER) * 3))

    right_active = np.any(right_pts)
    left_active  = np.any(left_pts)

    if right_active and left_active:
        hand_center = (np.mean(right_pts, axis=0) + np.mean(left_pts, axis=0)) / 2
    elif right_active:
        hand_center = np.mean(right_pts, axis=0)
    elif left_active:
        hand_center = np.mean(left_pts, axis=0)
    else:
        hand_center = np.zeros(3)

    if body_pts is not None:
        shoulder_center = (body_pts["left_shoulder"] + body_pts["right_shoulder"]) / 2
        hip_center      = (body_pts["left_hip"]      + body_pts["right_hip"])      / 2
        nose            =  body_pts["nose"]

        features.extend([
            np.linalg.norm(hand_center - shoulder_center),
            np.linalg.norm(hand_center - hip_center),
            np.linalg.norm(hand_center - nose),
        ])
    else:
        features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


def add_motion(sequence):
    sequence   = np.array(sequence)
    motion     = np.zeros_like(sequence)
    motion[1:] = sequence[1:] - sequence[:-1]

    return np.concatenate([sequence, motion], axis=1)


def process_frame(frame, holistic_or_result, last_valid, precomputed=False):
    if precomputed:
        result = holistic_or_result
    else:
        # original behaviour — used when called standalone
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic_or_result.process(rgb)

    if result.right_hand_landmarks and result.left_hand_landmarks:
        handedness = "Both"
    elif result.right_hand_landmarks:
        handedness = "Right"
    elif result.left_hand_landmarks:
        handedness = "Left"
    else:
        handedness = None

    right_pts, left_pts = get_hand_points(result)
    pose_pts            = get_pose_points(result)

    if result.right_hand_landmarks or result.left_hand_landmarks:
        right_pts = normalize_hand(right_pts)
        left_pts  = normalize_hand(left_pts)
        if pose_pts is not None:
            pose_pts = normalize_body(pose_pts)

        features   = extract_features(right_pts, left_pts, pose_pts)
        last_valid = features
        return features, last_valid, handedness

    return last_valid, last_valid, handedness


def normalize_length(seq, target_len=20):
    base  = seq[:, :BASE_FEATURE_SIZE]
    x_old = np.linspace(0, 1, base.shape[0])
    x_new = np.linspace(0, 1, target_len)
    base  = interp1d(x_old, base, axis=0, kind='linear')(x_new).astype(np.float32)

    motion     = np.zeros_like(base)
    motion[1:] = base[1:] - base[:-1]

    return np.concatenate([base, motion], axis=1)


def flip_sequence(seq):
    seq  = seq.copy()
    base = seq[:, :BASE_FEATURE_SIZE].copy()

    base[:, 0:63:3]   = 1 - base[:, 0:63:3]
    base[:, 63:126:3] = 1 - base[:, 63:126:3]

    right_block     = base[:, 0:63].copy()
    left_block      = base[:, 63:126].copy()
    base[:, 0:63]   = left_block
    base[:, 63:126] = right_block

    right_pts  = base[:, 0:63].reshape(-1, 21, 3)
    left_pts   = base[:, 63:126].reshape(-1, 21, 3)
    pose_block = base[:, 126:147].reshape(-1, 7, 3)

    right_active = np.any(right_pts != 0, axis=(1, 2))
    left_active  = np.any(left_pts  != 0, axis=(1, 2))

    hand_center = np.zeros((base.shape[0], 3), dtype=np.float32)
    for i in range(base.shape[0]):
        if right_active[i] and left_active[i]:
            hand_center[i] = (np.mean(right_pts[i], axis=0) + np.mean(left_pts[i], axis=0)) / 2
        elif right_active[i]:
            hand_center[i] = np.mean(right_pts[i], axis=0)
        elif left_active[i]:
            hand_center[i] = np.mean(left_pts[i],  axis=0)

    shoulder_center = (pose_block[:, 0, :] + pose_block[:, 1, :]) / 2
    hip_center      = (pose_block[:, 4, :] + pose_block[:, 5, :]) / 2
    nose            =  pose_block[:, 6, :]

    base[:, 147] = np.linalg.norm(hand_center - shoulder_center, axis=1)
    base[:, 148] = np.linalg.norm(hand_center - hip_center,      axis=1)
    base[:, 149] = np.linalg.norm(hand_center - nose,            axis=1)

    motion     = np.zeros_like(base)
    motion[1:] = base[1:] - base[:-1]

    return np.concatenate([base, motion], axis=1).astype(np.float32)


def augment_sequence(seq):
    base = seq[:, :BASE_FEATURE_SIZE].copy()

    if np.random.rand() < 0.5:
        base[:, 0:63:3]   = 1 - base[:, 0:63:3]
        base[:, 63:126:3] = 1 - base[:, 63:126:3]
        right_block     = base[:, 0:63].copy()
        left_block      = base[:, 63:126].copy()
        base[:, 0:63]   = left_block
        base[:, 63:126] = right_block

    if np.random.rand() < 0.7:
        motion = base[1:] - base[:-1]
        scale  = np.random.uniform(0.8, 1.2)
        motion *= scale
        temp    = np.zeros_like(base)
        temp[0] = base[0]
        for i in range(1, len(base)):
            temp[i] = temp[i-1] + motion[i-1]
        base = temp

    if np.random.rand() < 0.6:
        base += np.random.normal(0, 0.003, base.shape)

    motion     = np.zeros_like(base)
    motion[1:] = base[1:] - base[:-1]

    return np.concatenate([base, motion], axis=1).astype(np.float32)