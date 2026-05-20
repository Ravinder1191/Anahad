import numpy as np
from collections import deque

class MotionDetector:
    def __init__(self, window=20, static_threshold=0.0003, dynamic_threshold=0.0012, confirm_frames=6):
        self.window = window
        self.static_threshold = static_threshold
        self.dynamic_threshold = dynamic_threshold
        self.confirm_frames = confirm_frames

        self.motion_history = deque(maxlen=window)
        self.pending_mode = None
        self.pending_count = 0
        self.confirmed_mode = "STATIC"

    def update(self, mp_result):
        hand_state = self._get_hand_state(mp_result)

        if hand_state is None:
            self._reset()
            return "UNKNOWN"

        self.motion_history.append(hand_state)

        if len(self.motion_history) < self.window // 3:
            return self.confirmed_mode

        motion_score = self._motion_score()
        candidate = self._classify(motion_score)

        if candidate == self.pending_mode:
            self.pending_count += 1
        else:
            self.pending_mode = candidate
            self.pending_count = 1

        if self.pending_count >= self.confirm_frames:
            self.confirmed_mode = self.pending_mode

        return self.confirmed_mode

    def _get_hand_state(self, mp_result):
        hands = [
            h for h in (mp_result.right_hand_landmarks, mp_result.left_hand_landmarks)
            if h is not None
        ]

        if not hands:
            return None

        hand_features = []

        for hand in hands:
            pts = np.array([[lm.x, lm.y] for lm in hand.landmark], dtype=np.float32)
            center = np.mean(pts, axis=0)
            spread = float(np.mean(np.linalg.norm(pts - center, axis=1)))
            hand_features.append(np.array([center[0], center[1], spread], dtype=np.float32))

        return np.mean(np.stack(hand_features), axis=0)

    def _motion_score(self):
        pts = np.array(self.motion_history)

        center_var = float(np.mean(np.var(pts[:, :2], axis=0)))
        spread_var = float(np.var(pts[:, 2]))

        if len(pts) > 1:
            velocity = float(np.mean(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)))
        else:
            velocity = 0.0

        return center_var + (spread_var * 0.7) + (velocity * 0.04)

    def _classify(self, motion_score):
        if motion_score < self.static_threshold:
            return "STATIC"
        elif motion_score > self.dynamic_threshold:
            return "DYNAMIC"
        else:
            return self.confirmed_mode

    def _reset(self):
        self.motion_history.clear()
        self.pending_mode = None
        self.pending_count = 0
