import numpy as np
import pandas as pd
import time
from src.features import palm_size, extract_features
from src.model_utils import apply_rules

WINDOW         = 5
CONF_THRESHOLD = 0.55
STABLE_TIME    = 0.5

def static_pipeline(results,
                     pts_buffer,
                     previous_label,
                     gesture_start_time,
                     model,
                     encoder,
                     feature_order,
                     TH):

    if results is None or results.multi_hand_landmarks is None:
        pts_buffer.clear()
        return "", previous_label, gesture_start_time

    if len(results.multi_hand_landmarks) > 1:
        pts_buffer.clear()
        return "", previous_label, gesture_start_time

    hand       = results.multi_hand_landmarks[0]
    handedness = results.multi_handedness[0].classification[0].label

    pts      = np.array([[lm.x, lm.y] for lm in hand.landmark], dtype=np.float32)
    norm_pts = pts - pts[0]
    if handedness == "Right":
        norm_pts[:, 0] *= -1

    pts_buffer.append(norm_pts)

    detected_label = ""
    confidence     = 0.0

    if len(pts_buffer) == WINDOW:
        mean_pts = np.mean(np.stack(pts_buffer), axis=0)
        mean_pts /= palm_size(mean_pts)

        angles, dists, _ = extract_features(mean_pts)

        feat = {f"x{i}": mean_pts[i, 0] for i in range(21)}
        feat.update({f"y{i}": mean_pts[i, 1] for i in range(21)})

        X_df       = pd.DataFrame([feat])[feature_order]
        probs      = model.predict_proba(X_df)[0]
        confidence = float(np.max(probs))
        raw_label  = encoder.inverse_transform([np.argmax(probs)])[0]

        detected_label = apply_rules(raw_label, angles, dists, mean_pts, TH)

    stable_label = ""
    current_time = time.time()

    if len(detected_label) == 1 and confidence > CONF_THRESHOLD:
        if detected_label != previous_label:
            previous_label     = detected_label
            gesture_start_time = current_time
        else:
            if gesture_start_time and (current_time - gesture_start_time >= STABLE_TIME):
                stable_label = detected_label
    else:
        previous_label     = ""
        gesture_start_time = None

    return stable_label, previous_label, gesture_start_time