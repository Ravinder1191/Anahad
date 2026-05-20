import joblib
import numpy as np
import pandas as pd
from src.features import fingers_distance, calculate_wrist_mcp_angle

def load_model(model_path, encoder_path, feature_order_path):
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    feature_order = joblib.load(feature_order_path)
    return model, encoder, feature_order

def decode_prediction(encoder, prediction):
    prediction_id = int(np.asarray(prediction).argmax())
    if hasattr(encoder, "classes_"):
        return encoder.inverse_transform([prediction_id])[0]
    if hasattr(encoder, "categories_"):
        one_hot = np.zeros((1, len(encoder.categories_[0])), dtype=np.float32)
        one_hot[0, prediction_id] = 1.0
        decoded = encoder.inverse_transform(one_hot)[0]
        return decoded[0] if isinstance(decoded, (list, tuple, np.ndarray)) else decoded
    raise TypeError(f"Unsupported encoder type: {type(encoder).__name__}")


def compute_mean_distance(landmark_data, label, index1, index2):
    distances = []
    subset = landmark_data[landmark_data["label"] == label]
    for _, row in subset.iterrows():
        pts = np.array([[row[f"x{i}"], row[f"y{i}"]] for i in range(21)])
        distances.append(fingers_distance(pts[index1], pts[index2]))
    return np.mean(distances) if distances else 0.0


def compute_thresholds(angle_csv, landmarks_csv):
    angle_data = pd.read_csv(angle_csv)
    landmark_data = pd.read_csv(landmarks_csv)

    th = {}
    angle_means = angle_data.groupby("label").mean(numeric_only=True)

    def safe_angle_mean(label, column, fallback=0.0):
        if label in angle_means.index and column in angle_means.columns:
            return float(angle_means.loc[label, column])
        return float(fallback)

    th["c_o"] = (safe_angle_mean("C", "thumb_angle", 120.0) + safe_angle_mean("O", "thumb_angle", 80.0)) / 2

    landmark_data["wrist_tilt"] = landmark_data.apply(
        lambda row: calculate_wrist_mcp_angle(row["x0"], row["y0"], row["x9"], row["y9"]), axis=1)
    g_tilt = landmark_data[landmark_data["label"] == "G"]["wrist_tilt"]
    th["g_q_threshold"] = float(g_tilt.mean()) if not g_tilt.empty else 0.0

    th["i_o"] = (safe_angle_mean("O", "pinky_angle", 120.0) + safe_angle_mean("I", "pinky_angle", 140.0)) / 2
    th["v_dist"] = compute_mean_distance(landmark_data, "V", 8, 12)
    th["x_index"] = safe_angle_mean("X", "index_angle", 130.0)
    th["fist_pip"] = (compute_mean_distance(landmark_data, "S", 6, 10) + compute_mean_distance(landmark_data, "S", 10,
                                                                                               14) + compute_mean_distance(
        landmark_data, "S", 14, 18)) / 3

    return th


def apply_rules(label, angles, dists, mean_pts, TH):
    avg_finger_angle = (angles["index"] + angles["middle"] + angles["ring"] + angles["pinky"]) / 4
    is_fist = all(angle > 160 for angle in (angles["index"], angles["middle"], angles["ring"], angles["pinky"]))

    if label in ("B", "C"):
        return "B" if avg_finger_angle > TH["c_o"] else "C"  # Matches lowercase key

    elif label in ("R", "L", "A", "D", "E", "F", "O", "Q", "W", "K", "Y", "I"):
        return label

    elif label in ("X", "H", "P"):
        if angles["index"] < TH["x_index"]:  # Matches lowercase key
            return "X"
        return "H" if mean_pts[8, 1] > mean_pts[5, 1] else "P"

    elif label in ("U", "V"):
        return "V" if dists["index_middle"] > TH["v_dist"] else "U"  # Matches lowercase key

    elif label in ("Z", "J"):
        return "Unknown"

    elif label in ("G", "M", "S"):
        if not is_fist: return "G"
        thumb_height = mean_pts[4, 1]
        knuckle_height = mean_pts[17, 1]
        return "M" if thumb_height < knuckle_height else "S"

    elif label in ("N", "T"):
        thumb_to_index_mcp = fingers_distance(mean_pts[4], mean_pts[6])
        thumb_to_ring_mcp = fingers_distance(mean_pts[4], mean_pts[14])
        if abs(thumb_to_index_mcp - thumb_to_ring_mcp) < 0.05: return label
        return "T" if thumb_to_index_mcp < thumb_to_ring_mcp else "N"

    else:
        return "Unknown"