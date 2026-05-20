import cv2 as cv
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import time
import threading
import queue
from collections import deque, Counter
from keras.models import load_model
from flask import Flask, Response, render_template, jsonify, request

from config.paths import (
    static_model_path, static_encoder_path, feature_order_path,
    dynamic_model_path, dynamic_encoder_path, dynamic_scaler_path,
    angle_csv_path, hand_landmark_path
)
from src.features import palm_size
from src.model_utils import compute_thresholds
from src.helper_function import init_mediapipe, add_motion, process_frame, normalize_length
from src.static_predictor import static_pipeline
from src.word_formation_layer import WordFormation
from src.motion_detector import MotionDetector

print("Loading models...")
s_model   = joblib.load(static_model_path)
s_encoder = joblib.load(static_encoder_path)
f_order   = joblib.load(feature_order_path)
d_model   = load_model(dynamic_model_path, compile=False)
d_encoder = joblib.load(dynamic_encoder_path)
d_scaler  = joblib.load(dynamic_scaler_path)
TH        = compute_thresholds(angle_csv_path, hand_landmark_path)


d_model.predict(np.zeros((1, 20, 300)), verbose=0)
s_model.predict_proba(pd.DataFrame([np.zeros(len(f_order))], columns=f_order))

holistic = init_mediapipe()
hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.4)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 360)

word_engine = WordFormation()
motion_detect = MotionDetector(
    window=20,
    static_threshold=0.0003,
    dynamic_threshold=0.0012,
    confirm_frames=6
)

pts_buf = deque(maxlen=5)
s_prev_label = ""
s_gesture_start = None

seq_buf = deque(maxlen=20)
d_last_valid = None
d_vote_buf = deque(maxlen=6)

d_disp = ""
d_disp_conf = 0.0
s_disp = ""
s_disp_conf = 0.0

CONF_D = 0.75
CONF_S = 0.55

DYNAMIC_WINDOW = 20
DYNAMIC_MIN_FRAMES = 8
DYNAMIC_PREDICT_EVERY = 2
DYNAMIC_STABLE_COUNT = 2
DYNAMIC_MARGIN = 0.05
DYNAMIC_COOLDOWN = 0.9

INFER_WIDTH = 480
INFER_HEIGHT = 270

state_lock = threading.Lock()
state = {
    "mode": "AUTO",
    "active": "None",
    "motion": "UNKNOWN",
    "dynamic": "",
    "dynamic_conf": 0.0,
    "static": "",
    "static_conf": 0.0,
    "word": "",
    "sentence": "",
    "fps": 0,
    "current_word": "",
    "d_label": "",
    "d_conf": 0.0,
    "s_label": "",
    "s_conf": 0.0,
    "motion_mode": "UNKNOWN",
    "s_count": 0,
    "s_confirmed": ""
}

frame_queue = queue.Queue(maxsize=1)
encode_queue = queue.Queue(maxsize=1)
gru_input_q = queue.Queue(maxsize=1)

latest_frame = None
frame_lock = threading.Lock()

gru_lock = threading.Lock()
gru_result = {"label": "","conf": 0.0,"final": "",
              "last_label": "","last_time": time.time(),
              "candidate": "","candidate_count": 0}

def normalize_hand_orientation(features, handedness):
    f = features.copy()
    if handedness == "Left":
        f[0::3] = 1 - f[0::3]
    return f

def prepare_dynamic_sequence(sequence):
    seq = np.asarray(add_motion(sequence), dtype=np.float32)

    if len(seq) < DYNAMIC_WINDOW:
        seq = normalize_length(seq, target_len=DYNAMIC_WINDOW)

    return seq.astype(np.float32)

def reset_pipeline():
    global s_prev_label, s_gesture_start, d_last_valid
    global d_disp, d_disp_conf, s_disp, s_disp_conf

    pts_buf.clear()
    seq_buf.clear()
    d_vote_buf.clear()
    motion_detect._reset()

    s_prev_label = ""
    s_gesture_start = None
    d_last_valid = None

    d_disp = ""
    d_disp_conf = 0.0
    s_disp = ""
    s_disp_conf = 0.0

    with gru_lock:
        gru_result.update({
            "label": "",
            "conf": 0.0,
            "final": "",
            "last_label": "",
            "last_time": time.time(),
            "candidate": "",
            "candidate_count": 0
        })

    with state_lock:
        state["active"] = "None"
        state["dynamic"] = ""
        state["dynamic_conf"] = 0.0
        state["static"] = ""
        state["static_conf"] = 0.0
        state["current_word"] = ""
        state["d_label"] = ""
        state["d_conf"] = 0.0
        state["s_label"] = ""
        state["s_conf"] = 0.0
        state["s_count"] = 0
        state["s_confirmed"] = ""


def camera_reader():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv.flip(frame, 1)

        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass

        frame_queue.put(frame)

def gru_worker():
    while True:
        seq_scaled, last_label, last_time = gru_input_q.get()

        pred = d_model.predict(seq_scaled, verbose=0)[0]
        d_conf = float(np.max(pred))
        d_label = d_encoder.inverse_transform([np.argmax(pred)])[0]

        top2 = np.sort(np.partition(pred, -2)[-2:])
        d_margin = float(top2[-1] - top2[-2])

        final_label = ""
        current_time = time.time()

        with gru_lock:
            candidate = gru_result["candidate"]
            candidate_count = gru_result["candidate_count"]

        if d_conf > CONF_D and d_margin >= DYNAMIC_MARGIN:
            if d_label == candidate:
                candidate_count += 1
            else:
                candidate = d_label
                candidate_count = 1

            stable_dynamic = candidate_count >= DYNAMIC_STABLE_COUNT

            if stable_dynamic and d_label != last_label:
                final_label = d_label
                last_label = d_label
                last_time = current_time
            elif stable_dynamic and current_time - last_time > DYNAMIC_COOLDOWN:
                final_label = d_label
                last_time = current_time
        else:
            candidate = ""
            candidate_count = 0

        with gru_lock:
            gru_result["label"] = d_label
            gru_result["conf"] = d_conf
            gru_result["final"] = final_label
            gru_result["last_label"] = last_label
            gru_result["last_time"] = last_time
            gru_result["candidate"] = candidate
            gru_result["candidate_count"] = candidate_count


def inference_loop():
    global s_prev_label, s_gesture_start, d_last_valid
    global d_disp, d_disp_conf, s_disp, s_disp_conf

    frame_n = 0
    fps = 0
    fps_n = 0
    fps_t = time.time()

    while True:
        frame = frame_queue.get()

        while not frame_queue.empty():
            try:
                frame = frame_queue.get_nowait()
            except queue.Empty:
                break

        frame_n += 1

        try:
            infer_frame = cv.resize(frame, (INFER_WIDTH, INFER_HEIGHT))
            rgb = cv.cvtColor(infer_frame, cv.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            mp_res = holistic.process(rgb)
            rgb.flags.writeable = True

            hand_present = bool(mp_res.left_hand_landmarks or mp_res.right_hand_landmarks)
            motion_mode = motion_detect.update(mp_res) if hand_present else "UNKNOWN"

            with state_lock:
                current_mode = state["mode"]

            pipeline = motion_mode if current_mode == "AUTO" else current_mode

            s_confirmed = ""
            s_conf = 0.0

            if pipeline == "STATIC" and hand_present:
                hand_res = hands.process(rgb)

                stable_label, s_prev_label, s_gesture_start = static_pipeline(
                    hand_res, pts_buf, s_prev_label, s_gesture_start,
                    s_model, s_encoder, f_order, TH
                )

                if stable_label:
                    s_confirmed = stable_label.lower()
                    s_disp = stable_label.lower()

                if hand_res and hand_res.multi_hand_landmarks and len(pts_buf) == 5:
                    mean = np.mean(np.stack(list(pts_buf)), axis=0)
                    mean /= palm_size(mean)

                    feat = {f"x{i}": mean[i, 0] for i in range(21)}
                    feat.update({f"y{i}": mean[i, 1] for i in range(21)})

                    probs = s_model.predict_proba(pd.DataFrame([feat])[f_order])[0]
                    s_conf = float(np.max(probs))
                    s_disp_conf = s_conf

            elif pipeline == "DYNAMIC" or not hand_present:
                pts_buf.clear()
                s_prev_label = ""
                s_gesture_start = None
                s_disp = ""
                s_disp_conf = 0.0

            d_confirmed = ""
            d_conf = 0.0

            if pipeline == "DYNAMIC" and hand_present:
                if mp_res.right_hand_landmarks:
                    handedness = "Right"
                elif mp_res.left_hand_landmarks:
                    handedness = "Left"
                else:
                    handedness = None

                features, d_last_valid, _ = process_frame(
                    infer_frame, mp_res, d_last_valid, precomputed=True
                )

                if features is not None:
                    features = normalize_hand_orientation(features, handedness)
                    seq_buf.append(features)

                if len(seq_buf) >= DYNAMIC_MIN_FRAMES and frame_n % DYNAMIC_PREDICT_EVERY == 0:
                    seq = prepare_dynamic_sequence(seq_buf)
                    seq_scaled = d_scaler.transform(
                        seq.reshape(-1, seq.shape[1])
                    ).reshape(1, DYNAMIC_WINDOW, -1)

                    with gru_lock:
                        ll = gru_result["last_label"]
                        lt = gru_result["last_time"]

                    if gru_input_q.full():
                        try:
                            gru_input_q.get_nowait()
                        except queue.Empty:
                            pass

                    gru_input_q.put((seq_scaled, ll, lt))

                with gru_lock:
                    d_label = gru_result["label"]
                    d_conf = gru_result["conf"]
                    d_confirmed = gru_result["final"]
                    gru_result["final"] = ""

                if d_label and d_conf >= CONF_D:
                    d_vote_buf.append(d_label)
                    voted = Counter(d_vote_buf).most_common(1)[0]

                    if voted[1] >= 2:
                        d_disp = voted[0]
                        d_disp_conf = d_conf

            elif pipeline == "STATIC" or not hand_present:
                seq_buf.clear()
                d_vote_buf.clear()
                d_disp = ""
                d_disp_conf = 0.0

                with gru_lock:
                    gru_result["label"] = ""
                    gru_result["conf"] = 0.0
                    gru_result["final"] = ""
                    gru_result["candidate"] = ""
                    gru_result["candidate_count"] = 0

            if pipeline == "DYNAMIC" and d_confirmed:
                active = "Dynamic"
            elif pipeline == "STATIC" and s_confirmed:
                active = "Static"
            elif pipeline == "DYNAMIC" and d_conf >= CONF_D:
                active = "Dynamic"
            elif pipeline == "STATIC" and s_conf >= CONF_S:
                active = "Static"
            else:
                active = "None"

            current_word, sentence = word_engine.update(
                s_confirmed if active == "Static" else "",
                d_confirmed if active == "Dynamic" else "",
                active,
                hand_present
            )

            fps_n += 1
            if time.time() - fps_t >= 1.0:
                fps = fps_n
                fps_n = 0
                fps_t = time.time()

            with state_lock:
                state["active"] = active
                state["motion"] = motion_mode
                state["dynamic"] = d_disp
                state["dynamic_conf"] = d_disp_conf
                state["static"] = s_disp
                state["static_conf"] = s_disp_conf
                state["word"] = current_word
                state["sentence"] = sentence
                state["fps"] = fps
                state["current_word"] = current_word
                state["d_label"] = d_disp
                state["d_conf"] = d_disp_conf
                state["s_label"] = s_disp
                state["s_conf"] = s_disp_conf
                state["motion_mode"] = motion_mode
                state["s_count"] = len(pts_buf)
                state["s_confirmed"] = s_confirmed

        except Exception:
            import traceback
            traceback.print_exc()

        if encode_queue.full():
            try:
                encode_queue.get_nowait()
            except queue.Empty:
                pass

        encode_queue.put(frame)


def encoder_thread():
    global latest_frame

    while True:
        frame = encode_queue.get()
        ret, buf = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 50])

        if ret:
            with frame_lock:
                latest_frame = buf.tobytes()

app = Flask(__name__)

def generate_frames():
    while True:
        with frame_lock:
            frame = latest_frame

        if frame is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

        time.sleep(0.005)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/state")
def get_state():
    with state_lock:
        return jsonify(state)


@app.route("/control", methods=["POST"])
def control():
    data = request.get_json(silent=True) or {}
    action = data.get("action", "")

    if action == "mode":
        with state_lock:
            state["mode"] = {"AUTO": "STATIC", "STATIC": "DYNAMIC"}.get(
                state["mode"], "AUTO"
            )
        reset_pipeline()
    elif action == "reset":
        word_engine.reset()
        reset_pipeline()
    elif action == "backspace":
        word_engine.backspace()
    elif action == "space":
        word_engine.manual_space()
    elif action == "commit_word":
        word_engine.commit_word()

    return jsonify({"status": "ok"})


if __name__ == "__main__":
    threading.Thread(target=camera_reader, daemon=True).start()
    threading.Thread(target=gru_worker, daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()
    threading.Thread(target=encoder_thread, daemon=True).start()

    app.run(host="0.0.0.0",port=5000,threaded=True,debug=False,use_reloader=False)
