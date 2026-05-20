import time
import numpy as np
from helper_function import add_motion, process_frame


def dynamic_pipeline(frame,
                     mp_res,          
                     sequence_dynamic,
                     last_valid,
                     dynamic_model,
                     dynamic_scaler,
                     dynamic_encoder,
                     frame_count,
                     last_label,
                     last_time,
                     CONF_THRESHOLD=0.6,
                     COOLDOWN=1.2):

    d_label     = ""
    d_conf      = 0.0
    final_label = ""

    # precomputed=True → skips holistic.process() entirely
    features, last_valid, _ = process_frame(frame, mp_res, last_valid, precomputed=True)

    if features is not None:
        sequence_dynamic.append(features)

    if len(sequence_dynamic) >= 5 and frame_count % 2 == 0:
        seq = add_motion(sequence_dynamic)
        seq = np.asarray(seq, dtype=np.float32)

        if len(seq) < 20:
            pad = np.zeros((20 - len(seq), seq.shape[1]))
            seq = np.vstack((pad, seq))

        seq_scaled = dynamic_scaler.transform(
            seq.reshape(-1, seq.shape[1])
        ).reshape(1, 20, -1)

        pred    = dynamic_model.predict(seq_scaled, verbose=0)[0]
        d_conf  = float(np.max(pred))
        d_label = dynamic_encoder.inverse_transform([np.argmax(pred)])[0]

    current_time = time.time()

    if d_conf > CONF_THRESHOLD:
        if d_label != last_label:
            final_label = d_label
            last_label  = d_label
            last_time   = current_time
        elif current_time - last_time > COOLDOWN:
            final_label = d_label
            last_time   = current_time

    return final_label, d_label, d_conf, last_valid, last_label, last_time