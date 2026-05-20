import numpy as np
import os
from src.helper_function import normalize_length, flip_sequence, augment_sequence

data_path   = "np files data"
output_path = "fliped augmenated data"

target_samples = 100
min_raw = 10

os.makedirs(output_path, exist_ok=True)

labels = sorted(os.listdir(data_path))

if not labels:
    print("ERROR: No labels found in data_path")
    print("Check path:", data_path)
    exit()

for label in labels:

    input_folder  = os.path.join(data_path, label)
    output_folder = os.path.join(output_path, label)

    if not os.path.isdir(input_folder):
        continue

    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.endswith(".npy")]
    print(f"\nProcessing {label} ({len(files)} raw samples)")

    if len(files) < min_raw:
        print(f"Only {len(files)} raw videos — consider recording more")

    valid_files = []

    for f in files:
        path = os.path.join(input_folder, f)
        try:
            seq = np.load(path)
            if seq.shape != (20, 300):
                print(f"  Skipped wrong shape {seq.shape}: {f}")
                continue
            if np.isnan(seq).any():
                print(f"  Skipped NaN: {f}")
                continue
            seq = normalize_length(seq)
            np.save(os.path.join(output_folder, f), seq)
            valid_files.append(f)
        except Exception as e:
            print(f"  Error {f}: {e}")
            continue

    if not valid_files:
        print(f"No valid files for {label} — skipping")
        continue

    for f in valid_files:
        seq     = np.load(os.path.join(output_folder, f))
        flipped = flip_sequence(seq)
        base    = f.replace(".npy", "")
        np.save(os.path.join(output_folder,
                f"{base}_flip.npy"), flipped)

    current_count = len(valid_files) * 2
    files_array   = np.array(valid_files)

    if current_count < target_samples:
        needed = target_samples - current_count
        aug_id = 0

        while aug_id < needed:
            file = np.random.choice(files_array)
            seq  = np.load(os.path.join(output_folder, file))  # normalized original ✔
            aug  = augment_sequence(seq)
            base = file.replace(".npy", "")
            np.save(os.path.join(output_folder,
                    f"{base}_aug{aug_id}.npy"), aug)
            aug_id += 1

        print(f"  Added {needed} augmented samples")
    else:
        print(f"  Enough samples ({current_count})")

print("Final counts:")
for label in sorted(os.listdir(output_path)):
    folder = os.path.join(output_path, label)
    if not os.path.isdir(folder):
        continue
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    status = "done" if len(files) >= 80 else "failed"
    print(f"  {status} {label}: {len(files)} samples")