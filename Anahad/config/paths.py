from pathlib import Path

config_dir = Path(__file__).resolve().parent
project_root = config_dir.parent

static_dir          = project_root / "Models" / "static models"
static_model_path   = str(static_dir / "model.pkl")
static_encoder_path = str(static_dir / "encoder.pkl")
feature_order_path  = str(static_dir / "feature_order.pkl")

dynamic_dir          = project_root / "Models" / "dynamic models"
dynamic_model_path   = str(dynamic_dir / "tuned gru model.h5")
dynamic_encoder_path = str(dynamic_dir / "tuned encoder.pkl")
dynamic_scaler_path  = str(dynamic_dir / "tuned scaler.pkl")

data_dir           = project_root / "data_pipeline"
angle_csv_path     = str(data_dir / "joint_angles.csv")
hand_landmark_path = str(data_dir / "hand_landmarks.csv")

cleaned_data_dir    = str(project_root / "cleaned data")
augmented_data_dir  = str(project_root / "augmented data")
flipped_data_dir    = str(project_root / "fliped augmenated data")