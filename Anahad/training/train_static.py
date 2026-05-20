import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

data_path = "landmarks.csv"
data = pd.read_csv(data_path)

z_cols = [f"z{i}" for i in range(21)]
x = data.drop(columns=z_cols + ["label"])
y = data["label"]

print("Feature shape:", x.shape)

encoder = LabelEncoder()
encoded = encoder.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(
        x, encoded, test_size=0.2, random_state=42, stratify=encoded
    )
# MLP Pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp",MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42,
                ),
            ),
        ]
    )

pipe.fit(x_train, y_train)

y_test_pred = pipe.predict(x_test)

print("Train accuracy:", pipe.score(x_train, y_train))
print("Test accuracy :", pipe.score(x_test, y_test))
print(confusion_matrix(y_test, y_test_pred))

# Saving model
model_path = "model.pkl"
encoder_path = "encoder.pkl"
feature_order_path = "feature_order.pkl"

joblib.dump(pipe, model_path)
joblib.dump(encoder, encoder_path)
joblib.dump(list(x.columns), feature_order_path)

print("Model saved to:", model_path)
print("Encoder saved to:", encoder_path)
print("Feature order saved to:", feature_order_path)