import os
import joblib
import numpy as np
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Attention, GlobalAveragePooling1D, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# build model function for hyperparmater tunning
def build_model(hp):

    units1 = hp.Int('units1', 64, 160, step=16)
    units2 = hp.Int('units2', 32, 96, step=8)
    dropout = hp.Float('dropout', 0.25, 0.5, step=0.05)
    l2_reg = hp.Float('l2_reg', 1e-4, 5e-3, sampling='log')
    lr = hp.Float('lr', 3e-4, 1e-3, sampling='log')
    bidirectional = hp.Boolean('bidirectional')

    inputs = Input(shape=(x_train.shape[1], x_train.shape[2]))

    if bidirectional:
        x = Bidirectional(GRU(units1, return_sequences=True))(inputs)
    else:
        x = GRU(units1, return_sequences=True)(inputs)

    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = GRU(units2, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    attn = Attention()([x, x])

    x = GlobalAveragePooling1D()(attn)

    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout * 0.8)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=1.0),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )


data_path = "fliped augmenated data"

x = []
y = []

labels = sorted(os.listdir(data_path))

for label in labels:
    folder = os.path.join(data_path, label)
    if not os.path.isdir(folder):
      continue

    for file in os.listdir(folder):
        if file.endswith(".npy"):
            seq = np.load(os.path.join(folder, file))
            x.append(seq)
            y.append(label)

x = np.array(x)
y = np.array(y)

print("x shape:", x.shape)
print("y shape:", y.shape)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

samples, timesteps, features = x.shape

scaler = StandardScaler()
x_reshaped = x.reshape(-1, features)
x_scaled = scaler.fit_transform(x_reshaped)
x = x_scaled.reshape(samples, timesteps, features)

x = x_scaled.reshape(samples, timesteps, features)
print("sclaed x shape:", x.shape)
# %%
x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.2, random_state=42
)
print(f"x train shape: {x_train.shape}")
print(f"y train shape: {y_train.shape}")
# %%
class_weights_array = compute_class_weight(class_weight='balanced', classes=np.unique(y_train),
y=y_train)

class_weight_dict = dict(enumerate(class_weights_array))
print("Class weights:")

for idx, weight in class_weight_dict.items():
    label_name = encoder.inverse_transform([idx])[0]
    print(f"  {label_name}: {weight:.3f}")

input_shape = (x.shape[1],x.shape[2])
num_classes = len(np.unique(y))
# Searching best paramterters
tuner = kt.RandomSearch(build_model, objective='val_accuracy', max_trials=10, executions_per_trial=1, directory='tuner_results', project_name='gru_tuning',overwrite=True)
tuner.search(x_train, y_train, epochs=50, validation_data=(x_test, y_test),  class_weight=class_weight_dict, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])
# fitting best best paramters in final model
best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hp.values)

best_model.save("final tuned gru model.h5")
joblib.dump(scaler, "final tuned scaler.pkl")
joblib.dump(encoder, "final tuned encoder.pkl")