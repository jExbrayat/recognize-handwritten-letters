import pickle

import numpy as np
from keras.callbacks import EarlyStopping, History
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.metrics import F1Score
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

dataset = np.loadtxt("data/raw/original-dataset.csv", delimiter=",")

X = dataset[:, 1:]
y = dataset[:, 0]

# Normalize X values between 0 and 1
X = X.astype("float32") / 255

X, X_test, y, y_test = train_test_split(X, y, random_state=42)

# Save test dataset to test model without reloading whole dataset after training
test_dataset = np.concatenate((y_test.reshape(-1, 1), X_test), axis=1)
np.savetxt("data/processed/test-dataset.csv", test_dataset, delimiter=",")

# Transform targets into categorical variables
y = to_categorical(y)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Input(shape=(784,)))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(26, activation="softmax"))

model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(),
    metrics=["accuracy", F1Score()],
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=50,
    mode="min",
    restore_best_weights=True,
    start_from_epoch=150,
)

history = History()

model.fit(
    X,
    y,
    epochs=750,
    validation_split=0.2,
    callbacks=[early_stopping, history],
)

# Get model's loss and accuracy
model_eval = model.evaluate(X, y)

# Save model's results
save_path = "data/model_weights/cnn"

model.save(f"{save_path}/cnn_experiment_21-12-23.h5")

with open(f"{save_path}/cnn_experiment_history_21-12-23.pkl", "wb") as history_file:
    pickle.dump(history.history, history_file)

with open(f"{save_path}/cnn_experiment_evaluate.pkl", "wb") as eval_file:
    pickle.dump(model_eval, eval_file)
