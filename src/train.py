import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

def train_model(X, y):
    try:
        model = Sequential()
        model.add(Dense(16, input_dim=X.shape[1], activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(X, y, epochs=10, batch_size=10)

        print("Neural network trained successfully")
        return model

    except Exception as e:
        print(f"Training error: {e}")


def save_model(model):
    model.save("models/model.h5")