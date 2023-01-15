import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

MODEL_PATH = "./models"

EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.00001
OPT = keras.optimizers.Adam(learning_rate = LEARNING_RATE)

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

df = pd.read_csv("./data/emotion_landmarks.csv")

print("dataframe shape:", df.shape)

X = df.iloc[:,2:].values
y = df.iloc[:,1].values

ohe = OneHotEncoder()
y = ohe.fit_transform(y.reshape(-1, 1)).toarray()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.1,shuffle=True, random_state=42)

print("X shape:",X.shape)
print("y shape:",y.shape)

model = Sequential()
model.add(Dense(2048, input_dim=X.shape[1], activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(y.shape[1], activation="softmax"))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer=OPT, metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data = (X_test,y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

model.save(os.path.join(MODEL_PATH,'ANN.h5'))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show(block=True)
plt.savefig("./tmp/training_history.png")



