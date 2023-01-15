from keras.models import load_model
import numpy as np
import cv2
import os
import mediapipe as mp

import utils

MODEL_PATH = "./models"
emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
mp_face_mesh = mp.solutions.face_mesh
indecies = np.array([
  362, 384, 386, 388, 363, 390, 374, 381,\
  464, 441, 443,  445, 446, 448, 450, 452,\
  33, 161, 159, 157, 133, 154, 145, 163,\
  226, 225, 223, 221, 244, 232, 230, 228,\
  336, 296, 334, 293, 300, 295, 285,\
  70, 63, 105, 66, 107, 55, 65,\
  129, 206, 212, 202, 194, 200, 418, 422, 432, 426, 358,\
  167, 92, 57, 106, 83, 313, 335, 287, 322, 393,\
  0, 37, 40, 61, 91, 84, 314, 321, 291, 270, 267,\
  13, 82, 80, 78, 88, 87, 317, 318, 308, 310, 312,\
  174, 134, 48, 240, 2, 460, 278, 420, 399,\
  132, 58, 136, 149, 148, 377, 378, 365, 288, 361
])

model = load_model(os.path.join(MODEL_PATH,'ANN.h5'))

print(model.summary())

img = cv2.imread("IMG_0587.jpg")

with mp_face_mesh.FaceMesh(
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as face_mesh:

    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    results = face_mesh.process(img)

    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    all_landmarks = np.array(results.multi_face_landmarks[0].landmark)

    all_landmarks = all_landmarks[indecies]

    distances = utils.calculate_all_distances(all_landmarks)
    print(distances.shape)

pred = model.predict(np.array([distances]))

print(pred)
