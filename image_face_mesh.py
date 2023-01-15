import cv2
import mediapipe as mp
import numpy as np


indecies = np.array([
  362, 384, 386, 388, 363, 390, 374, 381,\
  464, 441, 443, 445, 446, 448, 450, 452,\
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

# For static images:
mp_drawing = mp.solutions.drawing_utils #helps draw all the 468 landmarks -----------------------------------------
mp_drawing_styles = mp.solutions.drawing_styles # styling --------------------------
mp_face_mesh = mp.solutions.face_mesh #main model import -----------------------------------------

IMAGE_FILES = ["IMG_0587.jpg"]
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    width = image.shape[1]
    height = image.shape[0]
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()
    
    for face_landmarks in results.multi_face_landmarks:
    #   print('face_landmarks:', face_landmarks)
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_iris_connections_style())
    # print(results.multi_face_landmarks[0].landmark[2])
    # center = results.multi_face_landmarks[0].landmark[2]
    for i in indecies:
      p = results.multi_face_landmarks[0].landmark[i]
      cv2.circle(annotated_image, (int(p.x*width),int(p.y*height)), 10, (0,0,255), 6)
    cv2.imwrite('tmp/annotated_image' + str(idx) + '.png', annotated_image)