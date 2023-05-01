import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection=mp_face_detection.FaceDetection(0.75)
import numpy as np
import tensorflow as tf
from keras.models import load_model
model = tf.keras.models.load_model("new_model4")
class_names=['awake' 'sleep']
# For webcam input:
cap = cv2.VideoCapture(1)
while True:
  success, image = cap.read()
  image.flags.writeable = False
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  results = face_detection.process(image)
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  if results.detections:
    for detection in results.detections:
      bboxc=detection.location_data.relative_bounding_box
      ih, iw, ic = image.shape
      xmin = int(bboxc.xmin * iw)
      ymin = int(bboxc.ymin * ih)
      w = int(bboxc.width * iw)
      h = int(bboxc.height * ih)
      x1, y1 = xmin, ymin
      x2, y2 = x1 + w, y1 + h
      x1,y1,x2,y2=x1,y1,x2,y2
      cv2.rectangle(image, (x1, y1), (x2, y2), (250, 0, 250), 3)
      try:
        cropped_img = image[y1:y2, x1:x2]
        resize = tf.image.resize(cropped_img, (224, 224))
        predictions = model.predict(np.expand_dims(resize/255, axis=0))
        predicted_label =predictions[0].argmax()
        if predicted_label == 1:
          window_name = 'Image'
          font = cv2.FONT_HERSHEY_SIMPLEX
          org = (50, 50)
          fontScale = 1
          color = (255, 0, 0)
          thickness = 2
          cv2.putText(image, 'sleep', org, font, fontScale, color, thickness, cv2.LINE_AA)
          cv2.rectangle(image, (x1, y1), (x2, y2), (250, 0, 250), 3)
          cv2.namedWindow("Resize", cv2.WINDOW_NORMAL)
          cv2.resizeWindow("Resize", 400, 400)
          cv2.imshow("Resize", image)
        else:
          window_name = 'Image'
          font = cv2.FONT_HERSHEY_SIMPLEX
          org = (50, 50)
          fontScale = 1
          color = (255, 0, 255)
          thickness = 2
          image = cv2.putText(image, 'awake' , org, font, fontScale, color, thickness, cv2.LINE_AA)
          cv2.rectangle(image, (x1, y1), (x2, y2), (250, 0, 250), 3)
          cv2.namedWindow("Resize", cv2.WINDOW_NORMAL)
          cv2.resizeWindow("Resize", 600, 400)
          cv2.imshow("Resize", image)
      except:
        window_name = 'Image'
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (0, 0, 255)
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), (250, 0, 250), 3)
        cv2.namedWindow("Resize", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Resize", 800, 500)
        cv2.imshow("Resize", image)


  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cap.release()
