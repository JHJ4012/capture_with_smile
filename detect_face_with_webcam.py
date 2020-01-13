import cv2
import numpy as np
from keras.models import load_model
from datetime import datetime

model = load_model('model/happy.h5')

face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
predict_array = []

while True:
    ret, frame = cap.read()
    predict_array.clear()
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_gray, 1.3,5)    #return 값 : 얼굴의 위치 (x, y, w, h)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x-20,y-20), (x+w+20, y+h+20), (255,0,0),2)
        roi_gray = image_gray[y-20:y+h+20, x-20:x+w+20]
        roi_gray = cv2.resize(roi_gray, (64,64))
        roi_gray = np.array(roi_gray)
        roi_gray = roi_gray.reshape(-1, 64, 64, 1)
        # print(roi_gray.shape)
        predict = model.predict(roi_gray)
        if predict >= 0.5:
            y_pred_logical = 1
        else:
            y_pred_logical = 0
        # print(predict)
        predict_array.append(y_pred_logical)

    if len(predict_array) != 0 and predict_array.count(1) == len(predict_array):
        print(predict_array)
        now = datetime.now().strftime("%d_%H-%M-%S")
        cv2.imwrite('photos/%s.jpg'%(now), frame)
    cv2.imshow('df', frame)

    k = cv2.waitKey(20)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
