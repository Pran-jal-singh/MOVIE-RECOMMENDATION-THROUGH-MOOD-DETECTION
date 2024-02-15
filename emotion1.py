import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.utils import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# load model
model = load_model(r'C:\Users\SRI SHIKA.L\Documents\E-RMS\final_model.h5')


face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface.xml')

i=0
GR_dict={0:(0,255,0),1:(0,0,255)}

model = tf.keras.models.load_model(r'C:\Users\SRI SHIKA.L\Documents\E-RMS\final_model.h5')
face_cascade = cv2.CascadeClassifier(r'C:\Users\SRI SHIKA.L\Documents\E-RMS\haarcascade_frontalface.xml')

output=[]
cap = cv2.VideoCapture(0)

while (i<=20):
    ret, img = cap.read()
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img,1.05,5)

    for x,y,w,h in faces:

        face_img = img[y:y+h,x:x+w] 

        resized = cv2.resize(face_img,(224,224))
        reshaped=resized.reshape(1, 224,224,3)/255
        predictions = model.predict(reshaped)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise')
        predicted_emotion = emotions[max_index]
        output.append(predicted_emotion)
            
            
            
        cv2.rectangle(img,(x,y),(x+w,y+h),GR_dict[1],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),GR_dict[1],-1)
        cv2.putText(img, predicted_emotion, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    i = i+1

    cv2.imshow('LIVE', img)
    key = cv2.waitKey(1)
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
print(output)
cap.release()
cv2.destroyAllWindows()
#final_output1 = st.mode(output)
import statistics as st
final_output = st.mode(output)
print(final_output)
