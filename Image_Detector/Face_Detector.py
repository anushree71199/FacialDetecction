#pip install opencv-python
import cv2 
import numpy as np
import matplotlib.pyplot as plt
#All the modules required to be called
Img = plt.imread("./group.jpg")  #Input the image you want to check the detection on and make sure it is present in the same directory
plt.imshow(Img) #To display the image you entered
F_D= cv2.CascadeClassifier("haarcascade_frontalface_alt.xml") #basic file for frontal face detection
Faces = F_D.detectMultiScale(Img) 
Faces.shape #To view the number of faces available in your entered image
#the looping statement
for face in Faces:
    x,y,w,h = face
    Img = cv2.rectangle(Img, (x,y), (x+w,y+h), (255,0,0), 10 )

    plt.imshow(Img) #Command to view the image after applying the detecting operation 

    #Thank-You
#Happy-Coding