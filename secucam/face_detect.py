import cv2 as cv
import os


capture = cv.VideoCapture(1)
capture2 = cv.VideoCapture('secucam/IMG_5035-HD.mov')

while True:
    isTrue, frame = capture2.read()
    
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    haar_cascade = cv.CascadeClassifier('secucam/haar_face.xml')
    eye_cascade = cv.CascadeClassifier('secucam/haar_eye.xml')
    
    faces = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=9)
    
    for (x,y,w,h) in faces:
        
        # cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
        face = gray[y:y+h,x:x+w]
        
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)
        
        eyes = eye_cascade.detectMultiScale(face,scaleFactor=1.1,minNeighbors=9,maxSize=(100,100))
        
        for (ex,ey,ew,eh) in eyes:
            
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
            cv.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(255,0,0),2)
    
    cv.putText(frame,f'{len(faces)}',(1,30),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),2)
    
    cv.imshow('Org',frame)
    
    if cv.waitKey(10) & 0xFF==ord('d'):
        break
    
capture.release()
cv.destroyAllWindows()