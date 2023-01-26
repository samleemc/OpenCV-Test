import cv2 as cv

capture = cv.VideoCapture(0)
capture2 = cv.VideoCapture('IMG_5035-HD.mov')

while True:
    isTrue, frame = capture2.read()
    
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    haar_cascade = cv.CascadeClassifier('haar_face.xml')
    eye_cascade = cv.CascadeClassifier('haar_eye.xml')
    
    faces = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
    
    for (x,y,w,h) in faces:
        
        # cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
        face = gray[y:y+h,x:x+w]
        
        eyes = eye_cascade.detectMultiScale(face,scaleFactor=1.1,minNeighbors=6,maxSize=(50,50))
        
        for (ex,ey,ew,eh) in eyes:
            
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(255,0,0),1)
    
    cv.putText(frame,f'{len(faces)}',(1,30),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),2)
    
    cv.imshow('Org',frame)
    
    if cv.waitKey(10) & 0xFF==ord('d'):
        break
    
capture.release()
cv.destroyAllWindows()