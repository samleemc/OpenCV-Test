import cv2 as cv

def rescaleFrame(frame, scale=0.75):    
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    
    dimensions = (width,height)
    
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

def rectangleFrame(frame,color=(0,255,0)):
    return cv.rectangle(frame,(0,0),(frame.shape[1],frame.shape[0]),color,thickness=2)
 
def grayScale(frame):
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
def blurFrame(frame,ksize=(0,0)):
    return cv.GaussianBlur(frame,ksize,cv.BORDER_DEFAULT)
 
def cannyFrame(frame,scale):
    return cv.Canny(frame,scale,scale)

def resizeFrame(frame):
    cropped = frame[60:420,:]
    return cv.resize(cropped,(1280,720),interpolation=cv.INTER_LINEAR)
     

capture = cv.VideoCapture(0)

scale = int(input('Please provide scale\n'))

while True:
    isTrue, frame = capture.read()
    
    
    
    processed1Frame = cannyFrame(frame,scale)
    contours, hierarchies = cv.findContours(processed1Frame, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    processed2Frame = cv.cvtColor(processed1Frame, cv.COLOR_GRAY2BGR)
    processed3Frame = cv.putText(processed2Frame,f'{len(contours)}',(1,30),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),2)
    
    # cv.imshow('Video',frame)
    cv.imshow('Original',cv.putText(frame,f'{len(contours)}',(1,30),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),2))
    cv.imshow('Processed1',processed3Frame)
    
    if cv.waitKey(10) & 0xFF==ord('d'):
        break
    
capture.release()
cv.destroyAllWindows()