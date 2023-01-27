import cv2 as cv
import os 

file = 'sam.jpg'

path = os.path.dirname(__file__)

print(os.path.join(path,file))

x = cv.imread(os.path.join(path,file))

cv.imshow('sam',x)

cv.waitKey(0)