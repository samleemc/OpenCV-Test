import numpy as np
import cv2 as cv
import os

directory = os.path.dirname(__file__)

capture = cv.VideoCapture(1)
# capture = cv.VideoCapture('/Users/sam/Downloads/python/OpenCV-Test/secucam/IMG_5035-HD.mov')



# Load face detector and recognizer
fd_model = os.path.join(directory,"face_detection_yunet_2022mar.onnx")
faceDetector = cv.FaceDetectorYN_create(fd_model,"",(0,0))

fr_model = os.path.join(directory,"face_recognition_sface_2021dec.onnx")
recognizer = cv.FaceRecognizerSF.create(fr_model,"")

# writer= cv.VideoWriter('basicvideo.mp4', cv.VideoWriter_fourcc(*'mp4v'), 10, (1920,1080))

targets = ['mum.jpg','jayden.jpg','sam.jpg','nicole.jpg']
targets_align = {}
targets_feature = {}
# Get target faces feature
for target in targets:
    name = target[:-4]
    target_image = cv.imread(os.path.join(directory,f'{target}'))
    height, width, _ = target_image.shape
    faceDetector.setInputSize((width, height))
    result, target_face = faceDetector.detect(target_image)
    if target_face is None:
        print(f'{target}' + ' does not have a face')
        exit()
    
    targets_align[name] = recognizer.alignCrop(target_image,target_face)
    targets_feature[name] = recognizer.feature(targets_align[target[:-4]])
    

while True:
    result, image = capture.read()
    
    if result is False:
        break
    
    height, width, _ = image.shape
    faceDetector.setInputSize((width, height))
    
    result, faces = faceDetector.detect(image)
    faces = faces if faces is not None else []
    
    
    
    for face in faces:
        isMatch = 0
        face_align = recognizer.alignCrop(image, face)
        face_feature = recognizer.feature(face_align)
        
        box = list(map(int, face[:4]))
        thickness = 2
        position = (box[0], box[1] - 10)
        font = cv.FONT_HERSHEY_SIMPLEX
        scale = 2
        thickness = 2
        
        for name,feature in targets_feature.items():
            cosine_score = recognizer.match(face_feature, feature, cv.FaceRecognizerSF_FR_COSINE)
            l2_score = recognizer.match(face_feature, feature, cv.FaceRecognizerSF_FR_NORM_L2)
            

            if cosine_score > 0.363 or l2_score < 1.128:
                isMatch += 1
                isMatch_name = name
                
                
                
                
        if isMatch != 1:
            color = (0, 0, 255)
            cv.rectangle(image, box, color, thickness, cv.LINE_AA)
            cv.putText(image, "Unknown", position, font, scale, color, thickness, cv.LINE_AA)
        else:
            color = (0, 255, 0)
            cv.rectangle(image, box, color, thickness, cv.LINE_AA)
            cv.putText(image, isMatch_name, position, font, scale, color, thickness, cv.LINE_AA)
                

    # writer.write(image)
    cv.imshow('test',image)
    
    key = cv.waitKey(1)
    if key == 27:
        break
 
# writer.release()   
capture.release()
cv.destroyAllWindows()