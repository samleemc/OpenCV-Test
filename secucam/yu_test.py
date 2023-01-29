import numpy as np
import cv2 as cv
import os
import datetime
from pathlib import Path

directory = os.path.dirname(__file__)

capture = cv.VideoCapture(1)
# capture = cv.VideoCapture('/Users/sam/Downloads/python/OpenCV-Test/secucam/IMG_5035-HD.mov')

# Load face detector and recognizer
fd_model = os.path.join(directory,"face_detection_yunet_2022mar.onnx")
faceDetector = cv.FaceDetectorYN_create(fd_model,"",(0,0))

fr_model = os.path.join(directory,"face_recognition_sface_2021dec.onnx")
recognizer = cv.FaceRecognizerSF.create(fr_model,"")

# writer= cv.VideoWriter('basicvideo.mp4', cv.VideoWriter_fourcc(*'mp4v'), 20, (1920,1080))

targets = ['mum.jpg','jayden.jpg','sam.jpg','nicole.jpg','sam2.jpg','johnny.jpg']
targets_align = {}
targets_feature = {}
# Get target faces feature
for target in targets:
    filepath = Path(__file__).parents[1] / 'data/faces' / f'{target}'
    name = target[:-4]
    target_image = cv.imread(str(filepath))
    height, width, _ = target_image.shape
    faceDetector.setInputSize((width, height))
    result, target_face = faceDetector.detect(target_image)
    if target_face is None:
        print(f'{target}' + ' does not have a face')
        exit()
    
    targets_align[name] = recognizer.alignCrop(target_image,target_face)
    targets_feature[name] = recognizer.feature(targets_align[target[:-4]])
    
start_time = []
end_time = []

while True:
    start_time.append(datetime.datetime.now())

    result, original_image = capture.read()
    
    if result is False:
        break
    
    # !!!!!!!!!!!!!!!!!!!!!!
    original_height, original_width, _ = original_image.shape
    scale = 0.3
    resize_height = int(original_height*scale)
    resize_width = int(original_width*scale)
    
    image = cv.resize(original_image,(resize_width,resize_height),interpolation=cv.INTER_AREA)
    
    height, width, _ = image.shape
    faceDetector.setInputSize((width, height))
    
    result, faces = faceDetector.detect(image)
    faces = faces if faces is not None else []
    
    for face in faces:
        start_face_time = datetime.datetime.now()
        isMatch = 0
        isMatch_name = {}
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
            

            if cosine_score > 0.4: # or l2_score < 1.128
                isMatch += 1
                isMatch_name[name] = 'C = ' + f'{cosine_score}' + ' , L2 = ' + f'{l2_score}'
                
                
                
                
        if isMatch == 0:
            color = (0, 0, 255)
            cv.rectangle(image, box, color, thickness, cv.LINE_AA)
            cv.putText(image, "Unknown", position, font, scale, color, thickness, cv.LINE_AA)

        if isMatch == 1:
            color = (0, 255, 0)
            cv.rectangle(image, box, color, thickness, cv.LINE_AA)
            cv.putText(image, list(isMatch_name.keys())[0], position, font, scale, color, thickness, cv.LINE_AA)
            
        if isMatch > 1:
            color = (255, 0, 0)
            cv.rectangle(image, box, color, thickness, cv.LINE_AA)
            cv.putText(image, "Multi-Face-Recognized", position, font, scale, color, thickness, cv.LINE_AA)
            for name, score in isMatch_name.items():
                print(name + " " + score) 
        
        end_face_time = datetime.datetime.now()
        face_duration = end_face_time - start_face_time
        # print('face duration = ' + str(face_duration.microseconds))

        
    
    
    # writer.write(original_image)
    start_render_time = datetime.datetime.now()
    cv.imshow('test',image)
    end_render_time = datetime.datetime.now()
    render_duration = end_render_time - start_render_time
    # print('render duration = ' + str(render_duration.microseconds))
    
    end_time.append(datetime.datetime.now())
    
    duration = (end_time[-1]-start_time[0])
    if duration.seconds*100000 + duration.microseconds > 1000000:
        fps_list = []
        for i in range (0,len(end_time)-1,1):
            fps_list.append(1000000//(end_time[i] - start_time[i]).microseconds)
        fps = sum(fps_list)/len(fps_list)
        print('Capable FPS= 'f'{round(fps)}', 'Actual FPS = 'f'{len(fps_list)}')
        start_time.clear()
        end_time.clear()
        
                
    
    key = cv.pollKey()
    if key == 27:
        break
    
    # print(str((end_time - start_time).microseconds//1000) + 'ms')
 
# writer.release()   
capture.release()
cv.destroyAllWindows()