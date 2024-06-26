import cv2
import numpy as np
from time import sleep

confThreshold=0.7
nmsThreshold=0.5
whT=320

classesFile='labels'
classNames=[]

with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration= 'yolov3.cfg'
modelWeights='yolov3.weights'

net=cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findobjects(output_list,img):
    hT,wT,cT=img.shape
    bbox=[]
    classIds=[]
    confs=[]

    for output in output_list:
        for detection in output:
            scores=detection[4:]
            classID=np.argmax(scores)
            confidence=scores[classID]
            if confidence>confThreshold:
                w,h=int(detection[2]*wT),int(detection[3]*hT)
                x,y=int((detection[0]*wT)-w/2),int((detection[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classID)
                confs.append(float(confidence))
    indices=cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    print("cars in this image are",len(indices))
    print("confidence score for this detection is",min(confs))
    return indices, bbox, classIds, confs

def draw_bbox(img, indices, bbox, classIds, confs, classNames, resize_factor=0.5):
    resized_img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)

    colors = np.random.uniform(0, 255, size=(len(classNames), 3))
    for i in indices:
        i = i
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        x, y, w, h = int(x * resize_factor), int(y * resize_factor), int(w * resize_factor), int(h * resize_factor)
        color = colors[classIds[i]]
        cv2.rectangle(resized_img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(resized_img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imshow("Detected Image", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def take_images():
    images = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
    indices_lengths = []

    for img_path in images:
        img = cv2.imread(img_path)

        blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)

        layerNames = net.getLayerNames()
        net.getUnconnectedOutLayers()
        outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        output_list = list(outputs)
        indices, bbox, classIds, confs = findobjects(output_list, img)
        draw_bbox(img, indices, bbox, classIds, confs, classNames)
        indices_lengths.append((len(indices), img_path))

    return indices_lengths

indices_lengths = take_images()
print(indices_lengths)
