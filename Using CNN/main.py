import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr

import util

sys.stderr = open(os.devnull, 'w')

#define constants
model_cfg_path = os.path.join('.', 'model', 'cfg', 'darknet-yolov3.cfg')
model_weights_path = os.path.join('.', 'model', 'weights', 'model.weights')
class_names_path = os.path.join('.', 'model', 'class.names')

input_dir = 'C:/Users/abhishek/OneDrive/Documents/ML/CV/Sem_Project/main/selfmain/yolov3/data'

#pykit learn lib---use 4 differnt clasifers...print 4 graphs-----print tyhe acuracy of each clasifier....and compare them

#load class names
with open(class_names_path, 'r') as f:
    class_names = [j[:-1] for j in f.readlines() if len(j.strip()) > 2]
    f.close()

#load image
net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)

    #load image

    img = cv2.imread(img_path)

    H, W, _ = img.shape

    # convert image
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

    # convert image
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

    # get detections
    net.setInput(blob)

    detections = util.get_outputs(net)

    # bboxes, class_ids, confidences
    bboxes = []
    class_ids = []
    scores = []

    for detection in detections:
        # [x1, x2, x3, x4, x5, x6, ..., x85]
        bbox = detection[:4]

        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]
        bbox_confidence = detection[4]

        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])

        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)

    # apply nms
    bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

    #plot
    reader = easyocr.Reader(['en'])
    for bbox_, bbox in enumerate(bboxes):
        xc, yc, w, h = bbox
        
        license_plate = img[int(yc - (h/2)):int(yc + (h/2)), int(xc - (w/2)):int(xc + (w/2)), :].copy()

        img = cv2.rectangle(img,
                            (int(xc - (w/2)), int(yc - (h/2))),
                            (int(xc + (w/2)), int(yc + (h/2))),
                            (0, 255, 0),
                            10)
        
        license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)

        _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)
        
        output = reader.readtext(license_plate_gray)
        
        for out in output:
            text_bbox, text, text_score = out
            if text_score > 0.4:
                print(text, text_score)

    fig, axs = plt.subplots(2, 2, figsize=(15, 15))  # 2 rows, 2 columns

    # Show the image with bounding boxes
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Image with Bounding Boxes")
    axs[0, 0].axis('off')  # Turn off axis

    # Show the license plate
    axs[0, 1].imshow(cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title("License Plate")
    axs[0, 1].axis('off')  # Turn off axis

    # Show the grayscale license plate
    axs[1, 0].imshow(license_plate_gray, cmap='gray')
    axs[1, 0].set_title("License Plate (Grayscale)")
    axs[1, 0].axis('off')

    # Show the thresholded license plate
    axs[1, 1].imshow(license_plate_thresh, cmap='gray')
    axs[1, 1].set_title("License Plate (Thresholded)")
    axs[1, 1].axis('off')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
