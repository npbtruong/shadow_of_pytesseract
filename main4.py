import os
import cv2
import numpy as np
import pytesseract as tess
from os.path import splitext


input_path = "./imgs/"
output_path = "outputs/"

imageList = [os.path.join(input_path, x) for x in os.listdir(input_path)]

def non_max_suppression_large(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(area)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

def findSpeechBubbles(imagePath, method = 'simple'):
    image = cv2.imread(imagePath)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageGrayBlur = cv2.GaussianBlur(imageGray,(3,5),0)
    binary = cv2.threshold(imageGrayBlur,235,255,cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]

    rects = []
    for contour in reversed(contours):
        x, y, w, h = cv2.boundingRect(contour)

        # Check aspect ratio and area
        aspect_ratio = w / float(h)
        if w < 800 and w > 25 and h < 1000 and h > 20 and aspect_ratio > 0.5 and aspect_ratio < 2:
            rects.append([x, y, x+w, y+h])

    # Apply non-maxima suppression to the bounding boxes
    rects = np.array(rects)
    pick = non_max_suppression_large(rects, overlapThresh=0.2)

    croppedImageDimList = []

    for (startX, startY, endX, endY) in pick:
        croppedImage = image[startY:endY, startX:endX]
        croppedImage = cv2.copyMakeBorder(croppedImage, 4, 4, 4, 4, cv2.BORDER_CONSTANT)

        # Preprocess the cropped image
        gray = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
        h, w = thresh.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(thresh, mask, (0,0), 0)
        kernel = np.ones((2,2),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = 255 - thresh

        # Perform OCR on the preprocessed image
        text = tess.image_to_string(thresh, lang='eng',config='--psm 6').replace('\r', ' ').replace("\n", " ")

        if len(text) >= 3:
            
            print(text)
            cv2.imshow('Grayscale Image', thresh)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            croppedImageDimList.append([startX, startY, endX, endY])

    return [image[y:y+h, x:x+w] for x, y, w, h in croppedImageDimList], croppedImageDimList

for i, image_path in enumerate(imageList):
    croppedImageList, croppedImageDimList = findSpeechBubbles(image_path)
    for j, dim in enumerate(croppedImageDimList):
        img_copy = cv2.imread(image_path)
        cv2.rectangle(img_copy, (dim[0], dim[1]), (dim[2], dim[3]), (51, 204, 51), 5)
        output_image = os.path.join(output_path, f"{splitext(os.listdir(input_path)[i])[0]}_{j}.jpg")
        cv2.imwrite(output_image, img_copy)
