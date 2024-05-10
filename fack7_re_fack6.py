import os
import string
import cv2
import easyocr
import numpy as np
from os.path import splitext

from pyt2s.services import stream_elements

input_path = "./imgs/"
output_path = "outputs/"

imageList = [os.path.join(input_path, x) for x in os.listdir(input_path)]

def remove_dup(croppedImageDimList , tts):
    if len(croppedImageDimList) == 0:
        return []
    
    working_list = croppedImageDimList.copy()
    working_list.sort(key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)

    i = 0
    while i < len(working_list):
        box = working_list[i]
        j = i + 1
        while j < len(working_list):
            other_box = working_list[j]
            xA = max(box[0], other_box[0])
            yA = max(box[1], other_box[1])
            xB = min(box[2], other_box[2])
            yB = min(box[3], other_box[3])
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            boxAArea = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
            boxBArea = (other_box[2] - other_box[0] + 1) * (other_box[3] - other_box[1] + 1)
            iou = interArea / float(boxAArea + boxBArea - interArea)
            if iou > 0.2:
                del working_list[j]
            else:
                j += 1
        i += 1
 
    new_croppedImageDimList = []
    new_tts = []
    for i, item in enumerate(croppedImageDimList):
        if item in working_list:
            new_croppedImageDimList.append(item)
            new_tts.append(tts[i])

    return new_croppedImageDimList, new_tts

def findSpeechBubbles(imagePath, method = 'simple'):
    image = cv2.imread(imagePath)
    image_copy = image.copy()
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageGrayBlur = cv2.GaussianBlur(imageGray,(3,5),0)
    binary = cv2.threshold(imageGrayBlur,235,255,cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]

    tts = []
    croppedImageDimList = []
    for contour in reversed(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        if w < 800 and w > 25 and h < 1000 and h > 20:
            croppedImage = image[y+4:y+h-4, x+4:x+w-4]
            croppedImage = cv2.copyMakeBorder(croppedImage, 4, 4, 4, 4, cv2.BORDER_CONSTANT)
            gray = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
            h, w = thresh.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(thresh, mask, (0,0), 0)
            kernel = np.ones((2,2),np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            reader = easyocr.Reader(['en'])
            results = reader.readtext(thresh)
            texts = [result[1] for result in results]
            text = ' '.join(texts)
            if len(text) >= 3:
                tts.append(text)
                croppedImageDimList.append([x, y, x+w, y+h])

    
    croppedImageDimList , tts = remove_dup(croppedImageDimList , tts)
    
    return croppedImageDimList , tts

for i, image_path in enumerate(imageList):
    croppedImageDimList , tts = findSpeechBubbles(image_path)
    for j, dim in enumerate(croppedImageDimList):
        img_copy = cv2.imread(image_path)
        cv2.rectangle(img_copy, (dim[0], dim[1]), (dim[2], dim[3]), (51, 204, 51), 5)
        output_image = os.path.join(output_path, f"{splitext(os.listdir(input_path)[i])[0]}_{j}.jpg")
        cv2.imwrite(output_image, img_copy)

        obj = stream_elements.StreamElements()
        format_tts = tts[j].translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).lower()
        print(format_tts)
        data = obj.requestTTS(format_tts, 'Kendra')
        output_audio = os.path.join(output_path, f"{splitext(os.listdir(input_path)[i])[0]}_{j}.mp3")
        with open(output_audio, '+wb') as file:
            file.write(data)
