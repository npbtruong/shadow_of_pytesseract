import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from matplotlib import image
from os.path import splitext
from skimage.io import imread, imsave
from skimage.transform import resize
from tensorflow.python.keras import models
import pytesseract as tess


input_path = "./samplex/"
output_path = "outputx/"


imglist = os.listdir(input_path)
imageList = [(lambda x: input_path +x)(x) for x in imglist]


def remove_dup(croppedImageDimList):
    for j in croppedImageDimList:
        x1 = j[0]
        y1 = j[1]
        x2 = j[2]
        y2 = j[3]
        for i in croppedImageDimList:
            x3 = i[0]
            y3 = i[1]
            x4 = i[2]
            y4 = i[3]
            if (x1 > x3) and  (x2 <= x4) and (y1 >= y3) and (y2 <= y4):
                croppedImageDimList.remove(j)
                break

#Check if the text boxes are inside the given box

def inside_check(outer_box, inner_box_list):
    x1 = outer_box[0]
    y1 = outer_box[1]
    x2 = outer_box[2]
    y2 = outer_box[3]
    
    inside_boxes = []
    inside = False
    count = 0
    for i in inner_box_list:
        x3 = i[0]
        y3 = i[1]
        x4 = i[2]
        y4 = i[3]
        if (x1 < x3) and (x2>x4) and (y1<y3) and (y2>y4):
            inside = True
            count += 1
            inside_boxes.append(i)
            
    return inside, count, inside_boxes

#Get all text boxes in the img

def get_text_boxes(img):
    h, w, _ = img.shape # assumes color image

    # run tesseract, returning the bounding boxes
    boxes = pytesseract.image_to_boxes(img) # also include any config options you use

    text_boxes = []

    # draw the bounding boxes on the image
    for b in boxes.splitlines():
        b = b.split(' ')
        x1 = int(b[1])
        y1 = h - int(b[2])
        x2 = int(b[3])
        y2 = h - int(b[4])
        text_boxes.append([x1, y1, x2, y2])
    
    return text_boxes

# Tesseract OCR
def OCR(img):
    text = tess.image_to_string(img)
    tes =  text.replace('\r', ' ')
    tes = tes.replace("\n", " ")
    length = len(tes)
    return tes, length

# find all speech bubbles in the given comic page and return a list of cropped speech bubbles (with possible false positives)
def findSpeechBubbles(imagePath, method = 'simple'):
    color = (51, 204, 51) 
    # read image
    image = cv2.imread(imagePath)

    # gray scale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # filter noise
    imageGrayBlur = cv2.GaussianBlur(imageGray,(3,3),0)
    if method != 'simple':
        # recognizes more complex bubble shapes
        imageGrayBlurCanny = cv2.Canny(imageGrayBlur,50,500)
        binary = cv2.threshold(imageGrayBlurCanny,235,255,cv2.THRESH_BINARY)[1]
    else:
        # recognizes only rectangular bubbles
        binary = cv2.threshold(imageGrayBlur,235,255,cv2.THRESH_BINARY)[1]
    # find contours
    contours = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]

    contours = list(reversed(contours))
    # get the list of cropped speech bubbles
    croppedImageList = []
    croppedImageDimList = []
    for contour in contours:
        rect = cv2.boundingRect(contour)
        [x, y, w, h] = rect
        start_point = (x,y) 
        end_point = (x+w, y+h)
        # filter out speech bubble candidates with unreasonable size
       
        if w < 400 and w > 60 and h < 500 and h > 25:
            croppedImage = image[y:y+h, x:x+w]
            croppedImageDim = [x, y, x+w, y+h]
            
            tes, length = OCR(croppedImage)
            if length >=2 :
                croppedImageDimList.append(croppedImageDim)
            
            croppedImageList.append(croppedImage)

    remove_dup(croppedImageDimList)
    remove_dup(croppedImageDimList)
    
    for i in croppedImageDimList:
        img = cv2.rectangle(image, (i[0], i[1]), (i[2], i[3]), color, 5) 

    return croppedImageList, croppedImageDimList, img

for i, image_path in enumerate(imageList):
    croppedImageList, croppedImageDimList, _ = findSpeechBubbles(image_path)
    for j, dim in enumerate(croppedImageDimList):
        img_copy = cv2.imread(image_path)
        cv2.rectangle(img_copy, (dim[0], dim[1]), (dim[2], dim[3]), (51, 204, 51), 5)
        output_image = os.path.join(output_path, f"{splitext(imglist[i])[0]}_{j}.jpg")
        cv2.imwrite(output_image, img_copy)
        print(f"Image \"{output_image}\" has been saved.. Number of Images Saved : {i+1}")
