import os
import cv2
import pytesseract as tess
from os.path import splitext

input_path = "./samplex/"
output_path = "outputx/"

imageList = [os.path.join(input_path, x) for x in os.listdir(input_path)]

def remove_dup(croppedImageDimList):
    croppedImageDimList[:] = [i for i in croppedImageDimList if not any((i[0] > j[0]) and (i[2] <= j[2]) and (i[1] >= j[1]) and (i[3] <= j[3]) for j in croppedImageDimList if i is not j)]

def findSpeechBubbles(imagePath, method = 'simple'):
    image = cv2.imread(imagePath)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageGrayBlur = cv2.GaussianBlur(imageGray,(3,3),0)
    binary = cv2.threshold(imageGrayBlur,235,255,cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]

    croppedImageDimList = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 400 and w > 60 and h < 500 and h > 25:
            croppedImage = image[y:y+h, x:x+w]
            text = tess.image_to_string(croppedImage).replace('\r', ' ').replace("\n", " ")
            if len(text) >= 2:
                croppedImageDimList.append([x, y, x+w, y+h])

    remove_dup(croppedImageDimList)
    remove_dup(croppedImageDimList)

    return [image[y:y+h, x:x+w] for x, y, w, h in croppedImageDimList], croppedImageDimList

for i, image_path in enumerate(imageList):
    croppedImageList, croppedImageDimList = findSpeechBubbles(image_path)
    for j, dim in enumerate(croppedImageDimList):
        img_copy = cv2.imread(image_path)
        cv2.rectangle(img_copy, (dim[0], dim[1]), (dim[2], dim[3]), (51, 204, 51), 5)
        output_image = os.path.join(output_path, f"{splitext(os.listdir(input_path)[i])[0]}_{j}.jpg")
        cv2.imwrite(output_image, img_copy)
        print(f"Image \"{output_image}\" has been saved.. Number of Images Saved : {i+1}")
