import os
import cv2
import pytesseract as tess
from os.path import splitext

input_path = "./imgs/"
output_path = "outputs/"
output_folder = "outputs2/"

imageList = [os.path.join(input_path, x) for x in os.listdir(input_path)]

def remove_dup(croppedImageDimList):
    croppedImageDimList[:] = [i for i in croppedImageDimList if not any((i[0] > j[0]) and (i[2] <= j[2]) and (i[1] >= j[1]) and (i[3] <= j[3]) for j in croppedImageDimList if i is not j)]

def findSpeechBubbles(imagePath, method = 'simple'):
    image = cv2.imread(imagePath)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageGrayBlur = cv2.GaussianBlur(imageGray,(3,5),0)
    binary = cv2.threshold(imageGrayBlur,235,255,cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]#RETR_TREE RETR_EXTERNAL

    croppedImageDimList = []
    for contour in reversed(contours):
        x, y, w, h = cv2.boundingRect(contour)
        

        if w < 800 and w > 25 and h < 1000 and h > 20:
        
            croppedImage = image[y:y+h, x:x+w]
            croppedImage = cv2.copyMakeBorder(croppedImage, 4, 4, 4, 4, cv2.BORDER_CONSTANT)

            # Preprocess the cropped image (example: convert to grayscale)
            gray = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)
            
            _, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
            # Use floodfill to change the outer white area to black
            h, w = thresh.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(thresh, mask, (0,0), 0)
            # Assuming 'thresh' is your image
            kernel = np.ones((2,2),np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Perform OCR on the preprocessed image
            text = tess.image_to_string(thresh, lang='eng',config='--psm 6').replace('\r', ' ').replace("\n", " ")
            print(text)
            if len(text) >= 3:
                # Display the original and grayscale images
                cv2.imshow('Grayscale Image', thresh)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                croppedImageDimList.append([x, y, x+w, y+h])

    remove_dup(croppedImageDimList)
    
    return [image[y:y+h, x:x+w] for x, y, w, h in croppedImageDimList], croppedImageDimList

for i, image_path in enumerate(imageList):
    croppedImageList, croppedImageDimList = findSpeechBubbles(image_path)
    for j, dim in enumerate(croppedImageDimList):
        img_copy = cv2.imread(image_path)
        cv2.rectangle(img_copy, (dim[0], dim[1]), (dim[2], dim[3]), (51, 204, 51), 5)
        output_image = os.path.join(output_path, f"{splitext(os.listdir(input_path)[i])[0]}_{j}.jpg")
        cv2.imwrite(output_image, img_copy)
