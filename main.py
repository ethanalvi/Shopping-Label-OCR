from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np
import time
from cv2 import ROTATE_90_CLOCKWISE, rotate, resize, imread, findContours, threshold, boundingRect, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE,IMREAD_GRAYSCALE
from cv2.dnn import readNet, blobFromImage
from imutils.object_detection import non_max_suppression
from imutils.contours import sort_contours
import imutils


class Image(BaseModel):
    image_base64: str

app = FastAPI()

def resize_image(img):
    (H, W) = img.shape[:2]
    
    # Rotate image back if image is rotated (Android does this for some reason)
    if (H > W):
        img = rotate(img, ROTATE_90_CLOCKWISE)
        (H, W) = img.shape[:2]

    # Increase image size maintaining aspect ratio
    img = imutils.resize(img, width=320)
    (H, W) = img.shape[:2]
    H -= H%32
    img = resize(img, (W, H))
    return img



# Price Grabbing
def get_price(char_results, char_boxes_per_region, region_boxes):
    numbers = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    index = -1

    # Return "" if no text was found
    if char_results == []:
        return ""

    # Find Index of Price Region Using $ symbol
    price = ""
    for i, region in enumerate(char_results):
        if "$" in region:
           index = i

    # If no $ symbol found, find region by height
    if index == -1:
        index_ignore_list = []
        while(True):
            top_height = 0
            for i, region in enumerate(char_boxes_per_region):
                if not region == []:
                    if not i in index_ignore_list:
                        heights = []
                        for x, y, w, h in region: heights.append(h)
                        max_height = max(heights)
                        if (max_height > top_height): 
                            top_height = max_height
                            index = i

            # Ensure the tallest region is not just text
            has_number = max([number in char_results[index] for number in numbers])
            if has_number: break
            else:
                index_ignore_list.append(index)

    # Check if there are small numbers that represent cents
    if not '.' in char_results[index]:
        height = char_boxes_per_region[index][0][3]
        for i, (x, y, w, h) in enumerate(char_boxes_per_region[index]):
            if h/height < 0.75: # if the number is smaller by 75%
                char_results[index].insert(i, '.')
                break

    # Still no decimal, assume the OCR missed the decimal place
    if not '.' in char_results[index]:
        insert_index = len(char_results[index])-2
        char_results[index].insert(insert_index, '.')

    # Getting text from price region
    for char in char_results[index]:
        if not char == '$': price += char

    return price




@app.get("/")
async def read_root():
    return {"Hello": "World"}



@app.post("/price")
async def detect_price(image_string : Image):
    decodeit = open('./image.jpg', 'wb') 
    decodeit.write(base64.b64decode((image_string.image_base64))) 
    decodeit.close() 
    image = imread("./image.jpg")
    image_one_channel = imread("./image.jpg", IMREAD_GRAYSCALE)
    #image = cv2.resize(image, (1280, 768)) # NOTE Image shape must be in multiples of 32px. Our OCR is using 5:3 aspect ratio and will scale to 1280px x 768px
    #image_one_channel = cv2.resize(image_one_channel, (1280, 768))
    image = resize_image(image)
    image_one_channel = resize_image(image_one_channel)
    (H, W) = image.shape[:2]


    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]


    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = readNet("./frozen_east_text_detection.pb")
    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = blobFromImage(image, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()
    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))


    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.5: # THIS IS OUR MINIMUM CONFIDENCE VALUE PLEASE DON'T IGNORE THIS IS IMPORTANT
                continue
            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])




    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)


    # -- Cropping Text Regions --
    text_regions = []
    for i, (startX, startY, endX, endY) in enumerate(boxes):
        region = image_one_channel[startY:endY, startX:endX]
        text_regions.append(region)

    

    

    # -- Character Detection --  
    def get_char_shapes_from_image(image):
        # -- Perform image Preprocessing --

        # Thresh image
        ret, image_threshed = threshold(image, 150, 255, 0)

        # Negate image (change to white on black)
        image_threshed = abs(image_threshed - 255)


        # -- Find Location of Characters --

        # Find and Sort Contours of image
        cnts = findContours(image_threshed.copy(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sort_contours(cnts, method="left-to-right")[0]

        char_images = []
        char_boxes = []
        for c in cnts:
            (x, y, w, h) = boundingRect(c)
            # filter out bounding boxes that are too small
            if (w >= 10) and (h >= 10):
                # Crop image
                char_image = image[y:(y+h), x:(x+w)]

                # Prepare image for model input
                char_image = resize(char_image, (64, 64))
                #char_image = resize_image(char_image) # pad image and resize to 64x64
                #char_image = cv2.threshold(char_image, 150, 255, 0)[1] # Binarize the image
                char_image = np.array(char_image, dtype=np.float32) # cast to numpy array
                char_image = char_image/255                         
                char_images.append(char_image)
                char_boxes.append((x,y,w,h))
        return char_images, char_boxes




    # -- Detect Char Regions for each Text Region --
    char_images_per_region = []
    char_boxes_per_region = []
    for region in text_regions:
        char_images, char_boxes = get_char_shapes_from_image(region)
        char_images_per_region.append(char_images)
        char_boxes_per_region.append(char_boxes)


    # -- Predict Chars --

    from tensorflow.keras.models import load_model 

    # Load saved model
    model = load_model("./ocr_model.keras")

    character_list =['Y','Z','0','O','3','V','A','D','E','F','R','6','K','N','L','9','T','J','C','M','P','S','U','W','1','2','H','G','B','I','.','$','7','5','X','8','4','Q']

    # Send inputs through model
    char_results_per_region = []
    for region_char_images in char_images_per_region:
        if region_char_images != []:
            preds = model.predict(np.array(region_char_images), verbose=0)
            predicted_labels = np.argmax(preds, axis=1)
            predicted_labels = [character_list[pred] for pred in predicted_labels]
            #predicted_unicode = np.argmax(preds, axis=1) + 33
            #predicted_labels = [chr(code) for code in predicted_unicode]
            char_results_per_region.append(predicted_labels)
        else:
            char_results_per_region.append(['NA'])



    price = get_price(char_results_per_region, char_boxes_per_region, boxes)
    
    result = {"price": price, "chars": str(char_results_per_region)}
    return result
