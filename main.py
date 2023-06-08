from matplotlib import pyplot as plt
import cv2
import numpy as np
import pytesseract
import imutils 

def carplate_extract(image):   
    carplate_rects = carplate_haar_cascade.detectMultiScale(image,scaleFactor=1.1, minNeighbors=5)
    for x,y,w,h in carplate_rects: 
         carplate_img = image[y+5:y+h-5 ,x+5:x+w-1]
        
    return carplate_img

def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized_image

def segment_characters(image) :
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    dimensions = [LP_WIDTH/6, LP_WIDTH/2, LP_HEIGHT/10, 2*LP_HEIGHT/3]
    
    plt.imshow(img_binary_lp, cmap='gray')
    plt.show()
    cv2.imwrite(f"car_plate/{file_name}",img_binary_lp)

file_name = 'image1.png' 
img = cv2.imread(file_name) 
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)

carplate_haar_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml ')

carplate_overlay = img.copy() 
carplate_rects = carplate_haar_cascade.detectMultiScale(carplate_overlay,scaleFactor=1.1, minNeighbors=5)
for x,y,w,h in carplate_rects: 
    cv2.rectangle(carplate_overlay, (x,y), (x+w,y+h), (0,0,255), 5) 

detected_carplate_img = carplate_overlay
plt.imshow(cv2.cvtColor(detected_carplate_img, cv2.COLOR_BGR2RGB))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
carplate_extract_img = carplate_extract(gray)
image = enlarge_img(carplate_extract_img, 150)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
number_plate = pytesseract.image_to_string(f"car_plate/{file_name}", 
                                  config = f'--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')



number_plate = number_plate.strip()
number_plate = number_plate.replace(" ","").replace("S","5")


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(bfilter, 50, 100)
keypoints = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
        
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

font = cv2.FONT_HERSHEY_SIMPLEX
result = cv2.putText(img, text=number_plate, org=(approx[0][0][0], approx[1][0][1]-60), fontFace=font, fontScale=1, color=(0,0,255), thickness=3, lineType=cv2.LINE_AA)
result = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,0,255),3)
cv2.imwrite(f"result/{number_plate}.jpg",result)
cv2.imwrite(f"result/{number_plate}_plate.jpg",image)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))