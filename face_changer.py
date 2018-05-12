import cv2
import numpy as np

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def face_smoothing(img):
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(img,-1,kernel)
    return dst

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

cap = cv2.VideoCapture(1)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        #cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+int(h/2), x:x+w]
        roi_color = img[y:y+int(h/2), x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.circle(roi_color, (int(ex+(ew/2)), int(ey+(eh/2))+3), int(ew/2)-2, (255,255,255), -1)

            adds = ey // 5 
            eye_gray = roi_gray[ey-adds:ey+eh+adds, ex:ex+ew]
            eye_color = roi_color[ey-adds:ey+eh+adds, ex:ex+ew]
            ret, mask = cv2.threshold(eye_gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
            indices = np.where(mask==255)
            eye_color[indices[0], indices[1], :] = [255, 255, 255]
            
    img = increase_brightness(img)
    img = face_smoothing(img)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


