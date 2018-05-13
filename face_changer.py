import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

cap = cv2.VideoCapture(0)


def change_skin_color(img, ratio=0.65):
    height, width, depth = img.shape
    mask = np.zeros((height, width)).astype('uint8')
    cv2.rectangle(mask, (0, 0), (width, height), 255, thickness=-1)
    white_pic = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    new = cv2.addWeighted(img, ratio, white_pic, 1-ratio, 0)
    return new


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
    kernel = np.ones((5, 5), np.float32)/25
    dst = cv2.filter2D(img, -1, kernel)
    #dst = cv2.GaussianBlur(img, (5,5), 1.5)
    return dst


def whitening(img, rate=0.15):
    height, width, depth = img.shape
    mask = np.zeros((height, width)).astype('uint8')
    cv2.rectangle(mask, (0, 0), (width, height), 255, thickness=-1)
    patch_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    patch_hsv_temp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:]
    patch_hsv_temp[:, :, -1] = np.minimum(patch_hsv_temp[:, :, -1] +
                                          patch_hsv_temp[:, :, -1]*patch_mask[:, :, -1]*rate, 255).astype('uint8')

    img = cv2.cvtColor(patch_hsv_temp, cv2.COLOR_HSV2BGR)[:]
    return img


eye_x_ratio = 0
eye_y_ratio = 0
eye_length = 0
eye_length_list = []

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+int(h/2), x:x+w]
        roi_color = img[y:y+int(h/2), x:x+w]

        if eye_length > 0:
            eye_x = int(eye_x_ratio * w)
            eye_y = int(eye_y_ratio * h)
            half_w = int(w/2)

            new_x = 0
            another_x = 0
            new_y = 0
            if eye_x < half_w:
                new_x, new_y = eye_x + x, eye_y + y
                cv2.circle(img, (new_x, new_y),
                           eye_length, (255, 255, 255), -1)
                another_x = new_x

                another_eye_x = (half_w - eye_x) + half_w
                new_x, new_y = another_eye_x + x, eye_y + y
                cv2.circle(img, (another_eye_x+x, eye_y+y),
                           eye_length, (255, 255, 255), -1)

                line_y = ((y+h) - new_y)//2 + new_y
                line_x = new_x
                another_line_x = another_x
                cv2.line(img, (line_x, line_y), (line_x+w//2, line_y),
                         (255, 255, 255), thickness=2)
                cv2.line(img, (another_line_x, line_y), (another_line_x -
                                                         w//2, line_y), (255, 255, 255), thickness=2)

            elif eye_x > half_w:
                new_x, new_y = eye_x + x, eye_y + y
                cv2.circle(img, (new_x, new_y),
                           eye_length, (255, 255, 255), -1)
                another_x = new_x

                another_eye_x = half_w - (eye_x - half_w)
                new_x, new_y = another_eye_x + x, eye_y + y
                cv2.circle(img, (new_x, new_y),
                           eye_length, (255, 255, 255), -1)

                line_y = ((y+h) - new_y)//2 + new_y
                line_x = another_x
                another_line_x = new_x
                cv2.line(img, (line_x, line_y), (line_x+w//2, line_y),
                         (255, 255, 255), thickness=2)
                cv2.line(img, (another_line_x, line_y), (another_line_x -
                                                         w//2, line_y), (255, 255, 255), thickness=2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'yingshaoxo', (x, y-h//5), font,
                    1, (255, 255, 255), 1, cv2.LINE_AA)

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye_x_ratio = (ex+(ew/2))/w
            eye_y_ratio = (ey+(eh/2))/h

            eye_length = int(ew/2)
            eye_length_list.append(eye_length)
            eye_length = int(np.mean(eye_length_list))
            if len(eye_length_list) > 100:
                eye_length_list = eye_length_list[int(len(eye_length_list)/2):]

            # cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

    """
    # 美白方案1 , faster
    img = change_skin_color(img)
    img = whitening(img)
    img = cv2.bilateralFilter(img, 15, 35, 35)
    img = face_smoothing(img)
    img = increase_brightness(img)
    """

    # 美白方案2, slower
    img = change_skin_color(img, ratio=0.85)
    img = increase_brightness(img)
    img = cv2.edgePreservingFilter(img, flags=1, sigma_s=200, sigma_r=0.1)
    #img = cv2.stylization(img)
    #img = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)[0]

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
