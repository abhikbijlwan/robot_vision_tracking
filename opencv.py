import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)

first_frames = []
t = 0
while t < 100:
    ret, frame = cap.read()
    first_frames.append(frame)
    time.sleep(0.1)
    t += 1

background_img = np.mean(first_frames, axis=0).astype(np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    absdiff = cv2.absdiff(frame, background_img) #replace frame with disparity image
    cv2.imwrite("absdiff.png", absdiff)

    gray = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (13, 13), 0)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel) #dilation followed by erosion of image
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) #erosion followed by dilation
    _, fg_mask = cv2.threshold(closing, 34, 255, cv2.THRESH_BINARY) #25 is the sensitivity
    contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #drawing contours for only a specific size
    for contour in contours:
        if cv2.contourArea(contour) > 500:  #adjust threshold for object size, smaller for more noticeable contours 
            epsi = 0.0003
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsi * peri, True)

            m_array = cv2.moments(approx)
            cX = int(m_array['m10'] / m_array['m00'])
            cY = int(m_array['m01'] / m_array['m00'])
            text = 'X: ' + str(cX) + ', Y: ' + str(cY)

            cv2.drawContours(frame, [approx], -1, (0, 0, 255), 2)
            cv2.circle(frame, (cX, cY), 3, (255, 255, 255), -1)


            cv2.putText(frame, text, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) #

            # for points in contour:
            #     x, y = points[0]
            #     text = 'X: ' + str(np.int32(x)) + ', Y: ' + str(np.int32(y))
            #     # print(x.dtype)
            #     # print(y.dtype)

    #displaying images
    cv2.imshow("Orig", frame)
    cv2.imshow('Thresh', fg_mask)

    #break loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()