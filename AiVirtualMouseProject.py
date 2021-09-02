import mediapipe as mp
import HandTrackingModule as htm
import cv2
import numpy as np
import time
import autopy
###########################################
wcam, hcam = 680, 480
framer = 50
smoothening = 5

###########################################
cap = cv2.VideoCapture(0)
cap.set(3,wcam)
cap.set(3,hcam)

ptime =0
plocx, plocy = 0,0
clocx, clocy = 0,0
wscr, hscr = autopy.screen.size()
detector = htm.handDetector(maxHands=1, detectioncon=0.6)
while True:
    # 1. Find the hand landmarks
    success, img = cap.read()
    img = detector.findhands(img)
    lmlist, bbox = detector.findposition(img)

    # 2. Get the tip of index and middle finger
    if len(lmlist)!=0:
        # coordinates of our hand

        x1, y1 = lmlist[8][1:]
        # coordinates of index finger
        x2, y2 = lmlist[12][1:]
        print(x1, y1, x2, y2)
        # 3. check which finger are up
        fingers = detector.fingersUp()
        print(fingers)

        cv2.rectangle(img, (framer, framer), (wcam - framer, hcam - framer), (255, 0, 255), 2)
        # 4. Only index Finger : moving mode
        if fingers[1]==1 and fingers[2]==0:

            # 5. Convert the coordinates
            x3 = np.interp(x1, (framer,wcam-framer),(0,wscr))
            y3 = np.interp(y1, (framer,hcam),(0,hscr))
            # 6. smoothen value
            # 7. move mouse
            clocx = plocx+(x3-plocx)/smoothening
            clocy = plocy+(y3-plocy)/smoothening
            autopy.mouse.move(wscr-clocx,clocy)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            plocx, plocy = clocx, clocy
        # 8 . Both index and middle fingers are up: clicking mode
        if fingers[1]==1 and fingers[2]==1:
            length , img, lineinfo = detector.findDistance(8,12,img)
            print(length)
            # 10. click mouse if the distance short
            if length <30:
                cv2.circle(img,(lineinfo[4],lineinfo[5]),15,(0,255,0),cv2.FILLED)
                autopy.mouse.click()

        # 9. find distance between fingers

        # 11. to check the frame rate
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_COMPLEX, 2,(240,0,0),2)
    cv2.imshow('image',img)
    cv2.waitKey(1)
