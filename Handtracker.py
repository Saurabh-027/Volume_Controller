import cv2
import time
import numpy as np
import HmodTrec as hmt
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

width_cam, height_cam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(width_cam, 1)
cap.set(height_cam, 2)
past_time = 0

detector = hmt.Hand_Detection(detectionCon=0.8)


device = AudioUtilities.GetSpeakers()
interface = device.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
print(volume.GetVolumeRange())
volRange = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(-0.5, None)
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400

while True:
    success, img = cap.read()
    img = detector.search_hands(img)
    Hand_List = detector.find_position_of_hands(img, draw=False)
    if len(Hand_List) != 0:
        # print(lmList[4],lmList[8])

        x1, y1 = Hand_List[4][1], Hand_List[4][2]
        x2, y2 = Hand_List[8][1], Hand_List[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        length = math.hypot(x2 - x2, y1 - y2)
        # print(length)

        # Hand Range min-50, max-300
        # Volume Range min - 65, max - 0

        vol = np.interp(length, [50, 220], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 10, (0, 0, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)

    current_time = time.time()
    fps = 1 / (current_time - past_time)
    past_time = current_time

    cv2.putText(img, f'fps:{int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (200, 34, 0), 3)

    cv2.imshow("Img", img)
    cv2.waitKey(1)
