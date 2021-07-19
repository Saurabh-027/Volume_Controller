import cv2
import mediapipe as mp
import time


class Hand_Detection:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectConnection = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectConnection, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def search_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(img_rgb)
        # print(result.multi_hand_landmarks)

        if self.result.multi_hand_landmarks:
            for hand_print in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_print, self.mpHands.HAND_CONNECTIONS)

        return img

    def find_position_of_hands(self, img, hand_no=0, draw=True):
        lmList = []
        if self.result.multi_hand_landmarks:
            hand = self.result.multi_hand_landmarks[hand_no]
            for id_of_hand, lm in enumerate(hand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w, ), int(lm.y * h)
                # print(id,cx,cy)
                lmList.append([id_of_hand, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return lmList


def main():
    past_Time = 0
    cap = cv2.VideoCapture(0)
    detector = Hand_Detection()
    while True:
        suc, img = cap.read()
        img = detector.search_hands(img)
        hand_list = detector.find_position_of_hands(img)
        if len(hand_list) != 0:
            print(hand_list[4])

        current_time = time.time()
        fps = 1 / (current_time - past_Time)
        past_Time = current_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3,
                    (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
