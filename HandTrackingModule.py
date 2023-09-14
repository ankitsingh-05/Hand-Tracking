#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, max_hands=2, modelComplexity=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.modelComplex = modelComplexity
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.mp_hands = mp.solutions.hands
        # The four parameters of the hands() are
        # 1. static_image_mode (it tracks and detects, when false it sometimes detects and sometimes tracks)
        # 2. max_num_hands
        # 3. min_detection_confidence
        # 4. min_tracking_confidence
    
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.modelComplex , self.detection_confidence, self.tracking_confidence)

        # method to draw points
        self.mp_draw = mp.solutions.drawing_utils
        
    def findHands(self, image, draw=True):
        # converting bgr image to rgb
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # get the image and let the hands() process the image
        self.results = self.hands.process(rgb_img)

        # print(results.multi_hand_landmarks)

        # when my hand is in the frame
        if self.results.multi_hand_landmarks:
            # iterating through each landmark
            for hand_landmarks in self.results.multi_hand_landmarks:
                for ID, lm in enumerate(hand_landmarks.landmark):
                    if draw:
                        self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return image
    
    def findPosition(self, image, handNo=0, draw=True):
        
        lm_list = []
        # when my hand is in the frame
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for ID, lm in enumerate(myHand.landmark):
                # print(ID, lm)
                # height, width and channel of our image
                h,w,c = image.shape
                # centre of each landmark point
                cx = int(lm.x*w)
                cy = int(lm.y*h)
                # print(ID, cx, cy)
                lm_list.append([ID,cx,cy])
                if draw:
                    cv2.circle(image, (cx,cy), 13, (255,0 ,255), cv2.FILLED)
    
        return lm_list

# vid.release()
# cv2.destroyAllWindows()


def main():
    pTime=0
    cTime=0
    vid = cv2.VideoCapture(0)
    detector = HandDetector()
    
    while True:
        ret, image = vid.read()
        image = detector.findHands(image)
        lm_list = detector.findPosition(image)
        if len(lm_list) !=0:
            print(lm_list[4])
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
    
        cv2.putText(image, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow('Hand Tracking', image)
    
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




