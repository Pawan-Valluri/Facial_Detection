import cv2
from facenet_pytorch import MTCNN
import numpy as np


    


class Face_detect(object):

    def __init__(self,mtcnn):
        self.mtcnn = mtcnn

    def draw(self,frame,boxes, probs, landmarks):
        try:
            for box, prob , landmark in zip(boxes, probs, landmarks):
                cv2.rectangle(frame, (box[0], box[1]),(box[2], box[3]), (255,0,0))
                # cv2.putText(frame, str(prob), (box[2], box[3]),  cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0),1,cv2.LINE_AA)

                # cv2.circle(frame, tuple(landmark[0]), 4, (255, 0, 0),-1)
                # cv2.circle(frame, tuple(landmark[1]), 4, (255, 0, 0),-1)
                # cv2.circle(frame, tuple(landmark[2]), 4, (255, 0, 0),-1)
                # cv2.circle(frame, tuple(landmark[3]), 4, (255, 0, 0),-1)
                # cv2.circle(frame, tuple(landmark[4]), 4, (255, 0, 0),-1)

        except:
            pass

        return frame


    def run(self):
        cap = cv2.VideoCapture(0)

        while True :
            ret, frame = cap.read()
            

            try:
                boxes, prob, landmarks = self.mtcnn.detect(frame, landmarks = True)
                self.draw(frame,boxes, prob, landmarks)

            
            except:
                pass
            
            cv2.imshow("capture", frame)
            if cv2.waitKey((1)) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

mtcnn = MTCNN()
fcd = Face_detect(mtcnn)
fcd.run()