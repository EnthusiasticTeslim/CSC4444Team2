import cv2

from face_animator import FaceAnimator

def test():
    faceanim = FaceAnimator(video_source="C:\\Users\\Me\\Desktop\\LSU\\2022\\Fall\\CSC 4444\\project\\CSC4444Team2\\footage\\video.mp4")
    while True:
        faceanim.animate()                 
        # Press esc or ctrl + c to quit 
        cv2.waitKey(0)
        if(cv2.waitKey(1) & 0xff == 27): 
            break
    faceanim.end_session()
    
if __name__ == "__main__":
    test()
