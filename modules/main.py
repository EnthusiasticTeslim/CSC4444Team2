import cv2

from face_animator import FaceAnimator

def test():
    faceanim = FaceAnimator(video_source=0)
    while True:
        faceanim.animate()                 
        # Press esc or ctrl + c to quit 
        if(cv2.waitKey(1) & 0xff == 27): 
            break
        # cv2.waitKey(0)
        
    faceanim.end_session()
    
if __name__ == "__main__":
    test()
