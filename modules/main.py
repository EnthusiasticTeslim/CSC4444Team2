import cv2

from face_animator import FaceAnimator

def test():
    faceanim = FaceAnimator(0)
    print(faceanim.video_source)
    while True:
        faceanim.animate()
        # Press esc or ctrl + c to quit 
        if(cv2.waitKey(1) & 0xff == 27): 
            break
    faceanim.end_session()
    
if __name__ == "__main__":
    test()
