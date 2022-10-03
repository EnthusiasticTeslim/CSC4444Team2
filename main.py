import cv2

from face_animator import FaceAnimator

def test():
    faceanim = FaceAnimator(
        video_source='footage/video.mp4'
    )
    while True:
        faceanim.animate()
        # Press q or ctrl + c to quit 
        if(cv2.waitKey(1) & 0xff == ord("q")):
            break
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    test()
