import cv2
import os
from face_animator import FaceAnimator


def test():
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    video_file = os.path.join(script_dir, "../footage/video.mp4")
    faceanim = FaceAnimator(video_source=video_file)
    while True:
        faceanim.animate()
        # Press esc or ctrl + c to quit
        if(cv2.waitKey(1) & 0xff == 27):
            break
        # cv2.waitKey(0)

    faceanim.end_session()


if __name__ == "__main__":
    test()
