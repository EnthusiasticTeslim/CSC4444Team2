
import cv2
import mediapipe as mp
import numpy as np
import time

from rig_controller import RigController

# Mediapipe setup
mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
drawing_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)


class FaceAnimator():
    # cv2 capture
    _cap = None
    current_frame = None  # Will hold current video frame
    video_source = None # Webcam by default

    # Camera resolution
    cam_height = 1080
    cam_width = 1920

    # Numpy variables
    # NP arrays to hold rotation matrix
    rotation_vector = None
    translation_vector = None
    # Used for determining 3D rotation
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    # 3D model points.
    model_points = np.array([
        # Nose tip
        (0.0, 0.0, 0.0),
        # Chin
        (0.0, -330.0, -65.0),
        # Left eye left corner
        (-225.0, 170.0, -135.0),
        # Right eye right corne
        (225.0, 170.0, -135.0),
        # Left Mouth corner
        (-150.0, -150.0, -125.0),
        # Right mouth corner
        (150.0, -150.0, -125.0)
    ], dtype=np.float32)
    # Camera internals
    camera_matrix = np.array(
        [[cam_height, 0.0, cam_width/2],
         [0.0, cam_height, cam_height/2],
         [0.0, 0.0, 1.0]], dtype=np.float32
    )

    def __init__(self, video_source):
        if video_source is not None:
            self.video_source = video_sourcese
        lf.start_camera()
            

    def animate(self, rig_controller: RigController = None):
        self.get_current_frame()

        current_frameRGB = cv2.cvtColor(
            self.current_frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(current_frameRGB)

        if results.multi_face_landmarks:
            face_lms = results.multi_face_landmarks[0]
            shape = face_lms.landmark

            def get_2D_point(shape):
                ih, iw, ic = self.current_frame.shape
                return (int(shape.x * iw), int(shape.y * ih))

            # Will be used to determine head rotation using cv2.PnP()
            image_points = np.array([
                # Nose tip
                get_2D_point(shape[4]),
                # Chin
                get_2D_point(shape[175]),
                # Left eye left corner
                get_2D_point(shape[113]),
                # Right eye right corner
                get_2D_point(shape[446]),
                # Left Mouth corner
                get_2D_point(shape[57]),
                # Right mouth corner
                get_2D_point(shape[273])
            ], dtype=np.float32)

            # For testing purposes to calibrate image_points using indexes shown on image
            # Displayed in red
            for index, landmark in enumerate(shape):
                point = get_2D_point(landmark)
                x, y = point
                cv2.putText(
                    self.current_frame,
                    f'{index}',
                    (x, y),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 10, 225),
                    2
                )

            # Shows the selected landmarks
            for index, landmark in enumerate(image_points):
                x, y = int(landmark[0]), int(landmark[1])
                cv2.putText(
                    self.current_frame,
                    f'{index}',
                    (x, y),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 190, 225),
                    3
                )
                self.determine_head_rotation(image_points)
                self.show_3D_position(image_points)

                # Control rig to animate
                if rig_controller is not None:
                    rig_controller.control_bones(
                        face_shape=image_points,
                        rotation_vector=self.rotation_vector,
                        first_angle=self.first_angle
                    )

            # Optionally use mediapipe to draw landmarks
            # mpDraw.draw_landmarks(self.current_frame, face_lms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

        cv2.imshow('Face Detection', self.current_frame)
        cv2.waitKey(1)
        return {'PASS_THROUGH'}

    def determine_head_rotation(self, image_points: np.ndarray):
        # Refer to https://www.pythonpool.com/opencv-solvepnp/
        if self.rotation_vector is not None:
            (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(
                objectPoints=self.model_points,
                imagePoints=image_points,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                rvec=self.rotation_vector, tvec=self.translation_vector,
                useExtrinsicGuess=True
            )
        else:
            (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(
                objectPoints=self.model_points,
                imagePoints=image_points,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=False
            )

    # Shows the face direction
    def show_3D_position(self, image_points):
        nose_end_point2D, jacobian = cv2.projectPoints(
            np.array([(0., 0., 1000.)]),
            self.rotation_vector,
            self.translation_vector,
            self.camera_matrix,
            self.dist_coeffs
        )

        point1 = (
            int(image_points[0][0]),
            int(image_points[0][1])
        )

        point2 = (
            int(nose_end_point2D[0][0][0]),
            int(nose_end_point2D[0][0][1])
        )
        cv2.arrowedLine(self.current_frame, point1, point2, (5, 215, 255), 10)

    def start_camera(self):
        if isinstance(self.video_source, str):
            self._cap = cv2.VideoCapture(str(self.video_source))
        if isinstance(self.video_source, int): self._cap = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
        # TODO: Fix use aspect ratio instead??
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def get_current_frame(self):
        success, self.current_frame = self._cap.read()
        # Optimization
        self.current_frame.flags.writeable = False
        # TODO: Will need to resize image for standardization/consistency
        # Distorts image tho,
        self.current_frame = cv2.resize(
            self.current_frame,
            (self.cam_width, self.cam_height)
        )

    def end_session(self):
        cv2.destroyAllWindows()
        self._cap.release()
        self._cap = None
