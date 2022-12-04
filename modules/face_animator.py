
import cv2
import mediapipe as mp
import numpy as np
import time

from rig_controller import RigController

# Mediapipe setup
mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.9
)
drawing_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)


class FaceAnimator():
    # cv2 capture
    _cap = None
    current_frame = None  # Will hold current video frame
    video_source = None  # Webcam by default

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
            self.video_source = video_source
        self.start_camera()

    def get_2D_point(self, shape):
        ih, iw, ic = self.current_frame.shape
        return (int(shape.x * iw), int(shape.y * ih))

    def get_3D_point(self, shape):
        ih, iw, ic = self.current_frame.shape
        return (int(shape.x * iw), int(shape.y * ih), shape.z)

    def animate(self, rig_controller: RigController = None):
        self.get_current_frame()
        current_frameRGB = cv2.cvtColor(
            self.current_frame,
            cv2.COLOR_RGB2BGR
        )
        results = face_mesh.process(current_frameRGB)
        if results.multi_face_landmarks:
            face_lms = results.multi_face_landmarks[0]
            shape = face_lms.landmark
            # For testing purposes to calibrate image_points using indexes shown on image
            # Displayed in red
            for index, landmark in enumerate(shape):
                point = self.get_2D_point(landmark)
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
            # Will be used to determine head rotation using cv2.PnP()
            point_dict = {
                'nose_tip': self.get_3D_point(shape[1]),
                'chin': self.get_3D_point(shape[175]),
                'brow_L': self.get_3D_point(shape[105]),
                'brow_R': self.get_3D_point(shape[334]),
                'eye_corner_R': self.get_3D_point(shape[226]),
                'eye_corner_L': self.get_3D_point(shape[342]),
                'eyelid_up_L': self.get_3D_point(shape[159]),
                'eyelid_up_L': self.get_3D_point(shape[144]),
                'eyelid_up_R': self.get_3D_point(shape[386]),
                'eyelid_low_R': self.get_3D_point(shape[374]),
                'mouth_L': self.get_3D_point(shape[57]),
                'mouth_R': self.get_3D_point(shape[291]),
                'mouth_U': self.get_3D_point(shape[0]),
                'mouth_D': self.get_3D_point(shape[17])
            }
            point_names = {'nose_tip', 'chin', 'eye_corner_R',
                           'eye_corner_L', 'mouth_L', 'mouth_R'}
            image_2D_points = []
            image_3D_points = []
            for k, landmark in point_dict.items():
                x, y, z = landmark
                if k in point_names:
                    image_2D_points.append((x, y))
                    image_3D_points.append((x, y, z))
                    # Shows the selected landmarks
                cv2.putText(
                    self.current_frame,
                    f'.',
                    (x, y),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 190, 225),
                    3
                )
            face2D = np.array(image_2D_points, dtype=np.float64)
            face3D = np.array(image_3D_points, dtype=np.float64)
            self.determine_head_rotation(face2D, face3D)
            # Control rig to animate
            if rig_controller is not None:
                rig_controller.control_bones(
                    face_shape_dict=point_dict,
                    rotation_vector=self.rotation_vector,
                    first_angle=self.first_angle
                )
            # Optionally use mediapipe to draw landmarks
            # mpDraw.draw_landmarks(self.current_frame, face_lms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

        cv2.imshow('Face Detection', self.current_frame)
        cv2.waitKey(1)
        return {'PASS_THROUGH'}

    def determine_head_rotation(self, image_2D_points: np.ndarray, image_3D_points: np.ndarray):
        # Refer to https://www.pythonpool.com/opencv-solvepnp/
        if self.rotation_vector is not None:
            (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(
                objectPoints=image_3D_points,
                imagePoints=image_2D_points,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                rvec=self.rotation_vector, tvec=self.translation_vector,
                useExtrinsicGuess=True
            )
        else:
            (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(
                objectPoints=image_3D_points,
                imagePoints=image_2D_points,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=False
            )
        if not hasattr(self, 'first_angle'):
            self.first_angle = np.copy(self.rotation_vector)
        nose3D = image_3D_points[0]
        nose3D[2] = nose3D[2] * 8000
        self.show_3D_position(image_2D_points, nose3D=np.array([nose3D]))
        # Shows the face direction

    def show_3D_position(self, image_points, nose3D):
        nose_end_point2D, jacobian = cv2.projectPoints(
            nose3D,
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
        if isinstance(self.video_source, int):
            self._cap = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
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
