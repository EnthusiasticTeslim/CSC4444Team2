
from rig_controller import RigController
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

# Mediapipe setup
mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.9
)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

drawing_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)


class FaceAnimator():
    # cv2 capture
    _cap = None
    current_frame = None  # Will hold current video frame
    video_source = None  # Webcam by default

    # Camera resolution
    cam_height = 480
    cam_width = 640

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
            script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
            face_rel_path = "emotionDetection/Emotion_detect"
            face_model_path = os.path.join(script_dir, face_rel_path)
            self.emotion_detect_model = tf.keras.models.load_model(
                face_model_path
            )
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
            # for index, landmark in enumerate(shape):
            #     point = self.get_2D_point(landmark)
            #     x, y = point
            #     cv2.putText(
            #         self.current_frame,
            #         f'{index}',
            #         (x, y),
            #         cv2.FONT_HERSHEY_PLAIN,
            #         1,
            #         (0, 10, 225),
            #         2
            #     )
            # Will be used to determine head rotation using cv2.PnP()
            point_dict = {
                'nose_tip': self.get_3D_point(shape[1]),
                'chin': self.get_3D_point(shape[175]),
                'brow_L': self.get_3D_point(shape[105]),
                'brow_R': self.get_3D_point(shape[334]),
                'brow_Base': self.get_3D_point(shape[8]),
                'eye_corner_R': self.get_3D_point(shape[226]),
                'eye_corner_L': self.get_3D_point(shape[342]),
                'eyelid_up_L': self.get_3D_point(shape[159]),
                'eyelid_low_L': self.get_3D_point(shape[144]),
                'eyelid_up_R': self.get_3D_point(shape[386]),
                'eyelid_low_R': self.get_3D_point(shape[374]),
                'mouth_L': self.get_3D_point(shape[57]),
                'mouth_R': self.get_3D_point(shape[291]),
                'mouth_U': self.get_3D_point(shape[13]),
                'mouth_D': self.get_3D_point(shape[14])
            }
            selected_points = [
                point_dict['nose_tip'],
                point_dict['chin'],
                point_dict['eye_corner_L'],
                point_dict['eye_corner_R'],
                point_dict['mouth_L'],
                point_dict['mouth_R']
            ]
            image_2D_points = []
            image_3D_points = []
            for point in selected_points:
                x, y, z = point[0], point[1], point[2]
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
                )
            #Show emotions
            self.display_emotion()
            # Optionally use mediapipe to draw landmarks
            mp_draw.draw_landmarks(
                self.current_frame, 
                face_lms, mp_face_mesh.FACEMESH_CONTOURS, 
                drawing_spec, 
                drawing_spec
            )
            
        cv2.imshow('Face Detection', self.current_frame)
        cv2.waitKey(1)

    def determine_head_rotation(self, image_2D_points: np.ndarray, image_3D_points: np.ndarray):
        # Refer to https://www.pythonpool.com/opencv-solvepnp/
        if self.rotation_vector is not None:
            (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(
                objectPoints=self.model_points,
                imagePoints=image_2D_points,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                rvec=self.rotation_vector, tvec=self.translation_vector,
                useExtrinsicGuess=True
            )
        else:
            (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(
                objectPoints=self.model_points,
                imagePoints=image_2D_points,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=False
            )

        nose3D = image_3D_points[0]
        nose3D[2] = nose3D[2] * 8000
        # self.show_3D_position(image_2D_points, nose3D=np.array([nose3D]))
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
            self._cap = cv2.VideoCapture(self.video_source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def get_current_frame(self):
        success, self.current_frame = self._cap.read()
        # Optimization
        self.current_frame.flags.writeable = True
        # TODO: Will need to resize image for standardization/consistency
        # Distorts image tho,
        self.current_frame = cv2.resize(
            self.current_frame,
            (self.cam_width, self.cam_height)
        )

    def crop_img(self, img):
        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return
        else:
            image_rows, image_cols, _ = img.shape
            image_input = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detection = results.detections[0]
            location = detection.location_data
            relative_bounding_box = location.relative_bounding_box
            rect_start_point = _normalized_to_pixel_coordinates(
                relative_bounding_box.xmin,
                relative_bounding_box.ymin,
                image_cols,
                image_rows
            )
            rect_end_point = _normalized_to_pixel_coordinates(
                relative_bounding_box.xmin + relative_bounding_box.width,
                relative_bounding_box.ymin + relative_bounding_box.height,
                image_cols,
                image_rows
            )
            xleft, ytop = rect_start_point
            xright, ybot = rect_end_point
            crop_img = image_input[ytop: ybot, xleft: xright]
            target_img_size = 48
            return cv2.resize(crop_img, (target_img_size, target_img_size))

    def detect_emotion(self, img):
        img = self.crop_img(img)
        if img is not None:
            label_dict = {
                0: 'Angry', 1: 'Disgust',
                2: 'Fear', 3: 'Happy',
                4: 'Neutral', 5: 'Sad',
                6: 'Surprise'
            }
            img = np.expand_dims(img, axis=0)  # makes image shape (1,48,48)
            img = img.reshape(1, 48, 48, 1)
            result = self.emotion_detect_model.predict(img)
            result = list(result[0])
            img_index = result.index(max(result))
            return label_dict[img_index]

    def display_emotion(self):
        emotion = self.detect_emotion(self.current_frame)
        if emotion is not None:
            cv2.putText(
                self.current_frame,
                emotion,
                (50, 50),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 10, 225),
                2
            )

    def end_session(self):
        cv2.destroyAllWindows()
        self._cap.release()
        self._cap = None
