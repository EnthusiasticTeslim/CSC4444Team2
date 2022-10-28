# TODO: Implement
# Sole purpose is to move Blender bones

class RigController():
    bones = None
    head_rotation_bone="",
    # eye_brow_left_bone=None,
    # mouth_bone=None,
    # eye_brow_right_bone=None,
    # eyelid_up_left_bone=None,
    # eyelid_up_right_bone=None,
    # eyelid_down_left_bone=None,
    # eyelid_down_right_bone=None

    def __init__(
        self,
        bones_from_rig,
        head_rotation_bone,
        # eye_brow_left_bone,
        # mouth_bone,
        # eye_brow_right_bone,
        # eyelid_up_left_bone,
        # eyelid_up_right_bone,
        # eyelid_down_left_bone,
        # eyelid_down_right_bone
    ):

        self.bones = bones_from_rig
        self.head_rotation_bone = head_rotation_bone
        # mouth_bone = mouth_bone
        # eye_brow_left_bone = eye_brow_left_bone
        # eye_brow_right_bone = eye_brow_right_bone
        # eyelid_up_left_bone = eyelid_up_left_bone
        # eyelid_up_right_bone = eyelid_up_right_bone
        # eyelid_down_left_bone = eyelid_down_left_bone
        # eyelid_down_right_bone = eyelid_down_right_bone

    def _control_head_rotation(self, rotation_vector, first_angle):
        bone = self.bones[self.head_rotation_bone]
        # Rotation along x axis
        x_value = rotation_vector[0] - first_angle[0]
        # Rotation along y axis
        y_value = rotation_vector[1] - first_angle[1]
        # Rotation along z axis
        z_value = rotation_vector[2] - first_angle[2] 
        
        # Up/Down
        bone.rotation_euler[0] = x_value / 1 
        # Left/Right
        bone.rotation_euler[1] = z_value / 1.3 
        # Sideways
        bone.rotation_euler[2] = 1 * y_value / 1.5

        bone.keyframe_insert(data_path="rotation_euler", index=-1)

    def _control_mouth(self):
        pass

    def _control_eyebrows(self):
        pass

    def _control_eyelids(self):
        pass

    def control_bones(self, face_shape, rotation_vector, first_angle):
        self._control_head_rotation(rotation_vector, first_angle)
        # self._control_mouth()
        # self._control_eyebrows()
        # self._control_eyelids()
