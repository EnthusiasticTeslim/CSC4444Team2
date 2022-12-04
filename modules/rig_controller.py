# TODO: Implement
# Sole purpose is to move Blender bones
import numpy


class RigController():
    def __init__(
        self,
        bones_from_rig,
        head_rotation_bone: str,
        mouth_bone: str,
        eye_brow_left_bone: str,
        eye_brow_right_bone: str,
        eyelid_up_left_bone: str,
        eyelid_up_right_bone: str,
        eyelid_down_left_bone: str,
        eyelid_down_right_bone: str
    ):
        self.bones = bones_from_rig
        self.head_rotation_bone = self.bones[head_rotation_bone]
        self.mouth_bone = self.bones[mouth_bone]
        self.eye_brow_left_bone = self.bones[eye_brow_left_bone]
        self.eye_brow_right_bone = self.bones[eye_brow_right_bone]
        self.eyelid_up_left_bone = self.bones[eyelid_up_left_bone]
        self.eyelid_up_right_bone = self.bones[eyelid_up_right_bone]
        self.eyelid_down_left_bone = self.bones[eyelid_down_left_bone]
        self.eyelid_down_right_bone = self.bones[eyelid_down_right_bone]

    def _control_head_rotation(self, rotation_vector, first_angle):
        # Rotation along x axis
        x_value = rotation_vector[0] 
        # - first_angle[0]
        # Rotation along y axis
        y_value = rotation_vector[1]
        # - first_angle[1]
        # Rotation along z axis
        z_value = rotation_vector[2] 
        # - first_angle[2]
        # Up/Down
        self.head_rotation_bone.rotation_euler[0] = x_value / 1
        # Left/Right
        self.head_rotation_bone.rotation_euler[1] = z_value / 1.3
        # Sideways
        self.head_rotation_bone.rotation_euler[2] = 1 * y_value  / 1.5
        # Update animation
        self.head_rotation_bone.keyframe_insert(
            data_path="rotation_euler", index=-1)

    def _control_mouth(self, face_dict):
        mouth_top = numpy.asarray(face_dict['mouth_U'])
        mouth_bottom = numpy.asarray(face_dict['mouth_D'])
        mouth_L = numpy.asarray(face_dict['mouth_L'])
        mouth_R = numpy.asarray(face_dict['mouth_R'])
        self.mouth_bone.location[2] = self.smooth_value(
            "m_h", 2, -self.get_range(
                "mouth_height",
                numpy.linalg.norm(mouth_top - mouth_bottom)
            )
            * 0.06
        )
        self.mouth_bone.location[0] = self.smooth_value(
            "m_w", 2,
            (self.get_range(
                "mouth_width",
                numpy.linalg.norm(mouth_L - mouth_R)
            )
             - 0.5
             ) 
            * -0.04
        )
        self.mouth_bone.keyframe_insert(data_path="location", index=-1)

    # def _control_eyebrows(self,face_shape_dict):
    #     #eyebrows
    #     bones["brow_ctrl_L"].location[2] = self.smooth_value("b_l", 3, (self.get_range("brow_left", numpy.linalg.norm(shape[19] - shape[27])) -0.5) * 0.04)
    #     bones["brow_ctrl_R"].location[2] = self.smooth_value("b_r", 3, (self.get_range("brow_right", numpy.linalg.norm(shape[24] - shape[27])) -0.5) * 0.04)
        
    #     bones["brow_ctrl_L"].keyframe_insert(data_path="location", index=2)
    #     bones["brow_ctrl_R"].keyframe_insert(data_path="location", index=2)
        

    def _control_eyelids(self, face_dict):
        eyelid_up_L = numpy.asarray(face_dict['eyelid_up_L'])
        eyelid_up_R = numpy.asarray(face_dict['eyelid_up_R'])
        eyelid_low_L = numpy.asarray(face_dict['eyelid_low_L'])    
        eyelid_low_R = numpy.asarray(face_dict['eyelid_low_R'])
    
        l_open = self.smooth_value("e_l", 2, self.get_range("l_open", -numpy.linalg.norm(eyelid_up_L - eyelid_low_L))  )
        r_open = self.smooth_value("e_r", 2, self.get_range("r_open", -numpy.linalg.norm(eyelid_up_R - eyelid_low_R))  )
        eyes_open = (l_open + r_open) / 2.0 # looks weird if both eyes aren't the same...
        self.eyelid_up_right_bone.location[2] =   -eyes_open * 0.025 + 0.005
        self.eyelid_up_right_bone.keyframe_insert(data_path="location", index=2)
        
        self.eyelid_down_right_bone.location[2] =  eyes_open * 0.025 - 0.005
        self.eyelid_down_right_bone.keyframe_insert(data_path="location", index=2)
        
        self.eyelid_up_left_bone.location[2] =   -eyes_open * 0.025 + 0.005
        self.eyelid_up_left_bone.keyframe_insert(data_path="location", index=2)
        
        self.eyelid_down_left_bone.location[2] =  eyes_open * 0.025 - 0.005
        self.eyelid_down_left_bone.keyframe_insert(data_path="location", index=2)
        

    def control_bones(self, face_shape_dict, rotation_vector, first_angle):
        self._control_head_rotation(rotation_vector, first_angle)
        self._control_mouth(face_shape_dict)
        # self._control_eyebrows(face_shape_dict)
        self._control_eyelids(face_shape_dict)

    # Keeps a moving average of given length
    def smooth_value(self, name, length, value):
        if not hasattr(self, 'smooth'):
            self.smooth = {}
        if not name in self.smooth:
            self.smooth[name] = numpy.array([value])
        else:
            self.smooth[name] = numpy.insert(
                arr=self.smooth[name], obj=0, values=value)
            if self.smooth[name].size > length:
                self.smooth[name] = numpy.delete(
                    self.smooth[name], self.smooth[name].size-1, 0)
        sum = 0
        for val in self.smooth[name]:
            sum += val
        return sum / self.smooth[name].size

    # Keeps min and max values, then returns the value in a range 0 - 1
    def get_range(self, name, value):
        if not hasattr(self, 'range'):
            self.range = {}
        if not name in self.range:
            self.range[name] = numpy.array([value, value])
        else:
            self.range[name] = numpy.array(
                [min(value, self.range[name][0]), max(value, self.range[name][1])])
        val_range = self.range[name][1] - self.range[name][0]
        if val_range != 0:
            return (value - self.range[name][0]) / val_range
        else:
            return 0.0
