import bpy
"""
    Modal Execution:
    https://docs.blender.org/api/current/bpy.types.Operator.html#modal-execution
    
    Event types:
    https://docs.blender.org/api/current/bpy.types.html
    
"""
from rig_controller import RigController
from face_animator import FaceAnimator
        
cv_animation_operator_id = "wm.cv_animation_operator"

class CvAnimationOperator(bpy.types.Operator):
    bl_idname = cv_animation_operator_id
    bl_label = "Cv2 Animation Operator"
    animation_controller : FaceAnimator = None
    rig_controller : RigController = None
    _timer = None
    stop = False
    
    def __init__(self):
        super().__init__()
        print("Start Animation Operator")
        # Specify the rig name.
        rig_name = "RIG-Vincent"
        self.rig_controller = RigController(
            bones_from_rig=bpy.data.objects[rig_name].pose.bones,
            
            # In this example,"RIG-Vincent" has the bone that rotates head named "head_fk"
            head_rotation_bone="head_fk",
            
            # TODO: Update as RigController is implemented
            # eye_brow_left_bone="",
            # mouth_bone="",
            # eye_brow_right_bone="",
            # eyelid_up_left_bone="",
            # eyelid_up_right_bone="",
            # eyelid_down_left_bone="",
            # eyelid_down_right_bone=""
        )
        self.animation_controller = FaceAnimator(
            video_source=0
        )
        
    def __del__(self):
        print("End Animation Operator")
        
    def modal(self, context, event):
        # Cancel by pressing ESC or right mouse button
        if (event.type in {'RIGHTMOUSE', 'ESC'}) or self.stop:
            self.cancel(context)
            return {'CANCELLED'}
        
        # Animate
        if event.type == 'TIMER':
            self.animation_controller.animate(
               rig_controller=self.rig_controller
            )
            
        return {'PASS_THROUGH'}
        
    def stop_playback(self, scene):
        if scene.frame_current == scene.frame_end:
            bpy.ops.screen.animation_cancel(restore_frame=False)

    def execute(self, context):
        bpy.app.handlers.frame_change_pre.append(self.stop_playback)
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        self.animation_controller.end_session()

def register():
    bpy.utils.register_class(CvAnimationOperator)

def unregister():
    bpy.utils.unregister_class(CvAnimationOperator)

if __name__ == "__main__":
    register()
