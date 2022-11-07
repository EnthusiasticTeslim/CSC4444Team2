import bpy
"""
    bpy.types.WorkSpaceTool documentation:
    
    https://docs.blender.org/api/current/bpy.types.WorkSpaceTool.html
    
    This is boilerplate for UI.
    Running this will create a workspace tool to start animation.
    See the screen shots
"""
open_cv_panel_tool_id = "ui_plus.open_cv_panel_tool"

class OpenCVPanelTool(bpy.types.WorkSpaceTool):
    """Creates a Panel in the Object properties window"""
    bl_label = "CV Animation"
    bl_space_type = 'VIEW_3D'
    bl_context_mode='OBJECT'
    bl_idname = "ui_plus.open_cv_panel_tool"
    bl_options = {'REGISTER'}
    bl_icon = "ops.generic.select_circle"
        
    def draw_settings(context, layout, tool):
        row = layout.row()
        op = row.operator("wm.cv_animation_operator", text="Start Capture", icon="OUTLINER_OB_CAMERA")
        
def register():
    bpy.utils.register_tool(
        OpenCVPanelTool, 
        separator=True, 
        group=True
    )

def unregister():
     bpy.utils.unregister_tool(OpenCVPanelTool)

if __name__ == "__main__":
    register()