# CSC4444Team2
REQUIREMENTS:
- Blender 2.93 with Python 3.10.
- Windows PC, MacOS not supported yet due to camera permissions settings.
- Webcam or video source with a face.

RUNNING INSTRUCTIONS:
- Get the source code:
    - `git clone https://github.com/Endoplex/CSC4444Team2.git`
- Install cv2, mediapipe, and numpy to you blender modules folder as follows:
    - Open Command Prompt as Administrator 
    - cd "C:\Program Files\Blender Foundation\Blender 2.93\2.93\python\bin"`
    - python -m pip install --upgrade pip
    - python -m pip install opencv-contrib-python numpy==1.21.6 opencv-python mediapipe tensorflow
- Paste rig_controller and face_animator files to your blender modules folder. (`"C:/Program Files/Blender Foundation/Blender 2.93/2.93/scripts/modules/"`)
- Open blender/vincent-draft.blend using blender 
- Open `"Scripting"` tab. 

    ![screenshot 1](https://github.com/Endoplex/CSC4444Team2/blob/main/screenshots/screenshot1.png?raw=true)
- Update the video location in the script by updating the following variable:

    ![screenshot 2](https://github.com/Endoplex/CSC4444Team2/blob/main/screenshots/screenshot2.png?raw=true)
- Run the `animator.py` file on the dropdown.
- Open `"Layout"` tab. 
- A workspace tool will show up.

    ![screenshot 3](https://github.com/Endoplex/CSC4444Team2/blob/main/screenshots/screenshot3.png?raw=true)
- Click "Start Capture". It should start animation the character.