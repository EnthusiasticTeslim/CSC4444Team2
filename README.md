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
- Copy all the contents of the `CSC4444Team2/modules` (emotion_model, rig_controller and face_animator) and paste them to your Blender modules folder. (`"C:/Program Files/Blender Foundation/Blender 2.93/2.93/scripts/modules/"`)
- Open `blender/vincent-draft.blend` using Blender 
- Open `"Scripting"` tab. 

    ![screenshot 1](https://github.com/Endoplex/CSC4444Team2/blob/main/screenshots/Screenshot1.png?raw=true)
- Update the video location in the script by updating the following variable:

    ![screenshot 2](https://github.com/Endoplex/CSC4444Team2/blob/main/screenshots/Screenshot2.png?raw=true)
- Run the `animator.py` file on the dropdown.

    ![screenshot 3](https://github.com/Endoplex/CSC4444Team2/blob/main/screenshots/Screenshot3.png?raw=true)
- Open `"Layout"` tab.

    ![screenshot 4](https://github.com/Endoplex/CSC4444Team2/blob/main/screenshots/Screenshot4.png?raw=true) 
- A workspace tool will show up on the left side. Click on the tool.

    ![screenshot 5](https://github.com/Endoplex/CSC4444Team2/blob/main/screenshots/Screenshot5.png?raw=true)
- Click `"Start Capture"` on the right of the screen. 

    ![screenshot 6](https://github.com/Endoplex/CSC4444Team2/blob/main/screenshots/Screenshot6.png?raw=true)

- A window showing the video feed will show up and the character should start animating.

    ![screen_recording1](https://github.com/Endoplex/CSC4444Team2/blob/main/screenshots/Screen_recording1.gif)
- To close animation, click back on the the Blender window and press `"Esc"` key.