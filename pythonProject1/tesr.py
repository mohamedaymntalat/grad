from flask import Flask, render_template, request, jsonify
import numpy as np
import face_recognition as fr
import os
import cv2
from ffpyplayer.player import MediaPlayer


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Your existing face recognition code
vids = 'videos'
myVid = os.listdir(vids)
videos = []

for vid in myVid:
    video = cv2.imread(f'{vids}\{vid}')
    videos.append(video)

path = "imgs"

known_names = []
known_name_encodings = []

images = os.listdir(path)

for _ in images:
    image = cv2.imread(f'{path}\{_}')
    encoding = fr.face_encodings(image)[0]
    known_name_encodings.append(encoding)
    known_names.append(os.path.splitext(_)[0])


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
@app.route('/upload', methods=['POST'])
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        image = cv2.imread(filename)
        if image is None:
            return render_template('index.html', result="Failed to read uploaded image")

        face_locations = fr.face_locations(image)
        face_encodings = fr.face_encodings(image, face_locations)

        # Perform face recognition
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = fr.compare_faces(known_name_encodings, face_encoding)
            name = ""
            face_distances = fr.face_distance(known_name_encodings, face_encoding)
            best_match = np.argmin(face_distances)
            if matches[best_match]:
                name = known_names[best_match]
                vid_path = os.path.join(vids, name + '.mp4')
                if os.path.exists(vid_path):
                    # Display video
                    cap = cv2.VideoCapture(vid_path)
                    player = MediaPlayer(vid_path)
                    audio_frame, val = player.get_frame()
                    if not cap.isOpened():
                        print("Error opening video file")
                    while (cap.isOpened()):
                        ret, frame = cap.read()
                        if ret == True:
                            cv2.imshow('Frame', frame)
                            if val != 'eof' and audio_frame is not None:
                                frame, t = audio_frame
                            key = cv2.waitKey(36)
                            if key == 81 or key == 113:
                                break
                        else:
                            break
                else:
                    print(f'Video file not found: {vid_path}')
            else:
                print('Sorry we do not have data on this person right now.')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return render_template('index.html', result="Face recognized")
    else:
        return render_template('index.html', result="No file uploaded")


if __name__ == '__main__':
    app.run(debug=True)