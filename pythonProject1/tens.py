import numpy as np
import cv2
import os
import tensorflow as tf
import face_recognition as fr
from ffpyplayer.player import MediaPlayer

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        image = cv2.imread(filename)
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
                if not os.path.exists(vid_path):
                    path = os.path.join(path, name + '.jpg')
                    if os.path.exists(path):
                        img = cv2.imread(name + '.jpg')
                        cv2.imshow(name, img)
            else:
                print('Sorry we do not have data on this person right now.')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return "Face recognized"
    else:
        return "No file uploaded"
