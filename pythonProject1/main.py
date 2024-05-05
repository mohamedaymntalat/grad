import numpy as np
import face_recognition as fr
import os
import cv2
from ffpyplayer.player import MediaPlayer
import tensorflow
import keras


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
    # imgs.append(currentimg)
    image_path = path + _
    encoding = fr.face_encodings(image)[0]
    known_name_encodings.append(encoding)
    known_names.append(os.path.splitext(_)[0])
print(known_names)

image = cv2.imread('Nefer.jpg')

face_locations = fr.face_locations(image)

face_encodings = fr.face_encodings(image, face_locations)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = fr.compare_faces(known_name_encodings, face_encoding)
    name = ""
    face_distances = fr.face_distance(known_name_encodings, face_encoding)
    best_match = np.argmin(face_distances)
    print(matches)
    if matches[best_match]:
        name = known_names[best_match]
        vid_path = os.path.join(vids, name + '.mp4')
        print(vid_path)
        print(name)
        if os.path.exists(vid_path):
            # Create a VideoCapture object and read from input file
            cap = cv2.VideoCapture(vid_path)
            player = MediaPlayer(vid_path)
            audio_frame, val = player.get_frame()
            # fps = cap.get(cv2.CAP_PROP_FPS), sleep_ms = int(np.round((1 / fps) * 1000))
            # Check if camera opened successfully
            if not cap.isOpened():
                print("Error opening video file")
            # Read until video is completed
            while (cap.isOpened()):
                # Capture frame-by-frame
                ret, frame = cap.read()
                #fps = cap.get(cv2.CAP_PROP_POS_MSEC)
                #frm_delay = int(1000 / fps)
                if ret == True:
                    # Display the resulting frame
                    cv2.imshow('Frame', frame)
                    if val != 'eof' and audio_frame is not None:
                        # audio
                        frame, t = audio_frame
                    # Press Q on keyboard to exit
                    key = cv2.waitKey(36)
                    if key == 81 or key == 113:  # if you press q or Q exit
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

        # cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

