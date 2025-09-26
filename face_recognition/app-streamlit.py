import streamlit as st
import cv2
import face_recognition
import os
import numpy as np

# Folder berisi foto orang yang dikenal
KNOWN_FACES_DIR = r"D:\P3 Work\playground_platter\face_recognition\known-faces"

# --- Load known faces ---
@st.cache_resource
def load_known_faces():
    known_encodings = []
    known_names = []

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_folder = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue

        for filename in os.listdir(person_folder):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(person_folder, filename)

                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)

                for encoding in encodings:
                    known_encodings.append(encoding)
                    known_names.append(person_name)

    return known_encodings, known_names

known_encodings, known_names = load_known_faces()
st.success(f"✅ Loaded {len(known_encodings)} wajah dari {len(set(known_names))} orang")

# --- Streamlit UI ---
st.title("📸 Multi-Face Recognition App")
run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ Tidak bisa akses webcam.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unregistered"
        color = (255, 0, 0)

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

        if best_match_index is not None and matches[best_match_index]:
            confidence = round((1 - face_distances[best_match_index]) * 100, 2)
            name = f"Registered-{known_names[best_match_index]} ({confidence}%)"
            color = (0, 255, 0)

        cv2.rectangle(rgb_frame, (left, top), (right, bottom), color, 2)
        cv2.putText(rgb_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    FRAME_WINDOW.image(rgb_frame)

cap.release()
