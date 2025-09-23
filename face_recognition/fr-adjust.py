import cv2
import face_recognition
import os
import numpy as np

# Folder berisi foto orang yang dikenal
KNOWN_FACES_DIR = r"D:\P3 Work\playground_platter\face_recognition\known-faces"

known_encodings = []
known_names = []

# 1. Load semua wajah dari folder
for person_name in os.listdir(KNOWN_FACES_DIR):
    person_folder = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_folder):
        continue  # skip kalau bukan folder

    for filename in os.listdir(person_folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(person_folder, filename)

            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            for encoding in encodings:
                known_encodings.append(encoding)
                known_names.append(person_name)

print(f"✅ Loaded {len(known_encodings)} total encodings dari {len(set(known_names))} orang")

# 2. Buka webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ubah ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 3. Deteksi wajah
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 4. Bandingkan dengan semua wajah yang dikenal
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unregistered"
        color = (0, 0, 255)

        # 5. Ambil wajah dengan jarak terkecil
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = face_distances.argmin() if len(face_distances) > 0 else None

        if best_match_index is not None and matches[best_match_index]:
            name = f"Registered: {known_names[best_match_index]}"
            color = (0, 255, 0)  # hijau untuk known face

        # 6. Gambar kotak & nama
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 7. Tampilkan hasil
    cv2.imshow("Multi-Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
