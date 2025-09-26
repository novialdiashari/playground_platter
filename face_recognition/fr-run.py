import cv2
import face_recognition
import os
import numpy as np


def run_face_recognition(known_faces_dir, video_source=0, tolerance=0.5, process_every=2):
    """
    Multi-Face Recognition menggunakan face_recognition + OpenCV

    Params:
    --------
    known_faces_dir : str
        Path folder database wajah. Struktur:
        known_faces_dir/
            orang1/
                foto1.jpg
                foto2.png
            orang2/
                foto1.jpg
    video_source : int | str, default=0
        - 0 → webcam bawaan laptop
        - 1, 2, dst → webcam eksternal
        - "path/to/video.mp4" → file video
    tolerance : float, default=0.5
        Makin kecil makin ketat (0.4–0.6 umum dipakai)
    process_every : int, default=2
        Proses tiap N frame (biar realtime lebih ringan)
    """

    if not os.path.exists(known_faces_dir):
        raise ValueError(f"❌ Folder '{known_faces_dir}' tidak ditemukan!")

    known_encodings = []
    known_names = []

    # === 1. Load database wajah ===
    for person_name in os.listdir(known_faces_dir):
        person_folder = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        for filename in os.listdir(person_folder):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(person_folder, filename)

                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)

                for encoding in encodings:
                    known_encodings.append(encoding)
                    known_names.append(person_name)

    if not known_encodings:
        raise ValueError(f"❌ Tidak ada wajah yang berhasil di-load dari '{known_faces_dir}'")

    print(f"✅ Loaded {len(known_encodings)} encodings dari {len(set(known_names))} orang")

    # === 2. Buka video / webcam ===
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"❌ Tidak bisa membuka video source: {video_source}")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Video selesai atau kamera tidak memberikan frame.")
            break

        frame_count += 1
        if frame_count % process_every != 0:
            cv2.imshow("Multi-Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # === 3. Resize & konversi RGB ===
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # === 4. Deteksi wajah ===
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # scale back lokasi (karena tadi dikecilkan)
        face_locations = [(top*2, right*2, bottom*2, left*2) 
                          for (top, right, bottom, left) in face_locations]

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
            name = "Unregistered"
            color = (0, 0, 255)

            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = face_distances.argmin() if len(face_distances) > 0 else None

            if best_match_index is not None and matches[best_match_index]:
                confidence = round((1 - face_distances[best_match_index]) * 100, 2)
                name = f"{known_names[best_match_index]} ({confidence}%)"
                color = (0, 255, 0)

            # === 5. Gambar kotak + nama ===
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # === 6. Tampilkan hasil ===
        cv2.imshow("Multi-Face Recognition", frame)

        # tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ==== Cara pakai ====
if __name__ == "__main__":
    KNOWN_FACES_DIR = r"D:\P3 Work\playground_platter\face_recognition\known-faces"

    # Default webcam
    run_face_recognition(KNOWN_FACES_DIR)

    # Atau pakai file video:
    # run_face_recognition(KNOWN_FACES_DIR, r"D:\P3 Work\playground_platter\face_recognition\sources\video6.mp4")

    # Atau webcam eksternal (misalnya index 1)
    # run_face_recognition(KNOWN_FACES_DIR, 1)
