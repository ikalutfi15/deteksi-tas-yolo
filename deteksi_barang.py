import time
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
video_path = "video_uji3.mp4"  # Sesuaikan nama videomu
cap = cv2.VideoCapture(video_path)

# Kamus memori untuk menyimpan objek yang tertinggal agar tidak hilang kotaknya
# Format: {id: {'center': (x,y), 'start_time': float, 'box': [x1,y1,x2,y2], 'status': 'warning/lost', 'label': str, 'color': (b,g,r)}}
abandoned_objects = {}
ABANDONED_LIMIT = 3  # Batas waktu 3 detik
counter_id = 0

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break

  results = model.track(
      frame, persist=True, classes=[0, 24, 26, 28], conf=0.08
  )
  annotated_frame = frame.copy()
  current_time = time.time()

  current_frame_detected_luggages = []
  persons = []
  luggages = []

  if results[0].boxes.id is not None:
    boxes = results[0].boxes.xyxy.cpu().numpy()
    track_ids = results[0].boxes.id.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    for box, track_id, cls in zip(boxes, track_ids, classes):
      if int(cls) == 0:
        persons.append((int(track_id), box))
      else:
        luggages.append((int(track_id), box))

    # 1. Gambar kotak untuk ORANG
    for p_id, p_box in persons:
      cv2.rectangle(
          annotated_frame,
          (int(p_box[0]), int(p_box[1])),
          (int(p_box[2]), int(p_box[3])),
          (255, 0, 0),
          2,
      )
      cv2.putText(
          annotated_frame,
          f"id:{p_id} person",
          (int(p_box[0]), int(p_box[1] - 10)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          (255, 0, 0),
          2,
      )

    # 2. Proses tas yang terdeteksi AI pada frame ini
    for l_id, l_box in luggages:
      l_x_center = (l_box[0] + l_box[2]) / 2
      l_y_center = (l_box[1] + l_box[3]) / 2

      best_person_id = None
      for p_id, p_box in persons:
        margin_left = 120
        margin_right = 80
        margin_bottom = 80

        if (
            (p_box[0] - margin_left) <= l_x_center <= (p_box[2] + margin_right)
            and (p_box[1] - 40) <= l_y_center <= (p_box[3] + margin_bottom)
        ):
          best_person_id = p_id
          break

      if best_person_id is not None:
        # Tas dibawa pemilik (Aman)
        label_tas = f"id:{best_person_id} luggage (Safe)"
        box_color = (0, 255, 0)  # Hijau

        # Hapus dari memori abandoned jika sebelumnya tercatat di dekat sini
        keys_to_del = []
        for k, v in abandoned_objects.items():
          dist = np.sqrt(
              (v["center"][0] - l_x_center) ** 2
              + (v["center"][1] - l_y_center) ** 2
          )
          if dist < 80:
            keys_to_del.append(k)
        for k in keys_to_del:
          del abandoned_objects[k]

        # Gambar kotak hijau aman
        cv2.rectangle(
            annotated_frame,
            (int(l_box[0]), int(l_box[1])),
            (int(l_box[2]), int(l_box[3])),
            box_color,
            2,
        )
        cv2.putText(
            annotated_frame,
            label_tas,
            (int(l_box[0]), int(l_box[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2,
        )

      else:
        # Tas TIDAK ADA pemilik di dekatnya -> Cek apakah sudah ada di memori abandoned
        matched_key = None
        for k, v in abandoned_objects.items():
          dist = np.sqrt(
              (v["center"][0] - l_x_center) ** 2
              + (v["center"][1] - l_y_center) ** 2
          )
          if dist < 60:
            matched_key = k
            break

        if matched_key is not None:
          # Update posisi dan teruskan timer yang sudah berjalan
          abandoned_objects[matched_key]["center"] = (l_x_center, l_y_center)
          abandoned_objects[matched_key]["box"] = l_box
          elapsed_time = (
              current_time - abandoned_objects[matched_key]["start_time"]
          )
        else:
          # Daftarkan tas baru yang ditinggal
          counter_id += 1
          matched_key = counter_id
          abandoned_objects[matched_key] = {
              "center": (l_x_center, l_y_center),
              "start_time": current_time,
              "box": l_box,
          }
          elapsed_time = 0

        # Tentukan status warna & teks
        if elapsed_time >= ABANDONED_LIMIT:
          label_tas = f"LOST OBJECT! ({int(elapsed_time)}s)"
          box_color = (0, 0, 255)  # Merah
        else:
          remaining = int(ABANDONED_LIMIT - elapsed_time)
          label_tas = f"Warning: unattended ({max(0, remaining)}s)"
          box_color = (0, 255, 255)  # Kuning

        # Simpan status ke memori
        abandoned_objects[matched_key]["label"] = label_tas
        abandoned_objects[matched_key]["color"] = box_color

        current_frame_detected_luggages.append(matched_key)

        # Gambar kotak tas
        cv2.rectangle(
            annotated_frame,
            (int(l_box[0]), int(l_box[1])),
            (int(l_box[2]), int(l_box[3])),
            box_color,
            2,
        )
        cv2.putText(
            annotated_frame,
            label_tas,
            (int(l_box[0]), int(l_box[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2,
        )

  # 3. RENDER ULANG DARI MEMORI JIKA PADA FRAME INI YOLO GAGAL MENDETEKSI TAS (MISSED DETECTION)
  for k, v in abandoned_objects.items():
    # Jika tas ini tidak terdeteksi oleh YOLO pada frame saat ini, tetap render dari posisi terakhirnya di memori
    # Hitung ulang elapsed time agar timer tetap jalan walau objek tak terlihat kamera sesaat
    elapsed_time = current_time - v["start_time"]

    if elapsed_time >= ABANDONED_LIMIT:
      label_tas = f"LOST OBJECT! ({int(elapsed_time)}s)"
      box_color = (0, 0, 255)
    else:
      remaining = int(ABANDONED_LIMIT - elapsed_time)
      label_tas = f"Warning: unattended ({max(0, remaining)}s)"
      box_color = (0, 255, 255)

    # Ambil box terakhir yang tersimpan
    bx = v["box"]
    cv2.rectangle(
        annotated_frame,
        (int(bx[0]), int(bx[1])),
        (int(bx[2]), int(bx[3])),
        box_color,
        2,
    )
    cv2.putText(
        annotated_frame,
        label_tas,
        (int(bx[0]), int(bx[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        box_color,
        2,
    )

  cv2.imshow("Abandoned Object Detection", annotated_frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()