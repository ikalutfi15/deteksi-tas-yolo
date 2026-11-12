import csv
import math
import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

# ===================== PENGATURAN =====================
SOURCE = "video_uji2.mp4"    # 0 = webcam, atau path video / URL RTSP

# Gunakan file 'best.pt' hasil training kustom kamu di sini:
MODEL_PATH = "best (4).pt" 

IMGSZ = 960               # tas kecil di frame -> pakai ukuran > 640
CONF_PERSON = 0.35
CONF_BAG = 0.20
PERSON_CLASS = 0
BAG_CLASSES = {24: "backpack", 26: "handbag", 28: "suitcase"}

# --- Kriteria jarak dan waktu per status ---
R_ATTEND = 2.0        # m. Penjaga <= ini  -> ATTENDED
R_ABANDON = 3.0       # m. Penjaga > ini   -> syarat ABANDONED
T_SUSPICIOUS = 5.0    # s. 
T_ABANDONED = 10.0    # s. 
T_PROX = 5.0          # s. Dekat tas selama ini -> jadi pemilik pengganti
CARRY_WINDOW = 1.5    # s. Tas baru bergerak + ada di dalam kotak orang

# --- Kalibrasi jarak (piksel -> meter) ---
PERSON_HEIGHT_M = 1.7       # tinggi orang rata-rata
PIXELS_PER_METER = None     # None = pakai tinggi orang otomatis

# --- Pelacakan tas ---
MOVE_PX = 25        # geser > ini (piksel) = tas dianggap bergerak/dibawa
MATCH_PX = 80       # jarak maks (piksel) menyambung deteksi ke tas
STATE_TTL = 10.0    # detik; tas tak terlihat selama ini -> dilupakan
GHOST_DRAW = 3.0    # detik; kotak tas yang sesaat tak terdeteksi tetap digambar
SAVE_DIR = "kejadian"
LOG_PATH = os.path.join(SAVE_DIR, "log_kejadian.csv")
# =======================================================

# Nama tampil dan warna status (BGR) untuk badge status tas
STATES = {
    "ATTENDED": ("Dalam pengawasan", (0, 180, 0)),         # Hijau
    "TEMP_UNATTENDED": ("Terpisah sementara", (0, 216, 255)), # Kuning
    "SUSPICIOUS": ("Perlu perhatian", (0, 140, 255)),       # Oranye
    "ABANDONED": ("SECURITY ALERT", (0, 0, 230)),          # Merah
}

# WARNA UTAMA KOTAK (Dibedakan agar tidak sama)
PERSON_COLOR = (255, 140, 0)   # Warna Biru Cerah untuk Orang
BAG_COLOR = (0, 165, 255)      # Warna Oranye/Amber untuk Tas
GRAY = (130, 130, 130)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Bag:
    def __init__(self, key, box, label, t, owner):
        self.key = key              # Nomor urut pasangan (misal: 1, 2, dst)
        self.box = box
        self.label = label
        self.first_seen = t
        self.last_seen = t
        self.anchor = center(box)
        self.last_move_t = -1e9
        self.owner = owner          
        self.guardians = set()      
        self.prox = {}              
        self.unatt_since = None     
        self.aband_since = None     
        self.state = "ATTENDED"
        self.unatt_t = 0.0
        self.aband_t = 0.0
        self.d_guard = None
        self.link_pid = None        
        self.alerted = False
        self.review = None          
        self.logged_state = None


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def foot(b):
    return ((b[0] + b[2]) / 2.0, float(b[3]))


def dist_m(bag_box, person_box):
    cx, cy = center(bag_box)
    if person_box[0] <= cx <= person_box[2] and person_box[1] <= cy <= person_box[3]:
        return 0.0  
    bx, by = foot(bag_box)
    px, py = foot(person_box)
    d_px = math.hypot(bx - px, by - py)
    if PIXELS_PER_METER:
        return d_px / PIXELS_PER_METER
    person_h = max(person_box[3] - person_box[1], 1.0)
    return d_px / (person_h / PERSON_HEIGHT_M)


def nearest(box, persons):
    best_id, best_d = None, float("inf")
    for pid, pbox in persons:
        d = dist_m(box, pbox)
        if d < best_d:
            best_id, best_d = pid, d
    return best_id, best_d


def update_bag(bag, persons, t, dt):
    dists = {pid: dist_m(bag.box, pbox) for pid, pbox in persons}

    c = center(bag.box)
    if math.hypot(c[0] - bag.anchor[0], c[1] - bag.anchor[1]) > MOVE_PX:
        bag.anchor = c
        bag.last_move_t = t
        bag.unatt_since = None
        bag.aband_since = None
    carried = (t - bag.last_move_t) <= CARRY_WINDOW

    for pid, d in dists.items():
        if d <= R_ATTEND:
            bag.prox[pid] = bag.prox.get(pid, 0.0) + dt
            carrying = (d == 0.0 and carried)
            if pid == bag.owner or bag.prox[pid] >= T_PROX or carrying:
                bag.guardians.add(pid)
                if bag.owner is None and carrying:
                    bag.owner = pid
        else:
            bag.prox[pid] = 0.0
    for pid in bag.prox:
        if pid not in dists:
            bag.prox[pid] = 0.0

    guard_d = {p: dists[p] for p in bag.guardians if p in dists}
    if guard_d:
        bag.link_pid = min(guard_d, key=guard_d.get)
        bag.d_guard = guard_d[bag.link_pid]
    elif not bag.guardians and dists:
        bag.link_pid = min(dists, key=dists.get)
        bag.d_guard = dists[bag.link_pid]
    else:
        bag.link_pid, bag.d_guard = None, None
        
    near_attend = bag.d_guard is not None and bag.d_guard <= R_ATTEND
    near_abandon = bag.d_guard is not None and bag.d_guard <= R_ABANDON

    if near_attend:
        bag.unatt_since = None
    elif bag.unatt_since is None:
        bag.unatt_since = t
    if near_abandon:
        bag.aband_since = None
    elif bag.aband_since is None:
        bag.aband_since = t

    bag.unatt_t = 0.0 if bag.unatt_since is None else t - bag.unatt_since
    bag.aband_t = 0.0 if bag.aband_since is None else t - bag.aband_since

    if bag.aband_t >= T_ABANDONED:
        bag.state = "ABANDONED"
    elif bag.unatt_t >= T_SUSPICIOUS:
        bag.state = "SUSPICIOUS"
    elif bag.unatt_since is not None:
        bag.state = "TEMP_UNATTENDED"
    else:
        bag.state = "ATTENDED"
        bag.alerted = False
        bag.review = None


def review_flagged(bags, decision, t):
    n = 0
    for b in bags.values():
        if b.state in ("SUSPICIOUS", "ABANDONED") and b.review is None:
            b.review = decision
            log_event(t, b, "KEPUTUSAN_" + decision)
            n += 1
    return n


def log_event(t, bag, event):
    new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["waktu_video_s", "id_tas", "jenis", "kejadian",
                        "jarak_penjaga_m", "terpisah_2m_s", "terpisah_3m_s"])
        w.writerow([round(t, 1), bag.key, bag.label, event,
                    "" if bag.d_guard is None else round(bag.d_guard, 2),
                    round(bag.unatt_t, 1), round(bag.aband_t, 1)])


# ---------------------- Gambar ----------------------
def draw_tag(img, x, y, text, bg, below=False):
    scale = 0.5
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    x = int(x)
    y0 = int(y) + 3 if below else max(int(y) - th - 10, 0)
    cv2.rectangle(img, (x, y0), (x + tw + 10, y0 + th + 10), bg, -1)
    lum = 0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2]
    fg = (30, 30, 30) if lum > 150 else (255, 255, 255)
    cv2.putText(img, text, (x + 5, y0 + th + 4), FONT, scale, fg, 1, cv2.LINE_AA)


def draw_person(img, pbox, text):
    x1, y1, x2, y2 = [int(v) for v in pbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), PERSON_COLOR, 2)
    draw_tag(img, x1, y1, text, PERSON_COLOR)


def draw_bag(img, bag, persons_by_id):
    dismissed = bag.review == "FALSE_ALARM"
    name, scolor = STATES[bag.state]
    if dismissed:
        scolor = GRAY
    x1, y1, x2, y2 = [int(v) for v in bag.box]

    if not dismissed and bag.state in ("SUSPICIOUS", "ABANDONED"):
        cv2.rectangle(img, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), scolor, 3)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), BAG_COLOR, 2)
    draw_tag(img, x1, y1, f"LUGGAGE P{bag.key:02d}", BAG_COLOR)

    if dismissed:
        text = "Alarm palsu (diabaikan)"
    else:
        if bag.state in ("TEMP_UNATTENDED", "SUSPICIOUS"):
            text = f"{name} {bag.unatt_t:.0f}s"
        elif bag.state == "ABANDONED":
            text = f"{name} {bag.aband_t:.0f}s"
        else:
            text = name
        if bag.d_guard is not None:
            text += f" | d={bag.d_guard:.1f}m"
        if bag.review == "CONFIRMED":
            text += " [ditangani]"
    draw_tag(img, x1, y2, text, scolor, below=True)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # YOLO akan memuat file bobot 'best.pt' hasil training kustommu di sini
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(SOURCE)
    is_file = isinstance(SOURCE, str) and not SOURCE.lower().startswith(("rtsp", "http"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    conf_min = min(CONF_PERSON, CONF_BAG)

    bags, id2key = {}, {}
    next_key, frame_idx, last_t = 0, 0, None

    legend1 = (f"Attended: <={R_ATTEND:g}m | Terpisah sementara: >{R_ATTEND:g}m, <{T_SUSPICIOUS:g}s | "
               f"Perlu perhatian: >{R_ATTEND:g}m, >={T_SUSPICIOUS:g}s | "
               f"Alert: >{R_ABANDON:g}m, >={T_ABANDONED:g}s")
    legend2 = ("Kotak Biru=orang | Kotak Oranye=tas | "
               "f=alarm palsu  k=ditangani  q=keluar")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx / fps if is_file else time.time()
        frame_idx += 1
        dt = 0.0 if last_t is None else min(max(t - last_t, 0.0), 1.0)
        last_t = t

        r = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            classes=[PERSON_CLASS, *BAG_CLASSES], conf=conf_min,
            imgsz=IMGSZ, agnostic_nms=True, verbose=False,
        )[0]

        persons, bag_dets = [], []
        if r.boxes is not None and r.boxes.id is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for box, tid, cls, cf in zip(xyxy, ids, clss, confs):
                if cls == PERSON_CLASS:
                    if cf >= CONF_PERSON:
                        persons.append((int(tid), box))
                elif cls in BAG_CLASSES and cf >= CONF_BAG:
                    bag_dets.append((int(tid), box, BAG_CLASSES[cls]))

        seen = set()
        for tid, box, label in bag_dets:
            key = id2key.get(tid)
            if key not in bags or key in seen:
                key, best = None, MATCH_PX
                cb = center(box)
                for k, b in bags.items():
                    if k in seen:
                        continue
                    cb2 = center(b.box)
                    d = math.hypot(cb[0] - cb2[0], cb[1] - cb2[1])
                    if d < best:
                        key, best = k, d
                if key is None:
                    next_key += 1
                    key = next_key
                    owner, d = nearest(box, persons)
                    bags[key] = Bag(key, box, label, t, owner if d <= R_ATTEND else None)
                id2key[tid] = key
            bag = bags[key]
            bag.box, bag.label, bag.last_seen = box, label, t
            seen.add(key)

        for key in seen:
            b = bags[key]
            update_bag(b, persons, t, dt)
            if b.state != b.logged_state:
                if not (b.logged_state is None and b.state == "ATTENDED"):
                    log_event(t, b, b.state)
                b.logged_state = b.state
        for key in [k for k, b in bags.items() if t - b.last_seen > STATE_TTL]:
            del bags[key]

        out = frame.copy()
        h, w = out.shape[:2]
        persons_by_id = dict(persons)
        visible = [b for b in bags.values() if t - b.last_seen <= GHOST_DRAW]

        person_tags = {}
        for b in visible:
            for pid in b.guardians:
                if pid in persons_by_id:
                    mark = "" if (b.owner is None or pid == b.owner) else "*"
                    person_tags.setdefault(pid, []).append(f"P{b.key:02d}{mark}")
        
        for pid, pbox in persons:
            if pid in person_tags:
                text = "PERSON " + "/".join(person_tags[pid])
            else:
                text = f"PERSON #{pid}"
            draw_person(out, pbox, text)

        counts = {s: 0 for s in STATES}
        new_alerts = []
        for bag in visible:
            draw_bag(out, bag, persons_by_id)
            if bag.review != "FALSE_ALARM":
                counts[bag.state] += 1
            if bag.state == "ABANDONED" and bag.review is None and not bag.alerted:
                bag.alerted = True
                new_alerts.append(bag)

        y = 24
        cv2.rectangle(out, (0, 0), (320, 8 + 24 * len(STATES)), (30, 30, 30), -1)
        for s, (name, color) in STATES.items():
            cv2.putText(out, f"{name}: {counts[s]}", (10, y), FONT, 0.6,
                        color if s != "ATTENDED" else (80, 220, 80), 2)
            y += 24
        cv2.rectangle(out, (0, h - 44), (w, h), (30, 30, 30), -1)
        cv2.putText(out, legend1, (8, h - 26), FONT, 0.45, (255, 255, 255), 1)
        cv2.putText(out, legend2, (8, h - 8), FONT, 0.45, (200, 200, 200), 1)

        for bag in new_alerts:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SAVE_DIR, f"tas{bag.key}_{stamp}.jpg")
            cv2.imwrite(path, out)
            print(f"[ALERT] tas L{bag.key:02d} ({bag.label}) memenuhi kriteria ABANDONED, "
                  f"waktu video {t:.1f}s, tersimpan: {path}")

        if w > 1280:
            out = cv2.resize(out, (1280, int(h * 1280 / w)))
        cv2.imshow("Deteksi Tas Tertinggal", out)

        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord("q"):
            break
        elif key_pressed == ord("f"):
            print(f"[SATPAM] alarm palsu: {review_flagged(bags, 'FALSE_ALARM', t)} tas")
        elif key_pressed == ord("k"):
            print(f"[SATPAM] ditangani: {review_flagged(bags, 'CONFIRMED', t)} tas")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()