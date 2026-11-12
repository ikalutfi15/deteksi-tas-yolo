import csv
import math
import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

# ===================== PENGATURAN =====================
SOURCE = "video_uji16.mp4"    # 0 = webcam, atau path video / URL RTSP
MODEL_PATH = "yolov8n.pt"     # Gunakan yolov8n.pt agar lebih ringan di CPU
IMGSZ = 640                   # Turunkan ke 640 agar proses per frame jauh lebih cepat
CONF_PERSON = 0.35            # ambang untuk orang BARU
CONF_PERSON_TRACKED = 0.20    # [BARU] ambang lebih longgar untuk orang yang identitasnya sudah dikenal
CONF_BAG = 0.20
PERSON_CLASS = 0
BAG_CLASSES = {24: "backpack", 26: "handbag", 28: "suitcase"}

# --- Kriteria jarak dan waktu per status ---
R_ATTEND = 2.0      # m. Penjaga <= ini  -> ATTENDED
R_ABANDON = 3.0     # m. Penjaga > ini   -> syarat ABANDONED
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

# --- [BARU] Kesinambungan identitas orang (mengatasi identity switch) ---
TRACK_BUFFER_S = 3.0    # s. ByteTrack mengingat track yang hilang selama ini (bawaan hanya ~1 s)
RELINK_S = 8.0          # s. ID baru boleh disambung ke orang yang hilang paling lama sekian detik lalu
RELINK_DIST_H = 0.6     # jarak maks pusat kotak, dalam satuan tinggi badan (saat orang diam)
RELINK_SPEED_H = 0.5    # tambahan jarak per detik hilang, dalam satuan tinggi badan
RELINK_SIZE_TOL = 1.6   # rasio tinggi kotak lama vs baru yang masih dianggap orang yang sama
RELINK_IOU = 0.30       # kotak baru yang menumpuk >= ini dengan kotak acuan lama dianggap orang yang sama
OCC_OVERLAP = 0.15      # kotak yang tertutup orang lain >= ini dianggap sedang terhalang (tidak jadi acuan)
REF_STALE_S = 4.0       # s. kalau terhalang selama ini, kotak acuan tetap diperbarui
RELINK_SHRINK = 0.75    # kotak yang tingginya < ini x kotak acuan dianggap kotak sebagian (tidak jadi acuan)
GHOST_PERSON_S = 2.5    # s. orang yang hilang sesaat tetap dihitung di posisi terakhirnya

DEBUG_PRINT = False     # True = cetak ID orang, penjaga, dan jarak setiap frame (untuk pelacakan masalah)

SAVE_DIR = "kejadian"
LOG_PATH = os.path.join(SAVE_DIR, "log_kejadian.csv")
LOG_ID_PATH = os.path.join(SAVE_DIR, "log_identitas.csv")
# =======================================================

STATES = {
    "ATTENDED": ("Dalam pengawasan", (0, 180, 0)),          # Hijau
    "TEMP_UNATTENDED": ("Terpisah sementara", (0, 216, 255)), # Kuning
    "SUSPICIOUS": ("Perlu perhatian", (0, 140, 255)),       # Oranye
    "ABANDONED": ("SECURITY ALERT", (0, 0, 230)),          # Merah
}

PERSON_COLOR = (255, 140, 0)
BAG_COLOR = (0, 165, 255)
GRAY = (130, 130, 130)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Bag:
    def __init__(self, key, box, label, t, owner):
        self.key = key
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


# ======================================================================
# [BARU] Ingatan identitas orang
# ByteTrack hanya memakai posisi kotak, sehingga ID orang bisa berganti
# saat orang tertutup sesaat. Kelas ini memberi setiap orang "ID stabil"
# dan menyambung ID mentah yang baru ke orang yang baru saja hilang di
# tempat yang sama dengan ukuran yang mirip (sama seperti yang sudah
# dilakukan untuk tas lewat MATCH_PX).
#
# Penting: kotak "acuan" tiap orang HANYA diperbarui saat orang itu tidak
# tertutup orang lain. Kotak sebagian (saat tertutup) tidak dipakai, supaya
# orang yang duduk diam tetap bisa dikenali setelah orang lain lewat.
# ======================================================================
def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def covered_frac(a, b):
    """Bagian luas kotak a yang tertutup kotak b (0-1)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    return inter / max((a[2] - a[0]) * (a[3] - a[1]), 1.0)


class PersonMemory:
    def __init__(self):
        self.raw2sid = {}    # ID mentah ByteTrack -> ID stabil
        self.ref = {}        # ID stabil -> kotak acuan (terakhir yang tidak tertutup)
        self.ref_t = {}      # ID stabil -> waktu kotak acuan diperbarui
        self.seen_t = {}     # ID stabil -> waktu terakhir terlihat
        self.next_sid = 0
        self.relinks = []    # catatan semua penyambungan

    def known(self, raw_id):
        return raw_id in self.raw2sid

    def _match(self, box, t, sid):
        """((biaya, jarak_px, hilang_s), "cocok") bila cocok; selain itu (None, alasan)."""
        elapsed = t - self.seen_t[sid]
        if elapsed <= 0:
            return None, "masih terlihat"
        if elapsed > RELINK_S:
            return None, f"sudah hilang {elapsed:.1f}s (batas {RELINK_S:g}s)"
        rbox = self.ref[sid]
        lh = max(rbox[3] - rbox[1], 1.0)
        nh = max(box[3] - box[1], 1.0)
        ratio = nh / lh
        iou = box_iou(box, rbox)
        cx, cy = center(box)
        lx, ly = center(rbox)
        dist = math.hypot(cx - lx, cy - ly)
        allowed = lh * (RELINK_DIST_H + RELINK_SPEED_H * elapsed)
        if iou < RELINK_IOU:
            if ratio > RELINK_SIZE_TOL or ratio < 1.0 / RELINK_SIZE_TOL:
                return None, f"ukuran beda (x{ratio:.1f})"
            if dist > allowed:
                return None, f"posisi terlalu jauh ({dist / lh:.1f}x tinggi badan)"
        cost = dist / allowed + abs(math.log(ratio)) + (1.0 - iou)
        return (cost, dist, elapsed), "cocok"

    def resolve(self, dets, t):
        """
        dets: [(id_mentah, kotak)] pada frame ini.
        Mengembalikan (terlihat, hantu, penyambungan, orang_baru):
          terlihat    : [(id_stabil, kotak)]
          hantu       : [(id_stabil, kotak_acuan)] orang yang hilang sesaat
          penyambungan: ID baru yang disambung pada frame ini
          orang_baru  : ID baru yang TIDAK disambung padahal ada orang yang baru hilang
                        (berisi alasan penolakan, untuk penelusuran masalah)
        """
        n_before = len(self.relinks)
        visible, used, pending = [], set(), []

        for raw, box in dets:
            sid = self.raw2sid.get(raw)
            if sid is not None and sid not in used:
                used.add(sid)
                visible.append((sid, box))
            else:
                if sid is not None:      # ID stabil sudah dipakai kotak lain di frame ini
                    del self.raw2sid[raw]
                pending.append((raw, box))

        # sambungkan ID mentah yang belum dikenal ke orang yang baru hilang
        pairs = []
        for pi, (raw, box) in enumerate(pending):
            for sid in self.seen_t:
                if sid in used:
                    continue
                res, _ = self._match(box, t, sid)
                if res is not None:
                    pairs.append((res[0], pi, sid, res[1], res[2]))
        pairs.sort()
        assigned = {}
        for cost, pi, sid, dist, elapsed in pairs:
            if pi in assigned or sid in used:
                continue
            assigned[pi] = sid
            used.add(sid)
            self.relinks.append((t, pending[pi][0], sid, dist, elapsed))

        new_ids = []
        for pi, (raw, box) in enumerate(pending):
            sid = assigned.get(pi)
            if sid is None:              # orang baru
                reasons = []
                for cand in self.seen_t:
                    if cand in used:
                        continue
                    if 0 < t - self.seen_t[cand] <= RELINK_S:
                        reasons.append((cand, self._match(box, t, cand)[1]))
                self.next_sid += 1
                sid = self.next_sid
                used.add(sid)
                if reasons:
                    new_ids.append((t, raw, sid, reasons))
            self.raw2sid[raw] = sid
            visible.append((sid, box))

        # perbarui ingatan; kotak acuan tidak diubah saat orang tertutup orang lain
        for sid, box in visible:
            occluded = any(covered_frac(box, other) >= OCC_OVERLAP
                           for s2, other in visible if s2 != sid)
            if sid in self.ref:
                rh = max(self.ref[sid][3] - self.ref[sid][1], 1.0)
                shrunk = (box[3] - box[1]) < RELINK_SHRINK * rh    # kotak mendadak mengecil = kotak sebagian
            else:
                shrunk = False
            stale = (t - self.ref_t.get(sid, -1e9)) > REF_STALE_S
            if sid not in self.ref or stale or not (occluded or shrunk):
                self.ref[sid] = box
                self.ref_t[sid] = t
            self.seen_t[sid] = t

        ghosts = [(sid, self.ref[sid]) for sid, st in self.seen_t.items()
                  if sid not in used and 0 < t - st <= GHOST_PERSON_S]
        return visible, ghosts, self.relinks[n_before:], new_ids


def make_tracker_cfg(buffer_frames):
    """
    [BARU] Salin bytetrack.yaml bawaan Ultralytics lalu naikkan track_buffer.
    Semua parameter lain dibiarkan sama seperti bawaan versi yang terpasang.
    """
    try:
        import yaml
        import ultralytics
        src = os.path.join(os.path.dirname(ultralytics.__file__),
                           "cfg", "trackers", "bytetrack.yaml")
        with open(src, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["track_buffer"] = max(int(buffer_frames), 1)
        out = "bytetrack_kampus.yaml"
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"Tracker: {out} (track_buffer={cfg['track_buffer']} frame)")
        return out
    except Exception as e:
        print(f"Peringatan: gagal membuat konfigurasi tracker ({e}); memakai bytetrack.yaml bawaan.")
        return "bytetrack.yaml"


def update_bag(bag, persons, t, dt, ghosts=()):
    dists = {pid: dist_m(bag.box, pbox) for pid, pbox in persons}
    ghost_d = {pid: dist_m(bag.box, pbox) for pid, pbox in ghosts}   # [BARU] orang hilang sesaat
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
        # [BARU] hitungan dekat tidak direset saat orang hilang sesaat
        if pid not in dists and pid not in ghost_d:
            bag.prox[pid] = 0.0

    guard_d = {p: dists[p] for p in bag.guardians if p in dists}
    for p in bag.guardians:                      # [BARU] penjaga yang hilang sesaat tetap dihitung
        if p not in guard_d and p in ghost_d:
            guard_d[p] = ghost_d[p]
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


def _log_identitas(row):
    new = not os.path.exists(LOG_ID_PATH)
    with open(LOG_ID_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["waktu_video_s", "id_mentah", "id_stabil", "tindakan", "keterangan"])
        w.writerow(row)


def log_relink(ev):
    """[BARU] Catat ID orang yang disambung (bahan evaluasi identity switch)."""
    t, raw, sid, dist, elapsed = ev
    _log_identitas([round(t, 1), raw, sid, "disambung",
                    f"hilang {elapsed:.1f}s; selisih posisi {dist:.0f}px"])


def log_new_id(ev):
    """[BARU] Catat orang baru yang TIDAK disambung beserta alasannya."""
    t, raw, sid, reasons = ev
    _log_identitas([round(t, 1), raw, sid, "orang_baru",
                    "; ".join(f"P{c:02d}: {why}" for c, why in reasons)])


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
    draw_tag(img, x1, y1, f"LUGGAGE L{bag.key:02d}", BAG_COLOR)

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
        if bag.link_pid is not None:
            text += f" | P{bag.link_pid:02d}"
        if bag.review == "CONFIRMED":
            text += " [ditangani]"
    draw_tag(img, x1, y2, text, scolor, below=True)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(SOURCE)

    if not cap.isOpened():
        print(f"Error: Gagal membuka video '{SOURCE}'")
        return

    is_file = isinstance(SOURCE, str) and not SOURCE.lower().startswith(("rtsp", "http"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    conf_min = min(CONF_PERSON, CONF_PERSON_TRACKED, CONF_BAG)
    tracker_cfg = make_tracker_cfg(round(TRACK_BUFFER_S * fps))   # [BARU]

    bags, id2key = {}, {}
    pm = PersonMemory()                                           # [BARU]
    next_key, frame_idx, last_t = 0, 0, None

    legend1 = (f"Attended: <={R_ATTEND:g}m | Terpisah sementara: >{R_ATTEND:g}m, <{T_SUSPICIOUS:g}s | "
               f"Perlu perhatian: >{R_ATTEND:g}m, >={T_SUSPICIOUS:g}s | "
               f"Alert: >{R_ABANDON:g}m, >={T_ABANDONED:g}s")
    legend2 = ("Kotak Biru=orang | Kotak Oranye=tas | "
               "f=alarm palsu  k=ditangani  q=keluar")

    print("Memulai pemutaran video... Tekan 'q' pada jendela video untuk keluar.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Video selesai atau gagal dibaca.")
            break

        t = frame_idx / fps if is_file else time.time()
        frame_idx += 1
        dt = 0.0 if last_t is None else min(max(t - last_t, 0.0), 1.0)
        last_t = t

        r = model.track(
            frame, persist=True, tracker=tracker_cfg,
            classes=[PERSON_CLASS, *BAG_CLASSES], conf=conf_min,
            imgsz=IMGSZ, agnostic_nms=True, verbose=False,
        )[0]

        person_dets, bag_dets = [], []
        if r.boxes is not None and r.boxes.id is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for box, tid, cls, cf in zip(xyxy, ids, clss, confs):
                if cls == PERSON_CLASS:
                    # [BARU] orang yang sudah dikenal boleh lolos dengan keyakinan lebih rendah
                    thr = CONF_PERSON_TRACKED if pm.known(int(tid)) else CONF_PERSON
                    if cf >= thr:
                        person_dets.append((int(tid), box))
                elif cls in BAG_CLASSES and cf >= CONF_BAG:
                    bag_dets.append((int(tid), box, BAG_CLASSES[cls]))

        # [BARU] ubah ID mentah menjadi ID stabil; dipanggil setiap frame
        persons, ghost_persons, relink_events, new_id_events = pm.resolve(person_dets, t)
        for ev in relink_events:
            print(f"[ID] t={ev[0]:.1f}s ID mentah #{ev[1]} disambung ke PERSON P{ev[2]:02d} "
                  f"(hilang {ev[4]:.1f}s, selisih posisi {ev[3]:.0f}px)")
            log_relink(ev)
        for ev in new_id_events:
            alasan = "; ".join(f"P{c:02d}: {why}" for c, why in ev[3])
            print(f"[ID] t={ev[0]:.1f}s ID mentah #{ev[1]} dianggap orang baru PERSON P{ev[2]:02d}; "
                  f"tidak disambung ke yang hilang -> {alasan}")
            log_new_id(ev)

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
            update_bag(b, persons, t, dt, ghost_persons)
            if DEBUG_PRINT:
                print(f"t={t:6.1f}s orang={[p for p, _ in persons]} hantu={[p for p, _ in ghost_persons]} "
                      f"penjaga={sorted(b.guardians)} d={b.d_guard} status={b.state}")
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

        for pid, pbox in persons:
            draw_person(out, pbox, f"PERSON P{pid:02d}")

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
    print(f"Selesai. Total ID orang yang disambung: {len(pm.relinks)} "
          f"(rincian di {LOG_ID_PATH})")


if __name__ == "__main__":
    main()