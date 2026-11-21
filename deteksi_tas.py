import csv
import math
import os
import time
from collections import Counter
from datetime import datetime, timedelta

import cv2
import numpy as np
from ultralytics import YOLO

# ===================== PENGATURAN =====================
SOURCE = "video_uji16.mp4"    # 0 = webcam, atau path video / URL RTSP
MODEL_PATH = "yolov8n.pt"     # Gunakan yolov8n.pt agar lebih ringan di CPU
IMGSZ = 640                   # Turunkan ke 640 agar proses per frame jauh lebih cepat
CONF_PERSON = 0.35            # ambang untuk orang BARU
CONF_PERSON_TRACKED = 0.20    # ambang lebih longgar untuk orang yang identitasnya sudah dikenal
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
BAG_MEMORY_S = 60.0 # detik; tas tak terlihat selama ini baru dilupakan statusnya (ID-nya tetap dicadangkan)
GHOST_DRAW = 3.0    # detik; kotak tas yang sesaat tak terdeteksi tetap digambar

# --- Kesinambungan identitas orang (mengatasi identity switch) ---
LOCK_PERSON_ID = True           # True = ID orang yang sudah terkonfirmasi dikunci pada jangkarnya
LOCK_ANCHOR_RADIUS_PX = 95.0    # radius (px) untuk mengunci kotak ke pemilik ID yang sudah terkunci
LOCK_MIN_IOU = 0.20             # tumpang tindih minimum dengan kotak acuan untuk tetap mengunci ID
STATIONARY_MAX_PX = 60.0        # batas geser pusat kotak (px) untuk menganggap orang duduk/diam
TRACK_BUFFER_S = 3.0            # s. ByteTrack mengingat track yang hilang selama ini
RELINK_S = 8.0                  # s. ID baru boleh disambung ke orang yang hilang paling lama sekian detik lalu
RELINK_DIST_H = 0.6             # jarak maks pusat kotak, dalam satuan tinggi badan
RELINK_SPEED_GAIN = 1.5         # tambahan jarak berdasarkan kecepatan
EDGE_PX = 10                    # orang yang terakhir terlihat menempel di tepi kiri/kanan layar
RELINK_SIZE_TOL = 1.6           # rasio tinggi kotak lama vs baru
RELINK_IOU = 0.30               # kotak baru yang menumpuk >= ini dengan kotak acuan lama
OCC_OVERLAP = 0.15              # kotak yang tertutup orang lain >= ini dianggap terhalang
REF_STALE_S = 4.0               # s. pembaruan kotak acuan jika terhalang terlalu lama
RELINK_SHRINK = 0.75            # kotak yang tingginya < ini x kotak acuan dianggap kotak sebagian
GHOST_PERSON_S = 2.5            # s. orang yang hilang sesaat tetap dihitung di posisi terakhirnya

# --- ID terkunci selama satu sesi ---
SESSION_S = 24 * 3600   # s. Selama sesi ini ID person dan tas TIDAK dipakai ulang
REPORT_DIR = "laporan"
PERSON_CONFIRM_S = 0.6  # s. orang baru diberi ID setelah terdeteksi selama ini
BAG_CONFIRM_S = 1.0     # s. tas baru diberi ID setelah terdeteksi selama ini
TENT_EXPIRE_S = 1.0     # s. kandidat yang tak terdeteksi selama ini dibuang
TENT_DIST_H = 0.5       # jarak maks antar deteksi kandidat orang, dalam satuan tinggi badan
DUP_IOU = 0.6           # kotak yang menumpuk >= ini dianggap deteksi ganda

# --- Pengenalan ulang orang yang kembali setelah lama pergi (EKSPERIMENTAL, bawaan MATI) ---
REID_ENABLED = False
REID_MIN = 0.985
REID_MARGIN = 0.01

DEBUG_PRINT = False

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
STATE_RANK = {"ATTENDED": 0, "TEMP_UNATTENDED": 1, "SUSPICIOUS": 2, "ABANDONED": 3}
STATE_BY_RANK = {v: k for k, v in STATE_RANK.items()}

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


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def covered_frac(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    return inter / max((a[2] - a[0]) * (a[3] - a[1]), 1.0)


def appearance(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = x2 - x1, y2 - y1
    if w < 20 or h < 40:
        return None
    fh, fw = frame.shape[:2]
    ya, yb = max(int(y1 + 0.15 * h), 0), min(int(y1 + 0.55 * h), fh)
    xa, xb = max(int(x1 + 0.2 * w), 0), min(int(x2 - 0.2 * w), fw)
    if yb - ya < 8 or xb - xa < 8:
        return None
    c = frame[ya:yb:3, xa:xb:3, ::-1].reshape(-1, 3).astype(int) // 64
    hist = np.bincount(c[:, 0] * 16 + c[:, 1] * 4 + c[:, 2], minlength=64).astype(float)
    return hist / hist.sum()


class Tentative:
    def __init__(self, need_hits):
        self.need = max(1, int(need_hits))
        self.items = []

    def hit(self, box, t, radius):
        cx, cy = center(box)
        best, found = radius, None
        for it in self.items:
            if it["t"] == t:
                continue
            d = math.hypot(cx - it["c"][0], cy - it["c"][1])
            if d < best:
                best, found = d, it
        if found is None:
            found = {"c": (cx, cy), "hits": 0, "t": t}
            self.items.append(found)
        found["c"], found["t"] = (cx, cy), t
        found["hits"] += 1
        if found["hits"] >= self.need:
            self.items.remove(found)
            return True
        return False

    def expire(self, t):
        self.items = [it for it in self.items if t - it["t"] <= TENT_EXPIRE_S]


class PersonMemory:
    def __init__(self, fps):
        self.raw2sid = {}        # ID mentah ByteTrack -> ID stabil
        self.locked_sids = set() # ID stabil yang sudah terkonfirmasi dan dikunci
        self.ref = {}            # ID stabil -> kotak acuan terakhir yang valid
        self.anchor = {}         # ID stabil -> posisi jangkar terkunci
        self.ref_t = {}          # ID stabil -> waktu kotak acuan diperbarui
        self.seen_t = {}         # ID stabil -> waktu terakhir terlihat
        self.app = {}            # ID stabil -> histogram warna
        self.vel = {}            # ID stabil -> kecepatan terakhir (px/s)
        self.samp = {}           # ID stabil -> (posisi, waktu)
        self.edge = {}           # ID stabil -> True bila di tepi layar
        self.records = {}        # ID stabil -> catatan rekap
        self.next_sid = 0
        self.relinks = []
        self.tent = Tentative(round(PERSON_CONFIRM_S * fps))
        self._prev_t = None

    def known(self, raw_id):
        return raw_id in self.raw2sid

    def _rec(self, sid, t):
        return self.records.setdefault(
            sid, {"first_t": t, "last_t": t, "seen_s": 0.0, "relinks": 0, "reid": 0})

    def _fits_locked_anchor(self, sid, box):
        """Cek apakah kotak berada di jangkar / area orang yang sudah dikunci."""
        if sid not in self.anchor:
            return False
        cx, cy = center(box)
        ax, ay = self.anchor[sid]
        d_anchor = math.hypot(cx - ax, cy - ay)
        iou_ref = box_iou(box, self.ref[sid]) if sid in self.ref else 0.0
        return d_anchor <= LOCK_ANCHOR_RADIUS_PX or iou_ref >= LOCK_MIN_IOU

    def _match(self, box, t, sid):
        elapsed = t - self.seen_t[sid]
        if elapsed <= 0:
            return None, "masih terlihat"
        if elapsed > RELINK_S:
            return None, f"sudah hilang {elapsed:.1f}s (batas {RELINK_S:g}s)"
        if self.edge.get(sid):
            return None, "terakhir terlihat di tepi layar (dianggap keluar)"
        rbox = self.ref[sid]
        lh = max(rbox[3] - rbox[1], 1.0)
        nh = max(box[3] - box[1], 1.0)
        ratio = nh / lh
        iou = box_iou(box, rbox)
        cx, cy = center(box)
        lx, ly = center(rbox)
        dist = math.hypot(cx - lx, cy - ly)
        allowed = lh * RELINK_DIST_H + RELINK_SPEED_GAIN * self.vel.get(sid, 0.0) * elapsed
        if iou < RELINK_IOU:
            if ratio > RELINK_SIZE_TOL or ratio < 1.0 / RELINK_SIZE_TOL:
                return None, f"ukuran beda (x{ratio:.1f})"
            if dist > allowed:
                return None, f"posisi terlalu jauh ({dist / lh:.1f}x tinggi badan)"
        cost = dist / allowed + abs(math.log(ratio)) + (1.0 - iou)
        return (cost, dist, elapsed), "cocok"

    def _reid(self, frame, box, t, used):
        if not REID_ENABLED or frame is None:
            return None
        a = appearance(frame, box)
        if a is None:
            return None
        scores = []
        for sid, ap in self.app.items():
            if sid in used or t - self.seen_t[sid] <= RELINK_S:
                continue
            scores.append((float(np.sqrt(a * ap).sum()), sid))
        if not scores:
            return None
        scores.sort(reverse=True)
        best, sid = scores[0]
        second = scores[1][0] if len(scores) > 1 else 0.0
        if best >= REID_MIN and best - second >= REID_MARGIN:
            return sid, best, t - self.seen_t[sid]
        return None

    def resolve(self, dets, t, frame=None):
        dt = 0.0 if self._prev_t is None else min(max(t - self._prev_t, 0.0), 1.0)
        self._prev_t = t
        n_before = len(self.relinks)
        visible, used, pending = [], set(), []

        # Tahap 1: Cegah ID terkunci direbut oleh orang lain yang bergerak melintas.
        # Jika raw_id menunjuk ke sid yang sudah dikunci, tetapi kotaknya jauh dari jangkar sid,
        # lepaskan dulu agar tidak mencuri ID orang yang duduk/terkunci.
        candidates_for_lock = []
        for raw, box in dets:
            sid = self.raw2sid.get(raw)
            if sid is not None and sid not in used:
                if LOCK_PERSON_ID and sid in self.locked_sids and not self._fits_locked_anchor(sid, box):
                    del self.raw2sid[raw]
                    pending.append((raw, box))
                    candidates_for_lock.append((raw, box))
                    continue
                used.add(sid)
                visible.append((sid, box))
            else:
                if sid is not None:
                    del self.raw2sid[raw]
                pending.append((raw, box))
                candidates_for_lock.append((raw, box))

        # Tahap 1b: Jika ada sid yang sudah terkunci namun belum terpasang di frame ini
        # (misalnya karena tid ByteTrack tertukar saat orang lewat), kembalikan sid
        # ke kotak yang paling cocok dengan jangkarnya terlebih dahulu.
        if LOCK_PERSON_ID:
            locked_candidates = []
            for sid in sorted(self.locked_sids):
                if sid in used or sid not in self.anchor:
                    continue
                ax, ay = self.anchor[sid]
                best_idx, best_d = None, LOCK_ANCHOR_RADIUS_PX
                for idx, (raw, box) in enumerate(pending):
                    if any(box_iou(box, vb) >= DUP_IOU for _, vb in visible):
                        continue
                    cx, cy = center(box)
                    d = math.hypot(cx - ax, cy - ay)
                    iou_r = box_iou(box, self.ref[sid]) if sid in self.ref else 0.0
                    if d <= best_d or iou_r >= LOCK_MIN_IOU:
                        score = d - (100.0 * iou_r)
                        if score < best_d:
                            best_d = score
                            best_idx = idx
                if best_idx is not None:
                    raw, box = pending.pop(best_idx)
                    self.raw2sid[raw] = sid
                    used.add(sid)
                    visible.append((sid, box))

        # Buang deteksi ganda
        kept = []
        for raw, box in pending:
            if any(box_iou(box, vb) >= DUP_IOU for _, vb in visible) or \
               any(box_iou(box, kb) >= DUP_IOU for _, kb in kept):
                continue
            kept.append((raw, box))
        pending = kept

        # Tahap 2: Sambungkan sisa deteksi ke orang yang baru hilang
        pairs = []
        for pi, (raw, box) in enumerate(pending):
            for sid in self.seen_t:
                if sid in used:
                    continue
                res, _ = self._match(box, t, sid)
                if res is not None:
                    lock_bonus = -10.0 if sid in self.locked_sids else 0.0
                    pairs.append((res[0] + lock_bonus, pi, sid, res[1], res[2]))
        pairs.sort()
        assigned = {}
        for _, pi, sid, dist, elapsed in pairs:
            if pi in assigned or sid in used:
                continue
            assigned[pi] = sid
            used.add(sid)
            self.relinks.append((t, pending[pi][0], sid, dist, elapsed))

        reids, new_ids = [], []
        for pi, (raw, box) in enumerate(pending):
            if pi in assigned:
                sid = assigned[pi]
                self._rec(sid, t)["relinks"] += 1
            else:
                h = max(box[3] - box[1], 1.0)
                if not self.tent.hit(box, t, TENT_DIST_H * h):
                    continue
                found = self._reid(frame, box, t, used)
                if found is not None:
                    sid = found[0]
                    reids.append((t, raw, sid, found[1], found[2]))
                    self._rec(sid, t)["reid"] += 1
                else:
                    reasons = []
                    for cand in self.seen_t:
                        if cand not in used and 0 < t - self.seen_t[cand] <= RELINK_S:
                            reasons.append((cand, self._match(box, t, cand)[1]))
                    self.next_sid += 1
                    sid = self.next_sid
                    if reasons:
                        new_ids.append((t, raw, sid, reasons))
                used.add(sid)
            self.raw2sid[raw] = sid
            visible.append((sid, box))
        self.tent.expire(t)

        # Perbarui ingatan & kunci ID orang yang sudah terkonfirmasi
        fw = frame.shape[1] if frame is not None else None
        for sid, box in visible:
            occluded = any(covered_frac(box, other) >= OCC_OVERLAP
                           for s2, other in visible if s2 != sid)
            if sid in self.ref:
                rh = max(self.ref[sid][3] - self.ref[sid][1], 1.0)
                shrunk = (box[3] - box[1]) < RELINK_SHRINK * rh
            else:
                shrunk = False
            stale = (t - self.ref_t.get(sid, -1e9)) > REF_STALE_S
            if sid not in self.ref or stale or not (occluded or shrunk):
                self.ref[sid] = box
                self.ref_t[sid] = t

            if sid not in self.anchor:
                self.anchor[sid] = center(box)
            elif not (occluded or shrunk):
                # Jika orang duduk/diam (geser <= STATIONARY_MAX_PX), pertahankan jangkar awal
                # agar tidak terseret jika ada orang lewat di depannya
                c = center(box)
                ax, ay = self.anchor[sid]
                if math.hypot(c[0] - ax, c[1] - ay) > STATIONARY_MAX_PX and self.vel.get(sid, 0.0) > 25.0:
                    self.anchor[sid] = c

            if not (occluded or shrunk):
                c = center(box)
                sc, st0 = self.samp.get(sid, (None, None))
                if sc is None:
                    self.samp[sid] = (c, t)
                elif t - st0 >= 0.5:
                    v = math.hypot(c[0] - sc[0], c[1] - sc[1]) / (t - st0)
                    self.vel[sid] = 0.5 * self.vel.get(sid, 0.0) + 0.5 * v
                    self.samp[sid] = (c, t)
                if fw is not None:
                    self.edge[sid] = box[0] <= EDGE_PX or box[2] >= fw - EDGE_PX

            if REID_ENABLED and frame is not None and not (occluded or shrunk):
                a = appearance(frame, box)
                if a is not None:
                    self.app[sid] = a if sid not in self.app else 0.9 * self.app[sid] + 0.1 * a

            self.seen_t[sid] = t
            rec = self._rec(sid, t)
            rec["last_t"] = t
            rec["seen_s"] += dt
            if rec["seen_s"] >= PERSON_CONFIRM_S * 0.5:
                self.locked_sids.add(sid)

        ghosts = [(sid, self.ref[sid]) for sid, st in self.seen_t.items()
                  if sid not in used and 0 < t - st <= GHOST_PERSON_S and not self.edge.get(sid)]
        return visible, ghosts, self.relinks[n_before:], reids, new_ids


class Session:
    def __init__(self, t0, fps, number):
        self.number = number
        self.t0 = t0
        self.clock0 = datetime.now()
        self.pm = PersonMemory(fps)
        self.bags = {}
        self.id2key = {}
        self.next_key = 0
        self.bag_rec = {}
        self.bag_tent = Tentative(round(BAG_CONFIRM_S * fps))
        self.counts = Counter()

    def clock(self, t):
        return self.clock0 + timedelta(seconds=t - self.t0)


def create_bag(S, tid, box, label, t, persons):
    S.next_key += 1
    key = S.next_key
    owner, d = nearest(box, persons)
    S.bags[key] = Bag(key, box, label, t, owner if d <= R_ATTEND else None)
    S.id2key[tid] = key
    S.bag_rec[key] = {"label": label, "first_t": t, "last_t": t, "seen_s": 0.0,
                      "max_state": 0, "alert": False, "review": ""}
    return key


def write_report(S, t_end, alasan):
    os.makedirs(REPORT_DIR, exist_ok=True)
    base = os.path.join(REPORT_DIR, f"rekap_{S.clock0.strftime('%Y%m%d_%H%M%S')}_sesi{S.number}")
    fmt = lambda tt: S.clock(tt).strftime("%Y-%m-%d %H:%M:%S")

    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["jenis", "id", "pertama_terlihat", "terakhir_terlihat", "lama_terlihat_s",
                    "penyambungan_id", "jenis_tas", "status_tertinggi", "alarm_abandoned", "keputusan_satpam"])
        for sid, r in sorted(S.pm.records.items()):
            w.writerow(["PERSON", f"P{sid:02d}", fmt(r["first_t"]), fmt(r["last_t"]),
                        round(r["seen_s"], 1), r["relinks"] + r["reid"], "", "", "", ""])
        for key, r in sorted(S.bag_rec.items()):
            w.writerow(["LUGGAGE", f"L{key:02d}", fmt(r["first_t"]), fmt(r["last_t"]),
                        round(r["seen_s"], 1), "", r["label"],
                        STATES[STATE_BY_RANK[r["max_state"]]][0],
                        "ya" if r["alert"] else "tidak", r["review"]])

    n_person, n_bag = len(S.pm.records), len(S.bag_rec)
    per_jenis = Counter(r["label"] for r in S.bag_rec.values())
    n_susp = sum(1 for r in S.bag_rec.values() if r["max_state"] >= STATE_RANK["SUSPICIOUS"])
    n_aband = sum(1 for r in S.bag_rec.values() if r["alert"])
    n_relink = sum(r["relinks"] for r in S.pm.records.values())
    n_reid = sum(r["reid"] for r in S.pm.records.values())
    dur = int(max(t_end - S.t0, 0))
    lines = [
        f"REKAP SESI {S.number}",
        f"Sumber       : {SOURCE}",
        f"Periode      : {fmt(S.t0)} s.d. {fmt(t_end)} (durasi {dur // 3600}j {dur % 3600 // 60}m {dur % 60}d)",
        f"Sesi berakhir: {alasan}",
        "",
        f"Jumlah ID person : {n_person}",
        f"Jumlah ID tas    : {n_bag}" + ("  (" + ", ".join(f"{k}: {v}" for k, v in per_jenis.items()) + ")" if per_jenis else ""),
        f"Tas yang mencapai 'Perlu perhatian' atau lebih : {n_susp}",
        f"Tas yang memicu SECURITY ALERT                 : {n_aband}",
        f"Keputusan petugas: alarm palsu {S.counts['FALSE_ALARM']}, ditangani {S.counts['CONFIRMED']}",
        f"ID orang yang disambung kembali : {n_relink}",
        f"Orang dikenali ulang (ReID)     : {n_reid}" + ("" if REID_ENABLED else "  (fitur mati)"),
        "",
        "Catatan: jumlah ID adalah perkiraan jumlah identitas unik. Orang yang keluar lalu kembali",
        "tercatat dengan ID baru bila tidak dapat dikenali ulang. Rincian per ID ada di file CSV.",
        "Pada sesi berikutnya nomor ID dimulai lagi dari 01.",
    ]
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[REKAP] tersimpan: {base}.txt dan {base}.csv")
    return base


def fmt_durasi(detik):
    return f"{detik / 3600:g} jam" if detik >= 3600 else f"{detik:g} detik"


def make_tracker_cfg(buffer_frames):
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
    ghost_d = {pid: dist_m(bag.box, pbox) for pid, pbox in ghosts}
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
        if pid not in dists and pid not in ghost_d:
            bag.prox[pid] = 0.0

    guard_d = {p: dists[p] for p in bag.guardians if p in dists}
    for p in bag.guardians:
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


def review_flagged(S, decision, t):
    n = 0
    for b in S.bags.values():
        if b.state in ("SUSPICIOUS", "ABANDONED") and b.review is None:
            b.review = decision
            S.bag_rec[b.key]["review"] = "alarm_palsu" if decision == "FALSE_ALARM" else "ditangani"
            S.counts[decision] += 1
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
    t, raw, sid, dist, elapsed = ev
    _log_identitas([round(t, 1), raw, sid, "disambung",
                    f"hilang {elapsed:.1f}s; selisih posisi {dist:.0f}px"])


def log_reid(ev):
    t, raw, sid, score, elapsed = ev
    _log_identitas([round(t, 1), raw, sid, "dikenali_ulang",
                    f"hilang {elapsed:.0f}s; kemiripan warna {score:.3f}"])


def log_new_id(ev):
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
    tracker_cfg = make_tracker_cfg(round(TRACK_BUFFER_S * fps))

    S, frame_idx, last_t = None, 0, None

    legend1 = (f"Attended: <={R_ATTEND:g}m | Terpisah sementara: >{R_ATTEND:g}m, <{T_SUSPICIOUS:g}s | "
               f"Perlu perhatian: >{R_ATTEND:g}m, >={T_SUSPICIOUS:g}s | "
               f"Alert: >{R_ABANDON:g}m, >={T_ABANDONED:g}s")
    legend2 = ("Kotak Biru=orang | Kotak Oranye=tas | "
               "f=alarm palsu  k=ditangani  q=keluar")

    print("Memulai pemutaran video... Tekan 'q' pada jendela video untuk keluar.")

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Video selesai atau gagal dibaca.")
                break

            t = frame_idx / fps if is_file else time.time()
            frame_idx += 1

            if S is None:
                S = Session(t, fps, 1)
                print(f"[SESI 1] dimulai {S.clock0:%Y-%m-%d %H:%M:%S}; ID direset setiap {fmt_durasi(SESSION_S)}.")
            elif t - S.t0 >= SESSION_S:
                write_report(S, t, f"batas {fmt_durasi(SESSION_S)} tercapai")
                S = Session(t, fps, S.number + 1)
                print(f"[SESI {S.number}] dimulai; ID person dan tas mulai lagi dari 01.")

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
                        thr = CONF_PERSON_TRACKED if S.pm.known(int(tid)) else CONF_PERSON
                        if cf >= thr:
                            person_dets.append((int(tid), box))
                    elif cls in BAG_CLASSES and cf >= CONF_BAG:
                        bag_dets.append((int(tid), box, BAG_CLASSES[cls]))

            persons, ghost_persons, relink_events, reid_events, new_id_events = S.pm.resolve(person_dets, t, frame)
            for ev in relink_events:
                print(f"[ID] t={ev[0]:.1f}s ID mentah #{ev[1]} disambung ke PERSON P{ev[2]:02d} "
                      f"(hilang {ev[4]:.1f}s, selisih posisi {ev[3]:.0f}px)")
                log_relink(ev)
            for ev in reid_events:
                print(f"[ID] t={ev[0]:.1f}s ID mentah #{ev[1]} dikenali ulang sebagai PERSON P{ev[2]:02d} "
                      f"(pergi {ev[4]:.0f}s, kemiripan warna {ev[3]:.3f})")
                log_reid(ev)
            for ev in new_id_events:
                alasan = "; ".join(f"P{c:02d}: {why}" for c, why in ev[3])
                print(f"[ID] t={ev[0]:.1f}s ID mentah #{ev[1]} jadi orang baru PERSON P{ev[2]:02d}; "
                      f"tidak disambung ke yang hilang -> {alasan}")
                log_new_id(ev)

            seen = set()
            for tid, box, label in bag_dets:
                key = S.id2key.get(tid)
                if key not in S.bags or key in seen:
                    key, best = None, MATCH_PX
                    cb = center(box)
                    for k, b in S.bags.items():
                        if k in seen:
                            continue
                        cb2 = center(b.box)
                        d = math.hypot(cb[0] - cb2[0], cb[1] - cb2[1])
                        if d < best:
                            key, best = k, d
                    if key is None:
                        if S.bag_tent.hit(box, t, MATCH_PX):
                            key = create_bag(S, tid, box, label, t, persons)
                        else:
                            continue
                    S.id2key[tid] = key
                bag = S.bags[key]
                bag.box, bag.label, bag.last_seen = box, label, t
                seen.add(key)
            S.bag_tent.expire(t)

            for key in seen:
                b = S.bags[key]
                update_bag(b, persons, t, dt, ghost_persons)
                rec = S.bag_rec[key]
                rec["last_t"] = t
                rec["seen_s"] += dt
                rec["max_state"] = max(rec["max_state"], STATE_RANK[b.state])
                if DEBUG_PRINT:
                    print(f"t={t:6.1f}s orang={[p for p, _ in persons]} hantu={[p for p, _ in ghost_persons]} "
                          f"penjaga={sorted(b.guardians)} d={b.d_guard} status={b.state}")
                if b.state != b.logged_state:
                    if not (b.logged_state is None and b.state == "ATTENDED"):
                        log_event(t, b, b.state)
                    b.logged_state = b.state
            for key in [k for k, b in S.bags.items() if t - b.last_seen > BAG_MEMORY_S]:
                del S.bags[key]

            out = frame.copy()
            h, w = out.shape[:2]
            persons_by_id = dict(persons)
            visible = [b for b in S.bags.values() if t - b.last_seen <= GHOST_DRAW]

            for pid, pbox in persons:
                lock_tag = " [LOCK]" if LOCK_PERSON_ID and pid in S.pm.locked_sids else ""
                draw_person(out, pbox, f"PERSON P{pid:02d}{lock_tag}")

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
                S.bag_rec[bag.key]["alert"] = True
                S.counts["ALERT"] += 1
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
                print(f"[SATPAM] alarm palsu: {review_flagged(S, 'FALSE_ALARM', t)} tas")
            elif key_pressed == ord("k"):
                print(f"[SATPAM] ditangani: {review_flagged(S, 'CONFIRMED', t)} tas")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if S is not None:
            write_report(S, last_t if last_t is not None else S.t0,
                         "program dihentikan atau video selesai (rekap sementara)")


if __name__ == "__main__":
    main()