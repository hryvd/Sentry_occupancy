#!/usr/bin/env python3
"""
change_monitor.py - lightweight two-tier change monitor for one camera/area (B2, B3, B4 ...).

QUIET   no significant vehicle change -> NO occupancy POSTs, only a device_heartbeat every 60 s.
ACTIVE  a bay's picture changed and stayed changed -> run the heavy 4-model fusion (inference hook) and POST the
        full final-state of every slot, repeating every `active_cycle_s` (<= 30 s, so >= 2 full updates/min)
        until `settle_cycles` consecutive cycles show no state change, then return to QUIET.
STALE   no fresh frame for > freshness_threshold_s (30 s) -> every slot is emitted as unknown / confidence null.

Change detection is cheap and illumination-robust (building shadows sweep this lot): each bay ROI is compared with
the reference by normalised cross-correlation on a small greyscale thumbnail, so a shadow or exposure shift keeps
the score near 0 while a car arriving or leaving drives it up. A change must persist `persist_frames` consecutive
frames, which filters walking pedestrians and umbrellas.

The wire envelope mirrors the Q&A contract (CANDIDATE-2026-10-30.2). If parking_occupancy.events is importable you
should swap `build_event` for its build_occupancy_event / build_heartbeat_event; this fallback only exists so the
monitor runs standalone.

    python scripts/lot_b3/change_monitor.py --selftest
    python scripts/lot_b3/change_monitor.py --polygons configs/polygons/area_b3.json --source rtsp://... --out events.jsonl
"""
import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np

DEFAULTS = dict(poll_interval_s=2.0, heartbeat_interval_s=60.0, active_cycle_s=20.0, settle_cycles=2,
                freshness_threshold_s=30.0, persist_frames=3, change_threshold=0.35, thumb=(32, 32),
                refresh_interval_s=0.0)         # 0 = strictly change-driven (no anti-entropy refresh)


def ncc_change(ta, tb):
    """1 - normalised cross-correlation of two (thumbnail, mask) pairs, taken over the bay polygon only
    (0 = same picture, ~1 = unrelated). Pixels outside the polygon would otherwise dominate the score."""
    (a, ma), (b, _) = ta, tb
    a = a[ma].astype(np.float32)
    b = b[ma].astype(np.float32)
    if a.size < 4:
        return 0.0
    a0, b0 = a - a.mean(), b - b.mean()
    den = np.linalg.norm(a0) * np.linalg.norm(b0)
    if np.linalg.norm(a0) < 1.0 or np.linalg.norm(b0) < 1.0:   # flat patch (bare pavement): brightness-relative test
        return float(abs(a.mean() / max(b.mean(), 1) - 1) > 0.35 or abs(np.linalg.norm(a0) - np.linalg.norm(b0)) > 8)
    return float(1.0 - np.dot(a0, b0) / den)


class Monitor:
    def __init__(self, cfg, polygons, inference, emit, clock=time.monotonic, **over):
        self.p = {**DEFAULTS, **over}
        self.cfg, self.inference, self.emit, self.clock = cfg, inference, emit, clock
        self.slots = [s for s in cfg["slots"] if s.get("visibility", "full") == "full"]
        self.boot_id, self.seq = str(uuid.uuid4()), 0
        self.ref = None                 # reference thumbnails per slot
        self.streak = {s["slot_id"]: 0 for s in self.slots}
        self.mode, self.stable = "QUIET", 0
        self.last_state = {}
        t = clock()
        self.last_frame_t = t
        self.next_heartbeat, self.next_cycle, self.next_refresh = t, None, None
        self.stale = False

    # ---- helpers -------------------------------------------------------------------------------
    def thumbs(self, frame):
        g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        out = {}
        for s in self.slots:
            pts = np.array(s["polygon_px"], np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            H, W = g.shape
            x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)   # polygon may leave the frame
            m = np.zeros(g.shape, np.uint8)
            cv2.fillPoly(m, [pts], 255)
            roi = cv2.bitwise_and(g, g, mask=m)[y0:y1, x0:x1]
            mk = m[y0:y1, x0:x1]
            if roi.size == 0:
                roi, mk = np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint8)
            tm = cv2.resize(mk, self.p["thumb"], interpolation=cv2.INTER_NEAREST) > 0
            out[s["slot_id"]] = (cv2.resize(roi, self.p["thumb"], interpolation=cv2.INTER_AREA), tm)
        return out

    def envelope(self, etype, data):
        self.seq += 1
        return {"event_id": str(uuid.uuid4()), "schema_version": "1.0.0", "event_type": etype,
                "occurred_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "producer_id": self.cfg["producer_id"], "correlation_id": None, "data": data}

    def heartbeat(self):
        self.emit(self.envelope("device_heartbeat", {"boot_id": self.boot_id, "camera_id": self.cfg["camera_id"],
                                                     "mode": self.mode}))

    def full_update(self, frame, reason):
        """Run heavy fusion and POST every slot. inference(frame) -> {slot_id: (state, confidence)}."""
        res = self.inference(frame)
        changed = res != self.last_state
        self.last_state = res
        for s in self.cfg["slots"]:
            st, conf = res.get(s["slot_id"], ("unknown", None))
            if st == "unknown":
                conf = None
            self.emit(self.envelope("occupancy_observation", {
                "lot_code": self.cfg["lot_code"], "space_id": s["space_id"], "physical_status": st,
                "confidence": conf, "boot_id": self.boot_id, "sequence_number": self.seq + 1,
                "calibration_version": self.cfg["calibration_version"], "model_version": "sentry-4model-b2-b4-v1",
                "reason": reason}))
        return changed

    # ---- main step -----------------------------------------------------------------------------
    def step(self, frame):
        """Call once per poll. frame=None means the source gave nothing this poll."""
        now = self.clock()
        if frame is not None:
            self.last_frame_t, self.stale = now, False
        elif now - self.last_frame_t > self.p["freshness_threshold_s"] and not self.stale:
            self.stale = True                                   # silent camera: everything becomes unknown
            self.full_update_unknown()
        if now >= self.next_heartbeat:                          # heartbeats continue even when stale/quiet
            self.heartbeat()
            self.next_heartbeat = now + self.p["heartbeat_interval_s"]
        if frame is None:
            return
        th = self.thumbs(frame)
        if self.ref is None:                                    # first frame: establish baseline state
            self.ref = th
            self.full_update(frame, "startup")
            return
        moved = []
        for sid, t in th.items():
            if ncc_change(t, self.ref[sid]) > self.p["change_threshold"]:
                self.streak[sid] += 1
            else:
                self.streak[sid] = 0
            if self.streak[sid] >= self.p["persist_frames"]:
                moved.append(sid)
        if self.mode == "QUIET":
            if moved:
                self.mode, self.stable = "ACTIVE", 0
                changed = self.full_update(frame, "change:" + ",".join(moved))
                self.next_cycle = now + self.p["active_cycle_s"]
                self.ref = th
                self.streak = {k: 0 for k in self.streak}
            elif self.p["refresh_interval_s"] and now >= (self.next_refresh or 0):
                self.full_update(frame, "anti_entropy")
                self.next_refresh = now + self.p["refresh_interval_s"]
        else:                                                   # ACTIVE: keep refreshing until it settles
            if now >= self.next_cycle:
                changed = self.full_update(frame, "active_cycle")
                self.next_cycle = now + self.p["active_cycle_s"]
                self.stable = 0 if (changed or moved) else self.stable + 1
                self.ref = th
                self.streak = {k: 0 for k in self.streak}
                if self.stable >= self.p["settle_cycles"]:
                    self.mode = "QUIET"

    def full_update_unknown(self):
        for s in self.cfg["slots"]:
            self.emit(self.envelope("occupancy_observation", {
                "lot_code": self.cfg["lot_code"], "space_id": s["space_id"], "physical_status": "unknown",
                "confidence": None, "boot_id": self.boot_id, "sequence_number": self.seq + 1,
                "calibration_version": self.cfg["calibration_version"], "model_version": "sentry-4model-b2-b4-v1",
                "reason": "stale_frame"}))
        self.last_state = {}


# ---- source / CLI --------------------------------------------------------------------------------
def baseline_inference(cfg):
    """Stand-in for the 4-model fusion: returns the manual baseline. Replace with fusion_4model per frame."""
    res = {s["slot_id"]: (s["baseline_state"], None if s["baseline_state"] == "unknown" else 0.95)
           for s in cfg["slots"]}
    return lambda frame: res


def selftest(root):
    cfg = json.load(open(os.path.join(root, "configs/polygons/area_b3.json")))
    base = cv2.imread(os.path.join(root, cfg["reference_image"]))
    ev, t, stamp = [], [0.0], []
    def sink(e):
        ev.append(e)
        if e["event_type"] == "occupancy_observation" and e["data"]["space_id"] == cfg["slots"][0]["space_id"]:
            stamp.append(t[0])                                         # one stamp per full update
    L5 = next(s for s in cfg["slots"] if s["slot_id"] == "L5")
    pts = np.array(L5["polygon_px"], np.int32)
    gone = base.copy()
    cv2.fillPoly(gone, [pts], (150, 150, 150))                         # L5 car leaves: bare pavement
    gone = cv2.GaussianBlur(gone, (9, 9), 0)
    shadow = (base * 0.7).astype(np.uint8)                             # global exposure/shadow shift
    walker = base.copy()
    R3 = np.array(next(s for s in cfg["slots"] if s["slot_id"] == "R3")["polygon_px"], np.int32)
    cv2.fillPoly(walker, [R3], (40, 40, 40))                           # pedestrian passes for 1 frame
    state = {"now": dict()}
    inferred = baseline_inference(cfg)
    flip = {"on": False}

    def inf(frame):
        r = dict(inferred(frame))
        if flip["on"]:
            r["L5"] = ("available", 0.93)
        return r
    m = Monitor(cfg, None, inf, sink, clock=lambda: t[0])

    def run(seconds, frame_fn, dt=2.0):
        end = t[0] + seconds
        while t[0] < end:
            m.step(frame_fn(t[0]))
            t[0] += dt

    def count(kind=None):
        return [e for e in ev if kind is None or e["event_type"] == kind]
    n_slots = len(cfg["slots"])
    run(2, lambda _: base)                                             # startup
    assert len(count("occupancy_observation")) == n_slots, "startup should emit one full state"
    mark = len(ev)
    run(300, lambda _: base)                                           # 5 quiet min
    quiet = ev[mark:]
    assert all(e["event_type"] == "device_heartbeat" for e in quiet), "QUIET must emit heartbeats only"
    hb = len(quiet)
    assert 4 <= hb <= 6, f"expected ~5 heartbeats in 5 quiet minutes, got {hb}"
    mark = len(ev)
    run(10, lambda _: shadow)                                          # exposure change must NOT trigger
    assert m.mode == "QUIET" and not count("occupancy_observation")[n_slots:], "shadow shift caused a false trigger"
    run(2, lambda _: walker)                                           # one-frame pedestrian
    run(10, lambda _: base)
    assert m.mode == "QUIET", "pedestrian blip caused a false trigger"
    flip["on"] = True
    t0 = t[0]
    run(100, lambda _: gone)                                           # L5 leaves and stays gone
    obs = [e for e in ev[mark:] if e["event_type"] == "occupancy_observation"]
    bursts = len(obs) // n_slots
    act = [x for x in stamp if x >= t0]
    gaps = [b - a for a, b in zip(act, act[1:])]
    assert bursts >= 3 and gaps and max(gaps) <= 30.0, f"active cadence must be <=30 s, got gaps {gaps}"
    assert m.mode == "QUIET", "should settle back to QUIET"
    l5 = [e for e in obs if e["data"]["space_id"] == L5["space_id"]]
    assert l5[0]["data"]["physical_status"] == "available"
    mark = len(ev)
    run(90, lambda _: None)                                            # camera goes silent
    unk = [e for e in ev[mark:] if e["event_type"] == "occupancy_observation"]
    assert len(unk) == n_slots and all(e["data"]["physical_status"] == "unknown" and e["data"]["confidence"] is None
                                       for e in unk), "stale camera must emit unknown/null for every slot"
    assert any(e["event_type"] == "device_heartbeat" for e in ev[mark:]), "heartbeats must continue when stale"
    seqs = [e["data"]["sequence_number"] for e in ev if e["event_type"] == "occupancy_observation"]
    print(f"SELFTEST OK: {n_slots} slots, {hb} quiet heartbeats, {bursts} full updates (max gap {max(gaps):.0f}s) while active, "
          f"stale -> unknown after silence")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--polygons")
    ap.add_argument("--source", help="video file, RTSP URL or camera index")
    ap.add_argument("--out", default="events.jsonl")
    a = ap.parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if a.selftest:
        return selftest(root)
    cfg = json.load(open(a.polygons))
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    f = open(a.out, "a")
    emit = lambda e: (f.write(json.dumps(e) + "\n"), f.flush())      # replace with HTTP POST + retry queue
    m = Monitor(cfg, None, baseline_inference(cfg), emit)          # replace inference with fusion_4model
    while True:
        ok, frame = cap.read()
        m.step(frame if ok else None)
        time.sleep(m.p["poll_interval_s"])


if __name__ == "__main__":
    sys.exit(main())
