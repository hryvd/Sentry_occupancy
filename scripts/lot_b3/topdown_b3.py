#!/usr/bin/env python3
"""
topdown_b3.py - ground-plane homography + top-down warp + slot plan for Area B3 (PROVISIONAL).

Homography: the painted accessible bay R10 is a 2.4 m wide rectangle (two parallel painted dividers, kerb-side
end and far-side end). Its four corners in the photo map to (0,0) (L, 0) (L, 2.4) (0, 2.4) metres, u = depth
away from the kerb end reversed (u = L at the kerb), v = along the kerb, top of photo = -v.
L (bay depth) is not painted anywhere we can see, so it is solved from a known size: the red Wigo in R9 has a
2.45 m wheelbase, and its two wheel-ground contacts must come out 2.45 m apart.

Outputs (outputs/lot_b3/): lot_b3_topview.png, lot_b3_slot_plan.png, and the matrix in configs/polygons/area_b3.homography.json
    python scripts/lot_b3/topdown_b3.py
"""
import json, os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMG = os.path.join(ROOT, "ui/samples/area_b3_raw.jpg")
PPM = 60                                                     # output pixels per metre
U0, U1, V0, V1 = -17.0, 5.5, -23.5, 6.5                      # plan window, metres
CORNERS = [(1500, 1475), (2140, 1605), (2110, 1862), (1380, 1672)]   # R10 bay: TL, TR, BR, BL in photo px
WIGO_WHEELS = [(1750, 1495), (2125, 1565)]                   # Wigo front / rear wheel ground contacts, photo px
WHEELBASE_M = 2.45
BAY_W = 2.4
GROUND_POLY = [(0, 1314), (0, 1160), (300, 1010), (700, 900), (1100, 760), (1300, 700), (1700, 740), (1850, 745),
               (2260, 745), (2215, 1420), (2185, 1932), (773, 1932), (773, 1455), (618, 1443)]


def homography(L):
    return cv2.getPerspectiveTransform(np.float32(CORNERS), np.float32([[0, 0], [L, 0], [L, BAY_W], [0, BAY_W]]))


def apply(H, pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def solve_L():
    f = lambda L: np.linalg.norm(np.diff(apply(homography(L), WIGO_WHEELS), axis=0)) - WHEELBASE_M
    lo, hi = 2.0, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def m2px(u, v):
    return ((np.asarray(u) - U0) * PPM, (np.asarray(v) - V0) * PPM)


# ---- regularised slot plan (metres; u right, v down, top of plan = top of photo) -------------------
LEFT_U = (-13.2, -8.2)                  # left bays, 5.0 m deep (their dividers end at u ~ -8.2 in the warp)
LEFT_V0, LEFT_PITCH = -19.2, 2.4        # L9 is level with R10 (both span v = 0..2.4), so L1 starts at -19.2
CENTRE_U = (-5.5, -2.9)                 # painted centre bay C1 (Camry): two lines 2.6 m apart
CENTRE_V = (-4.0, 1.6)
INFORMAL_M = {"A1": ("jeepney", (-6.3, -3.2, -18.0, -12.0)), "A2": ("sedan", (-6.3, -3.2, -11.0, -6.5)),
              "A3": ("sedan (top)", (-6.3, -3.2, -23.5, -20.0))}


def plan_slots(L):
    s = {}
    for k in range(1, 10):
        v0 = LEFT_V0 + LEFT_PITCH * (k - 1)
        s[f"L{k}"] = [(LEFT_U[0], v0), (LEFT_U[1], v0), (LEFT_U[1], v0 + LEFT_PITCH), (LEFT_U[0], v0 + LEFT_PITCH)]
    for k in range(1, 11):              # right row: 2.4 m pitch, R10 = v [0, 2.4] is the measured bay
        v0 = -2.4 - 2.4 * (9 - k) if k <= 9 else 0.0
        s[f"R{k}"] = [(0.0, v0), (L, v0), (L, v0 + BAY_W), (0.0, v0 + BAY_W)]
    s["C1"] = [(CENTRE_U[0], CENTRE_V[0]), (CENTRE_U[1], CENTRE_V[0]), (CENTRE_U[1], CENTRE_V[1]), (CENTRE_U[0], CENTRE_V[1])]
    return s


def font(sz, bold=True):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        if os.path.exists(f):
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


ZONE_COL = {"left": (112, 172, 232), "right": (242, 168, 114), "center": (120, 200, 200)}
STATE_COL = {"occupied": (215, 50, 50), "available": (40, 170, 70), "unknown": (140, 140, 140)}


def main():
    L = solve_L()
    H = homography(L)
    img = cv2.imread(IMG)
    Wp, Hp = int((U1 - U0) * PPM), int((V1 - V0) * PPM)
    S = np.array([[PPM, 0, -U0 * PPM], [0, PPM, -V0 * PPM], [0, 0, 1]], float)
    warped = cv2.warpPerspective(img, S @ H, (Wp, Hp), flags=cv2.INTER_LINEAR, borderValue=(28, 28, 28))
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.int32(GROUND_POLY)], 255)
    wm = cv2.warpPerspective(mask, S @ H, (Wp, Hp), flags=cv2.INTER_NEAREST)
    warped[wm == 0] = (28, 28, 28)
    cv2.imwrite(os.path.join(ROOT, "outputs/lot_b3/lot_b3_topview_raw.png"), warped)

    cfgp = os.path.join(ROOT, "configs/polygons/area_b3.json")
    cfg = json.load(open(cfgp))
    plan = plan_slots(L)
    inv = np.linalg.inv(H)
    state = {x["slot_id"]: x["baseline_state"] for x in cfg["slots"]}
    for x in cfg["slots"]:                                   # add ground-plane polygons (metres) + their photo-px footprint
        x["plan_polygon_m"] = [[round(a, 2), round(b, 2)] for a, b in plan[x["slot_id"]]]
        x["ground_polygon_px"] = [[round(a), round(b)] for a, b in apply(inv, plan[x["slot_id"]])]
    cfg["homography"] = "configs/polygons/area_b3.homography.json"
    cfg["informal_zones_m"] = {k: {"kind": v[0], "u0u1v0v1": list(v[1])} for k, v in INFORMAL_M.items()}
    json.dump(cfg, open(cfgp, "w"), indent=2)
    json.dump({"method": "R10 painted-bay rectangle + Wigo wheelbase scale", "bay_depth_L_m": round(L, 2),
               "H_photo_px_to_metres_u_v": H.tolist(), "plan_window_m": [U0, U1, V0, V1], "px_per_m": PPM,
               "accuracy": "indicative: ~0.5 m near R8-R10, worse toward L1/R1; anything above ground level (car roofs) is smeared outward"},
              open(os.path.join(ROOT, "configs/polygons/area_b3.homography.json"), "w"), indent=2)

    # ---- top view with overlay -------------------------------------------------------------------
    im = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)).convert("RGBA")
    ov = Image.new("RGBA", im.size); d = ImageDraw.Draw(ov)
    for u in range(int(np.ceil(U0 / 5) * 5), int(U1) + 1, 5):
        x = (u - U0) * PPM; d.line([(x, 0), (x, Hp)], fill=(255, 255, 255, 70)); d.text((x + 3, 3), f"u={u}", fill=(255, 255, 255, 200), font=font(14, False))
    for v in range(int(np.ceil(V0 / 5) * 5), int(V1) + 1, 5):
        y = (v - V0) * PPM; d.line([(0, y), (Wp, y)], fill=(255, 255, 255, 70)); d.text((3, y + 2), f"v={v}", fill=(255, 255, 255, 200), font=font(14, False))
    for sid, poly in plan.items():
        c = STATE_COL[state[sid]]
        pts = [tuple(np.array(m2px(a, b)).ravel()) for a, b in poly]
        d.polygon(pts, fill=c + (45,)); d.line(pts + [pts[0]], fill=c + (255,), width=3)
        cx, cy = np.mean(pts, 0); d.text((cx - 12, cy - 8), sid, fill=(255, 255, 0, 255), font=font(18))
    for k, (kind, (a, b, c0, c1)) in INFORMAL_M.items():
        x0, y0 = m2px(a, c0); x1, y1 = m2px(b, c1)
        d.rectangle([x0, y0, x1, y1], outline=(80, 150, 255, 255), width=3); d.text((x0 + 4, y0 + 4), k, fill=(150, 210, 255, 255), font=font(16))
    d.rectangle([20, Hp - 70, 20 + 5 * PPM, Hp - 58], fill=(255, 255, 255, 255)); d.text((20, Hp - 54), "5 m", fill=(255, 255, 255, 255), font=font(18))
    Image.alpha_composite(im, ov).convert("RGB").save(os.path.join(ROOT, "outputs/lot_b3/lot_b3_topview.png"))

    # ---- schematic slot plan ---------------------------------------------------------------------
    K = 30                                                    # px per metre
    pad_x, pad_y = 50, 110
    W2, H2 = int((U1 - U0) * K) + 2 * pad_x, int((V1 - V0) * K) + pad_y + 190
    pl = Image.new("RGB", (W2, H2), "white"); g = ImageDraw.Draw(pl)
    T = lambda u, v: (pad_x + (u - U0) * K, pad_y + (v - V0) * K)
    g.text((20, 18), "SLOT PLAN, lot B3 (provisional, to approximate scale)", fill="black", font=font(22))
    g.text((20, 52), "bays 2.4 m wide | top of plan = top of photo | L9 level with R10 | dot = state in photo", fill=(90, 90, 90), font=font(13, False))
    for sid, poly in plan.items():
        zone = {"L": "left", "R": "right", "C": "center"}[sid[0]]
        col = ZONE_COL[zone]
        pts = [T(a, b) for a, b in poly]
        g.polygon(pts, fill=col, outline=(40, 40, 40)); g.line(pts + [pts[0]], fill=(40, 40, 40), width=2)
        cx, cy = np.mean(pts, 0); g.text((cx - 9 * len(sid) / 2 - 12, cy - 9), sid, fill="black", font=font(15))
        rx = max(p_[0] for p_ in pts) - 16
        g.ellipse([rx - 6, cy - 6, rx + 6, cy + 6], fill=STATE_COL[state[sid]], outline="black")
    g.text(T(0.2, 0.1), "PWD", fill=(40, 40, 40), font=font(12))
    for k, (kind, (a, b, c0, c1)) in INFORMAL_M.items():
        col = (198, 228, 198)
        p0, p1 = T(a, c0), T(b, c1)
        g.rectangle([p0, p1], fill=col, outline=(40, 40, 40), width=2); g.text((p0[0] + 6, p0[1] + 6), f"{k}\n{kind}", fill="black", font=font(13))
    ly = H2 - 170
    g.rectangle([20, ly, 480, H2 - 20], outline=(40, 40, 40), width=2)
    n = lambda z: sum(1 for k in plan if k[0] == z)
    occ = sum(1 for v in state.values() if v == "occupied"); av = sum(1 for v in state.values() if v == "available"); un = sum(1 for v in state.values() if v == "unknown")
    for i, (t, col) in enumerate([(f"Left car bays: {n('L')}", ZONE_COL['left']), (f"Right car bays: {n('R')}  (R10 = accessible bay)", ZONE_COL['right']),
                                  (f"Centre painted bay: {n('C')}", ZONE_COL['center']), ("Informal / aisle (not bays): 3", (198, 228, 198))]):
        g.ellipse([34, ly + 14 + i * 26, 50, ly + 30 + i * 26], fill=col, outline="black"); g.text((60, ly + 13 + i * 26), t, fill="black", font=font(15))
    g.text((34, ly + 122), f"Bays: {len(plan)}   occupied {occ} / available {av}", fill="black", font=font(15))
    pl.save(os.path.join(ROOT, "outputs/lot_b3/lot_b3_slot_plan.png"))
    print("bay depth L =", round(L, 2), "m; wrote topview, slot plan, homography")


if __name__ == "__main__":
    main()
