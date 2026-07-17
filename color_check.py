#!/usr/bin/env python3
"""
Color-correct tile samples against a GretagMacbeth ColorChecker in the same photo.

Workflow:
  1. Click the 4 outer corners of the ColorChecker grid (in order: top-left patch
     center's outer corner, top-right, bottom-right, bottom-left) -- i.e. corners
     of the 6x4 patch array, not the whole card border.
  2. Click a small rectangle (two opposite corners) on a flat, non-specular area
     for each tile group you want measured. You'll be prompted for a label each time.
  3. Script builds a 3x3 linear correction matrix (least squares, sRGB-linear space)
     mapping photographed chart patches -> reference chart patches, applies it to
     the tile samples, and prints corrected sRGB + CIE Lab for each tile group.

Usage:
    python3 color_check.py photo.jpg
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

RAW_EXTS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"}

VERSION = "5.5"

# Reference sRGB values (0-255) for the 24 GretagMacbeth ColorChecker patches,
# row-major, top-left to bottom-right (standard published values).
REF_SRGB = np.array([
    [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67],
    [133, 128, 177], [103, 189, 170], [214, 126, 44], [80, 91, 166],
    [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
    [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31],
    [187, 86, 149], [8, 133, 161], [243, 243, 242], [200, 200, 200],
    [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52],
], dtype=np.float64)

PATCH_NAMES = [
    "dark skin", "light skin", "blue sky", "foliage",
    "blue flower", "bluish green", "orange", "purplish blue",
    "moderate red", "purple", "yellow green", "orange yellow",
    "blue", "green", "red", "yellow",
    "magenta", "cyan", "white 9.5", "neutral 8",
    "neutral 6.5", "neutral 5", "neutral 3.5", "black 2",
]

# sRGB (D65) linear -> XYZ, same matrix used inline in srgb_to_lab().
SRGB_TO_XYZ_D65 = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])

# Bradford chromatic adaptation, D65 -> D50 (the PCS white point DNG's
# ColorMatrix1 is defined against). Values per Lindbloom / W3C CSS Color 4.
BRADFORD_D65_TO_D50 = np.array([
    [1.0479298208405488, 0.022946793341019088, -0.05019222954313557],
    [0.029627815688159344, 0.990434484573249, -0.01707382502938514],
    [-0.009243058152591178, 0.015055144896577895, 0.7518742899580008],
])


def srgb_to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    out = np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1 / 2.4)) - 0.055)
    return np.clip(out * 255.0, 0, 255)


def srgb_to_lab(rgb255):
    """rgb255: Nx3 array, 0-255 sRGB, any float precision. Returns Nx3 Lab.
    Computed directly via CIE formulas in float64 -- no 8-bit quantization
    at any intermediate step (unlike routing through cv2.cvtColor on uint8)."""
    rgb255 = np.atleast_2d(np.asarray(rgb255, dtype=np.float64))
    lin = srgb_to_linear(rgb255)  # 0-1 linear-light R,G,B

    # sRGB (D65) linear -> XYZ
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = lin @ M.T

    # D65 reference white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x, y, z = xyz[:, 0] / Xn, xyz[:, 1] / Yn, xyz[:, 2] / Zn

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def f(t):
        return np.where(t > eps, np.cbrt(t), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=1)


def save_corner_debug_images(img, raw_pts, refined_pts, tag, labels=None, crop=70):
    """Save a zoomed PNG per point showing the seed (red) vs the sub-pixel
    refined position (green), for visual sanity-checking."""
    if labels is None:
        labels = ["TL", "TR", "BR", "BL"][:len(raw_pts)]
    bgr_img = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)  # 8-bit OK here,
    # this is just a visual debug PNG -- full-precision img is untouched for actual sampling
    for (rx, ry), (fx, fy), lbl in zip(raw_pts, refined_pts, labels):
        cx, cy = int(round(fx)), int(round(fy))
        x0, x1 = max(cx - crop, 0), min(cx + crop, img.shape[1])
        y0, y1 = max(cy - crop, 0), min(cy + crop, img.shape[0])
        patch = bgr_img[y0:y1, x0:x1].copy()
        scale = 8
        patch_big = cv2.resize(patch, (patch.shape[1] * scale, patch.shape[0] * scale),
                                interpolation=cv2.INTER_NEAREST)
        # map original coords into the upscaled crop
        rpx, rpy = int((rx - x0) * scale), int((ry - y0) * scale)
        fpx, fpy = int((fx - x0) * scale), int((fy - y0) * scale)
        cv2.drawMarker(patch_big, (rpx, rpy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)   # red = seed
        cv2.drawMarker(patch_big, (fpx, fpy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)   # green = refined
        cv2.rectangle(patch_big, (0, 0), (patch_big.shape[1] - 1, patch_big.shape[0] - 1), (255, 255, 0), 1)
        outname = f"corner_debug_{tag}_{lbl}.png"
        cv2.imwrite(outname, patch_big)
        print(f"  wrote {outname}  (red=seed, green=refined; seed={rx:.1f},{ry:.1f} "
              f"-> refined={fx:.1f},{fy:.1f}, moved {np.hypot(fx-rx, fy-ry):.2f}px)")


def coarse_locate_corner(gray, seed_xy, search_radius):
    """Wide-radius search for the strongest corner-like feature near a rough
    click, using Harris response weighted toward the seed (so it doesn't jump
    to some other, stronger, but wrong corner elsewhere in the search window).
    Handles the case where the initial click is tens of pixels off, which is
    common when clicking on a large image displayed at reduced screen size."""
    h, w = gray.shape
    x0, y0 = int(round(seed_xy[0])), int(round(seed_xy[1]))
    xs, xe = max(x0 - search_radius, 0), min(x0 + search_radius, w)
    ys, ye = max(y0 - search_radius, 0), min(y0 + search_radius, h)
    crop = gray[ys:ye, xs:xe].astype(np.float32)

    harris = cv2.cornerHarris(crop, blockSize=9, ksize=5, k=0.04)
    yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
    dist2 = (xx - (x0 - xs)) ** 2 + (yy - (y0 - ys)) ** 2
    proximity_weight = np.exp(-dist2 / (2 * (search_radius * 0.6) ** 2))
    score = harris * proximity_weight

    idx = np.unravel_index(np.argmax(score), score.shape)
    return (float(xs + idx[1]), float(ys + idx[0]))


def locate_chart(img, raw_corners, cols=6, rows=4, tag="chart", debug=False):
    """Locate the chart precisely using the 4 INTERIOR patch-grid-line
    intersections one step in from each outer corner, instead of the true
    outer corners. Rationale: the outer corner sits between an often-dark
    patch and a similarly dark card background/border -- an ambiguous,
    low-contrast boundary. One patch-step inward, every intersection is a
    genuine 4-way junction (colored patch / black grid line / colored patch),
    the same strong, unambiguous corner type checkerboard calibration relies
    on. This works even when 'dark skin' or 'black 2' sits right at a corner.

    raw_corners: rough clicks for the 4 OUTER corners (TL, TR, BR, BL), used
    only to seed the search directions -- never used directly as the anchor.

    Returns: 3x3 perspective transform H mapping normalized chart coordinates
    (u,v) in [0,1]x[0,1] to pixel coordinates.
    """
    gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    tl, tr, br, bl = [np.array(c, dtype=np.float64) for c in raw_corners]
    col_vec = (tr - tl) / cols
    row_vec = (bl - tl) / rows
    spacing = min(np.linalg.norm(col_vec), np.linalg.norm(row_vec))

    # Search radius is scaled to a fraction of one patch's spacing, so the
    # search cannot wander into a neighboring patch or off the chart entirely.
    search_radius = max(int(round(spacing * 0.4)), 15)
    # Fine window must stay well within the gap between patches -- too wide
    # and cornerSubPix pulls in gradient info from the *next* patch boundary
    # and locks onto the wrong feature entirely.
    fine_win = min(max(int(round(spacing * 0.05)), 7), 15)

    # One patch-step inward from each outer corner, in (name, seed) pairs.
    seeds = {
        "TL": tl + col_vec + row_vec,
        "TR": tr - col_vec + row_vec,
        "BR": br - col_vec - row_vec,
        "BL": bl + col_vec - row_vec,
    }

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = {}
    seed_list, refined_list, labels = [], [], []
    for name, seed in seeds.items():
        coarse = coarse_locate_corner(gray, seed, search_radius)
        pts = np.array([coarse], dtype=np.float32).reshape(-1, 1, 2)
        r = cv2.cornerSubPix(gray, pts, (fine_win, fine_win), (-1, -1), criteria)
        refined[name] = (float(r[0][0][0]), float(r[0][0][1]))
        seed_list.append(seed)
        refined_list.append(refined[name])
        labels.append(f"{name}int")

    if debug:
        save_corner_debug_images(img, seed_list, refined_list, tag, labels=labels,
                                  crop=max(int(search_radius * 1.5), 70))

    # Fit a perspective transform from the 4 known interior (u,v) grid-line
    # coordinates to their refined pixel locations.
    src_uv = np.array([
        [1.0 / cols, 1.0 / rows],
        [(cols - 1.0) / cols, 1.0 / rows],
        [(cols - 1.0) / cols, (rows - 1.0) / rows],
        [1.0 / cols, (rows - 1.0) / rows],
    ], dtype=np.float32)
    dst_px = np.array([refined["TL"], refined["TR"], refined["BR"], refined["BL"]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_uv, dst_px)
    return H


def patch_centers_from_H(H, cols=6, rows=4):
    """Map each patch's normalized center (u,v) through the chart's
    perspective transform to get its pixel-space center. Row-major order."""
    uv = np.array([[[(c + 0.5) / cols, (r + 0.5) / rows] for c in range(cols)]
                    for r in range(rows)], dtype=np.float32).reshape(1, -1, 2)
    px = cv2.perspectiveTransform(uv, H)[0]
    return [(float(x), float(y)) for x, y in px]


def estimate_half_from_H(H, cols=6, rows=4, frac=0.35):
    """Estimate a sampling half-window from the actual on-screen patch
    spacing implied by the fitted perspective transform."""
    pts_uv = np.array([[[0.5 / cols, 0.5 / rows],
                         [1.5 / cols, 0.5 / rows],
                         [0.5 / cols, 1.5 / rows]]], dtype=np.float32)
    px = cv2.perspectiveTransform(pts_uv, H)[0]
    dx = np.linalg.norm(px[1] - px[0])
    dy = np.linalg.norm(px[2] - px[0])
    return max(int(round(min(dx, dy) * frac)), 3)


def click_points(img, n, prompt):
    plt.figure(figsize=(10, 8))
    disp = np.clip(img, 0, 255).astype(np.uint8)  # display only -- full-precision img is untouched
    plt.imshow(disp)
    plt.axis("on")

    plt.title(prompt + "\nZoom/pan with the toolbar as needed, then press any KEY (not click) to begin")
    print(f"{prompt}\nUse the toolbar to zoom/pan as needed. When ready, press any KEYBOARD "
          f"key with the plot window focused (a mouse click here just zooms/pans, it won't count).")
    while not plt.waitforbuttonpress():
        pass   # mouse clicks (zoom/pan drags) return False -- keep waiting for a real keypress

    plt.title(prompt)
    pts = plt.ginput(n, timeout=0)
    plt.close()
    return [(int(round(x)), int(round(y))) for x, y in pts]


def sample_patch(img, cx, cy, half=8):
    cx, cy = int(round(cx)), int(round(cy))
    h, w = img.shape[:2]
    x0, x1 = max(cx - half, 0), min(cx + half, w)
    y0, y1 = max(cy - half, 0), min(cy + half, h)
    region = img[y0:y1, x0:x1].reshape(-1, 3)
    return np.median(region, axis=0)


def sample_rect(img, p0, p1):
    x0, x1 = sorted([p0[0], p1[0]])
    y0, y1 = sorted([p0[1], p1[1]])
    region = img[y0:y1, x0:x1].reshape(-1, 3)
    return np.median(region, axis=0)


def patch_exposure_stats(img, cx, cy, half):
    """Per-channel min/max/mean over a patch's sample window, plus the
    fraction of pixels sitting at or near full-scale (potential clipping).
    Values are in the same 0-255 sRGB-encoded scale as `img`."""
    cx, cy = int(round(cx)), int(round(cy))
    h, w = img.shape[:2]
    x0, x1 = max(cx - half, 0), min(cx + half, w)
    y0, y1 = max(cy - half, 0), min(cy + half, h)
    region = img[y0:y1, x0:x1].reshape(-1, 3)
    clip_frac = (region >= 254.5).mean(axis=0)
    return region.min(axis=0), region.max(axis=0), region.mean(axis=0), clip_frac


def print_exposure_report(img, centers, patch_half, patch_names, measured, meas_lin, ref_lin, label=""):
    """Report absolute exposure level per patch from the raw (uncorrected)
    chart sample, so the shot can be confirmed well-exposed before trusting
    the color fit: highlights not clipped, shadows not lifted by stray
    light/flare/specular reflection."""
    tag = f" ({label})" if label else ""
    print(f"\nExposure check{tag} -- raw sampled sRGB 0-255, before correction:")
    print(f"{'patch':<16}{'R':>8}{'G':>8}{'B':>8}{'clip%':>8}")
    any_clip = False
    for i, (cx, cy) in enumerate(centers):
        rmin, rmax, rmean, clip_frac = patch_exposure_stats(img, cx, cy, patch_half)
        clip_pct = clip_frac.max() * 100
        flag = ""
        if clip_pct > 0:
            flag = "  <-- possible clipping"
            any_clip = True
        print(f"{patch_names[i]:<16}{rmean[0]:8.1f}{rmean[1]:8.1f}{rmean[2]:8.1f}{clip_pct:8.1f}{flag}")
    if not any_clip:
        print("No patches show pixels near full-scale (255) -- no clipping detected.")

    w9, n5, b2 = patch_names.index("white 9.5"), patch_names.index("neutral 5"), patch_names.index("black 2")
    print(f"\n  white 9.5 : raw sRGB = {measured[w9].round(1).tolist()}"
          f"{'  <-- CLIPPED' if measured[w9].max() >= 254.5 else ''}")
    print(f"  neutral 5 : raw sRGB = {measured[n5].round(1).tolist()}")
    print(f"  black 2   : raw sRGB = {measured[b2].round(1).tolist()}")

    # Stray-light / flare check: compare the measured black:white linear
    # reflectance ratio against the reference chart's own true ratio. If the
    # camera sees relatively more light in the black patch than the physical
    # chart reflects, something (flare, veiling glare, a specular
    # reflection) is adding light into the shadows and reducing contrast.
    ref_ratio = ref_lin[b2].mean() / ref_lin[w9].mean()
    meas_ratio = meas_lin[b2].mean() / meas_lin[w9].mean()
    excess = meas_ratio / ref_ratio
    print(f"\n  black 2 / white 9.5 linear ratio: measured={meas_ratio:.4f} "
          f"vs reference={ref_ratio:.4f} ({excess:.2f}x reference)")
    if excess > 1.5:
        print("  <-- black 2 is brighter than the chart's own contrast would predict --")
        print("      suggests stray light, flare, or a specular reflection is lifting the shadows.")


def ciede2000(lab1, lab2):
    """
    CIEDE2000 color difference, vectorized over Nx3 Lab arrays.
    Follows Sharma, Wu, Dalal (2005), "The CIEDE2000 Color-Difference
    Formula: Implementation Notes, Supplementary Test Data, and
    Mathematical Observations" -- the standard reference implementation,
    including the clarifications that paper made to the original CIE
    formula's ambiguous edge cases (hue-difference wraparound, and using
    the *unprimed* mean chroma for the G factor).
    """
    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    Cbar = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7)))

    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p ** 2 + b1 ** 2)
    C2p = np.sqrt(a2p ** 2 + b2 ** 2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where((C1p * C2p) == 0, 0.0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    Lbarp = (L1 + L2) / 2.0
    Cbarp = (C1p + C2p) / 2.0

    hsum = h1p + h2p
    habs = np.abs(h1p - h2p)
    Hbarp = np.where(
        (C1p * C2p) == 0, hsum,
        np.where(habs > 180,
                 np.where(hsum < 360, (hsum + 360) / 2.0, (hsum - 360) / 2.0),
                 hsum / 2.0)
    )

    T = (1 - 0.17 * np.cos(np.radians(Hbarp - 30))
           + 0.24 * np.cos(np.radians(2 * Hbarp))
           + 0.32 * np.cos(np.radians(3 * Hbarp + 6))
           - 0.20 * np.cos(np.radians(4 * Hbarp - 63)))

    d_theta = 30 * np.exp(-(((Hbarp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbarp ** 7 / (Cbarp ** 7 + 25.0 ** 7))
    Sl = 1 + (0.015 * (Lbarp - 50) ** 2) / np.sqrt(20 + (Lbarp - 50) ** 2)
    Sc = 1 + 0.045 * Cbarp
    Sh = 1 + 0.015 * Cbarp * T
    Rt = -np.sin(np.radians(2 * d_theta)) * Rc

    kL = kC = kH = 1.0
    term_L = dLp / (kL * Sl)
    term_C = dCp / (kC * Sc)
    term_H = dHp / (kH * Sh)
    return np.sqrt(term_L ** 2 + term_C ** 2 + term_H ** 2 + Rt * term_C * term_H)


def run_two_chart_mode(img, debug_corners=False):
    print("TWO-CHART COMPARISON MODE")
    print("Click the 4 outer corners of CHART A's 6x4 patch grid: TL, TR, BR, BL.")
    corners_a_raw = click_points(img, 4, "Chart A: click 4 outer corners (TL, TR, BR, BL)")
    H_a = locate_chart(img, corners_a_raw, tag="A", debug=debug_corners)
    centers_a = patch_centers_from_H(H_a)
    half_a = estimate_half_from_H(H_a)
    print(f"  Chart A sample window: {2*half_a+1}x{2*half_a+1} px per patch")
    measured_a = np.array([sample_patch(img, cx, cy, half=half_a) for cx, cy in centers_a])

    print("Click the 4 outer corners of CHART B's 6x4 patch grid: TL, TR, BR, BL.")
    corners_b_raw = click_points(img, 4, "Chart B: click 4 outer corners (TL, TR, BR, BL)")
    H_b = locate_chart(img, corners_b_raw, tag="B", debug=debug_corners)
    centers_b = patch_centers_from_H(H_b)
    half_b = estimate_half_from_H(H_b)
    print(f"  Chart B sample window: {2*half_b+1}x{2*half_b+1} px per patch")
    measured_b = np.array([sample_patch(img, cx, cy, half=half_b) for cx, cy in centers_b])

    patch_names = [
        "dark skin", "light skin", "blue sky", "foliage",
        "blue flower", "bluish green", "orange", "purplish blue",
        "moderate red", "purple", "yellow green", "orange yellow",
        "blue", "green", "red", "yellow",
        "magenta", "cyan", "white 9.5", "neutral 8",
        "neutral 6.5", "neutral 5", "neutral 3.5", "black 2",
    ]

    ref_lin = srgb_to_linear(REF_SRGB)
    print_exposure_report(img, centers_a, half_a, patch_names, measured_a,
                           srgb_to_linear(measured_a), ref_lin, label="Chart A")
    print_exposure_report(img, centers_b, half_b, patch_names, measured_b,
                           srgb_to_linear(measured_b), ref_lin, label="Chart B")

    lab_a = srgb_to_lab(measured_a)
    lab_b = srgb_to_lab(measured_b)
    d_srgb = measured_b - measured_a
    d_lab = lab_b - lab_a
    delta_e76 = np.sqrt(np.sum(d_lab ** 2, axis=1))  # CIE76 dE
    delta_e00 = ciede2000(lab_a, lab_b)              # CIEDE2000 dE

    print("\n=== Chart A vs Chart B: per-patch comparison ===")
    hdr = (f"{'patch':<16}{'dR':>7}{'dG':>7}{'dB':>7}{'dL*':>7}{'da*':>7}{'db*':>7}"
           f"{'dE76':>8}{'dE00':>8}")
    print(hdr)
    order = np.argsort(-delta_e00)
    for i in order:
        dr, dg, db = d_srgb[i]
        dl, da, db_ = d_lab[i]
        # flag = "  <-- high" if delta_e76[i] > 3.0 else "" # commented out because the threshold is subjective and may not be appropriate for all use cases
        flag = ""
        print(f"{patch_names[i]:<16}{dr:7.1f}{dg:7.1f}{db:7.1f}{dl:7.1f}{da:7.1f}{db_:7.1f}"
              f"{delta_e76[i]:8.2f}{delta_e00[i]:8.2f}{flag}")

    print(f"\nMean dE76: {delta_e76.mean():.2f}   Max dE76: {delta_e76.max():.2f} "
          f"({patch_names[np.argmax(delta_e76)]})")
    print(f"Mean dE00: {delta_e00.mean():.2f}   Max dE00: {delta_e00.max():.2f} "
          f"({patch_names[np.argmax(delta_e00)]})")
    print("(dE < ~1 is imperceptible, ~1-3 perceptible on close inspection, "
          ">3 clearly visible mismatch. dE00 corrects CIE76's known "
          "over-weighting of blues/near-neutrals and is the more reliable "
          "figure where the two differ.)")


def ref_xyz_d50():
    """Reference XYZ (D50, Y=1 white-scale) for the 24 ColorChecker patches,
    derived from the same REF_SRGB (D65) values used for tile measurement,
    via sRGB->XYZ(D65) then Bradford-adapted to D50 -- the white point DNG's
    ColorMatrix1 is defined against."""
    ref_lin = srgb_to_linear(REF_SRGB)
    xyz_d65 = ref_lin @ SRGB_TO_XYZ_D65.T
    return xyz_d65 @ BRADFORD_D65_TO_D50.T


def calibrate_color_matrix(path, debug_corners=False):
    """
    Derive a DNG ColorMatrix1 (XYZ D50 -> camera-native RGB) from a
    ColorChecker chart shot, for embedding in _save_raw_dng() in
    vehicle_detect.py.

    Reuses the same chart-detection/sampling pipeline (locate_chart,
    patch_centers_from_H, sample_patch) as the normal tile-measurement mode,
    but samples the camera's raw, linear, white-balanced RGB -- no color
    matrix, no gamma -- instead of the finished sRGB rendering, since that's
    what a DNG's ColorMatrix1 has to map from. A second, normally-rendered
    copy of the same frame is used only so the chart is visible to click on.
    """
    import rawpy
    with rawpy.imread(path) as raw:
        rgb_display = raw.postprocess(
            use_camera_wb=True, no_auto_bright=True, output_bps=16,
            gamma=(2.4, 12.92), output_color=rawpy.ColorSpace.sRGB,
        )
        rgb_linear = raw.postprocess(
            use_camera_wb=True, no_auto_bright=True, output_bps=16,
            gamma=(1, 1), output_color=rawpy.ColorSpace.raw,
        )
    disp = rgb_display.astype(np.float64) / 65535.0 * 255.0
    lin  = rgb_linear.astype(np.float64) / 65535.0   # 0-1 linear camera RGB

    print("Click the 4 OUTER CORNERS of the 6x4 ColorChecker patch grid,")
    print("in order: top-left, top-right, bottom-right, bottom-left.")
    corners_raw = click_points(disp, 4, "Click 4 outer corners of patch grid: TL, TR, BR, BL")
    H = locate_chart(disp, corners_raw, tag="calib", debug=debug_corners)
    centers = patch_centers_from_H(H)
    patch_half = estimate_half_from_H(H)
    print(f"Chart sample window: {2*patch_half+1}x{2*patch_half+1} px per patch "
          f"({(2*patch_half+1)**2} pixels averaged)")

    cam_lin = np.array([sample_patch(lin, cx, cy, half=patch_half) for cx, cy in centers])
    xyz_d50 = ref_xyz_d50()

    # camera_rgb (row) ~= XYZ_d50 (row) @ ColorMatrix1.T -- solve for that,
    # then transpose to get ColorMatrix1 itself (camera_col = M @ xyz_col).
    X, *_ = np.linalg.lstsq(xyz_d50, cam_lin, rcond=None)
    color_matrix1 = X.T

    pred = xyz_d50 @ X
    err = pred - cam_lin
    per_patch_rmse = np.sqrt(np.mean(err ** 2, axis=1))
    rmse = np.sqrt(np.mean(err ** 2))

    print(f"\nColorMatrix1 fit RMSE over 24 patches (linear camera RGB, 0-1 scale): {rmse:.4f}")
    print("Per-patch residuals (predicted - measured, linear RGB units):")
    print(f"{'patch':<16}{'err_R':>9}{'err_G':>9}{'err_B':>9}{'RMSE':>9}")
    order = np.argsort(-per_patch_rmse)
    for i in order:
        dr, dg, db = err[i]
        flag = "  <-- high" if per_patch_rmse[i] > 1.5 * rmse else ""
        print(f"{PATCH_NAMES[i]:<16}{dr:9.4f}{dg:9.4f}{db:9.4f}{per_patch_rmse[i]:9.4f}{flag}")

    print("\nColorMatrix1 (XYZ D50 -> camera native RGB):")
    for row in color_matrix1:
        print("  " + "  ".join(f"{v:+.6f}" for v in row))

    denom = 10000
    flat_rational = [[int(round(v * denom)), denom] for v in color_matrix1.flatten()]
    print("\nPaste into _save_raw_dng() in vehicle_detect.py, right after the")
    print("existing t.set(DNGTag.AsShotNeutral, ...) block:")
    print("    t.set(DNGTag.CalibrationIlluminant1, CalibrationIlluminant.D65)")
    print(f"    t.set(DNGTag.ColorMatrix1, {flat_rational})")
    print("(needs: from pidng.defs import CalibrationIlluminant)")
    print("\nCalibrationIlluminant1 assumed D65 (outdoor daylight). This is a")
    print("single-illuminant matrix -- accuracy will degrade under noticeably")
    print("different lighting (dawn/dusk, artificial light at night).")


def main():
    print(f"color_correct.py version {VERSION}")

    two_chart = "--two-chart" in sys.argv
    if two_chart:
        sys.argv.remove("--two-chart")

    calibrate_matrix = "--calibrate-matrix" in sys.argv
    if calibrate_matrix:
        sys.argv.remove("--calibrate-matrix")

    debug_corners = "--debug-corners" in sys.argv
    if debug_corners:
        sys.argv.remove("--debug-corners")

    if len(sys.argv) < 2:
        print("Usage: python3 color_correct.py photo.jpg [--exclude patch1,patch2,...]")
        print("       python3 color_correct.py photo.jpg --two-chart")
        print("       python3 color_correct.py photo.dng --calibrate-matrix")
        print("       add --debug-corners to save zoomed PNGs of each corner refinement")
        print("Patch names: dark skin, light skin, blue sky, foliage, blue flower,")
        print("  bluish green, orange, purplish blue, moderate red, purple,")
        print("  yellow green, orange yellow, blue, green, red, yellow, magenta,")
        print("  cyan, white 9.5, neutral 8, neutral 6.5, neutral 5, neutral 3.5, black 2")
        sys.exit(1)

    exclude_names = set()
    if "--exclude" in sys.argv:
        idx = sys.argv.index("--exclude")
        exclude_arg = sys.argv[idx + 1]
        exclude_names = {n.strip().lower() for n in exclude_arg.split(",")}
        del sys.argv[idx:idx + 2]

    path = sys.argv[1]
    ext = os.path.splitext(path)[1].lower()

    if calibrate_matrix:
        if ext not in RAW_EXTS:
            print("--calibrate-matrix requires a raw file (e.g. .dng) -- it needs the")
            print("camera's raw sensor data, not an already color-processed image.")
            sys.exit(1)
        calibrate_color_matrix(path, debug_corners=debug_corners)
        return

    if ext in RAW_EXTS:
        try:
            import rawpy
        except ImportError:
            print("RAW file detected but 'rawpy' is not installed.")
            print("Install it with:  pip install rawpy")
            sys.exit(1)
        with rawpy.imread(path) as raw:
            # camera white balance, no auto brightness stretch, 16-bit linear-ish output,
            # sRGB output color space, no gamma applied by default settings below (gamma
            # (1,1) + no_auto_bright gives us values close to linear-scaled sensor data)
            rgb16 = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=16,
                gamma=(2.4, 12.92),  # standard sRGB-ish gamma so downstream math is consistent
                output_color=rawpy.ColorSpace.sRGB,
            )
        img = rgb16.astype(np.float64) / 65535.0 * 255.0  # keep full precision, no uint8 quantization
    else:
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"Could not read image: {path}")
            sys.exit(1)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)  # source is 8-bit JPEG,
        # but promote immediately so no *further* quantization happens downstream

    if two_chart:
        run_two_chart_mode(img, debug_corners=debug_corners)
        return

    print("Click the 4 OUTER CORNERS of the 6x4 ColorChecker patch grid,")
    print("in order: top-left, top-right, bottom-right, bottom-left.")
    corners_raw = click_points(img, 4, "Click 4 outer corners of patch grid: TL, TR, BR, BL")
    H = locate_chart(img, corners_raw, tag="single", debug=debug_corners)
    centers = patch_centers_from_H(H)
    patch_half = estimate_half_from_H(H)
    print(f"Chart sample window: {2*patch_half+1}x{2*patch_half+1} px per patch "
          f"({(2*patch_half+1)**2} pixels averaged)")

    patch_names = [
        "dark skin", "light skin", "blue sky", "foliage",
        "blue flower", "bluish green", "orange", "purplish blue",
        "moderate red", "purple", "yellow green", "orange yellow",
        "blue", "green", "red", "yellow",
        "magenta", "cyan", "white 9.5", "neutral 8",
        "neutral 6.5", "neutral 5", "neutral 3.5", "black 2",
    ]

    measured = np.array([sample_patch(img, cx, cy, half=patch_half) for cx, cy in centers])
    meas_lin = srgb_to_linear(measured)
    ref_lin = srgb_to_linear(REF_SRGB)

    print_exposure_report(img, centers, patch_half, patch_names, measured, meas_lin, ref_lin)

    fit_mask = np.array([name not in exclude_names for name in patch_names])
    if exclude_names:
        excluded_found = {n for n in patch_names if n in exclude_names}
        missing = exclude_names - excluded_found
        if missing:
            print(f"Warning: these --exclude names didn't match any patch: {missing}")
        print(f"Excluding from fit: {sorted(excluded_found)}\n")

    # Solve linear correction in linear-light RGB using only the non-excluded patches
    A, *_ = np.linalg.lstsq(meas_lin[fit_mask], ref_lin[fit_mask], rcond=None)

    # Report fit quality (computed only over the patches actually used in the fit)
    pred = meas_lin @ A
    pred_srgb = linear_to_srgb(pred)
    err_srgb = pred_srgb - REF_SRGB
    per_patch_err = np.sqrt(np.mean(err_srgb ** 2, axis=1))
    rmse = np.sqrt(np.mean(per_patch_err[fit_mask] ** 2))

    print(f"\nChart fit RMSE over {fit_mask.sum()} patches (sRGB 0-255 units): {rmse:.2f}\n")
    print("Per-patch residuals (predicted - reference, sRGB units):")
    print(f"{'patch':<16}{'err_R':>8}{'err_G':>8}{'err_B':>8}{'RMSE':>8}")
    order = np.argsort(-per_patch_err)
    for i in order:
        dr, dg, db = err_srgb[i]
        tag = ""
        if not fit_mask[i]:
            tag = "  (excluded from fit)"
        elif per_patch_err[i] > 1.5 * rmse:
            tag = "  <-- high"
        print(f"{patch_names[i]:<16}{dr:8.1f}{dg:8.1f}{db:8.1f}{per_patch_err[i]:8.1f}{tag}")
    print()

    tiles = {}
    while True:
        label = input("Label for next tile sample (blank to finish): ").strip()
        if not label:
            break
        print(f"Click two opposite corners of a flat sample rectangle on the '{label}' tile.")
        p0, p1 = click_points(img, 2, f"Click rect for: {label}")
        raw = sample_rect(img, p0, p1)
        raw_lin = srgb_to_linear(raw)
        corr_lin = raw_lin @ A
        corr_srgb = linear_to_srgb(corr_lin)
        lab = srgb_to_lab(corr_srgb)[0]
        tiles[label] = (raw, corr_srgb, lab)

    print("\n=== Results ===")
    for label, (raw, corr, lab) in tiles.items():
        print(f"\n{label}:")
        print(f"  raw sRGB      : {raw.round(1).tolist()}")
        print(f"  corrected sRGB: {corr.round(1).tolist()}")
        print(f"  Lab           : L*={lab[0]:.1f} a*={lab[1]:.1f} b*={lab[2]:.1f}")
        print(f"  hex (corrected): #{int(round(corr[0])):02x}{int(round(corr[1])):02x}{int(round(corr[2])):02x}")


if __name__ == "__main__":
    main()
