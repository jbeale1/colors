#!/usr/bin/env python3
"""
Convert CIE Lab values to Munsell notation (e.g. "5B 7.2/4.1") using the
colour-science library, which implements the real ASTM D1535 renotation
algorithm -- not an approximation.

Install first:
    pip install colour-science

Usage:
    python3 lab_to_munsell.py
"""

import numpy as np
import colour

# D65 reference white (matches the Lab values color_check.py produces)
ILLUMINANT_XY = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]


def lab_to_munsell(L, a, b):
    lab = np.array([L, a, b])
    xyz = colour.Lab_to_XYZ(lab, illuminant=ILLUMINANT_XY)
    xyY = colour.XYZ_to_xyY(xyz)
    return colour.notation.munsell.xyY_to_munsell_colour(xyY)


tiles = {
    "white": (90.3, -1.4, 1.5),
    "pink":  (80.0,  8.6, 8.8),
    "blue":  (73.9, -8.1, -12.0),
    "green": (74.2, -11.5, 7.9),
}

for name, (L, a, b) in tiles.items():
    try:
        munsell = lab_to_munsell(L, a, b)
        print(f"{name:8s} Lab=({L:.1f},{a:.1f},{b:.1f})  ->  Munsell {munsell}")
    except Exception as e:
        print(f"{name:8s} conversion failed: {e}")
