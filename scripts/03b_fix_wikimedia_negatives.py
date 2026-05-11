"""
03b_fix_wikimedia_negatives.py  (FINAL)
----------------------------------------
Generates replacement negative images programmatically using PIL.
No Wikimedia downloads — no rate limits, no wrong hashes, no 400/429 errors.

Each replacement is a simple but recognizable line drawing of a product
in the correct USPC class. These serve as open-source negative examples
(visually different from the positive patents but in the same product category).

Usage:
    python scripts/03b_fix_wikimedia_negatives.py
"""

import logging
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

load_dotenv()

ROOT          = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MANIFEST_PATH = ROOT / "data" / "manifest.csv"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SIZE = 1024   # output image size
BG   = 255    # white background
FG   = 0      # black lines

BAD_PATTERNS = [
    "Belt_Differential_Element", "AdhesiveFractures", "Broach_geometry",
    "Broach_tooth_geometry", "2020-03-20_16_51_24-Autodesk_Fusion_360",
    "A_hand-powered,_crank-driven,_milling_machine",
    "APL_53_foot_container", "APL_container",
    "1890s_prone_recumbent_bicycle", "21st_Century_Trilling",
    # partial downloads from earlier attempts
    "D6_wiki_chair_icon", "D6_wiki_chair_technical", "D6_wiki_chair_fontawesome",
    "D6_wiki_chair_bare", "D8_wiki_screwdriver", "D8_wiki_tools_icon",
    "D8_wiki_tools", "D8_wiki_icon_tools", "D14_wiki_phone_icon",
    "D6_wiki_chair_icon_v2", "D6_wiki_chair_drawing", "D6_wiki_chair_fa",
    "D6_wiki_chair_bare2",
]

# ── Drawing functions ─────────────────────────────────────────────────────────

def new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("L", (SIZE, SIZE), BG)
    draw = ImageDraw.Draw(img)
    return img, draw

def lw(w: int) -> int:
    """Scale line width relative to canvas size."""
    return max(1, int(w * SIZE / 512))

def draw_chair(draw: ImageDraw.ImageDraw):
    """Simple side-view chair outline."""
    m = SIZE // 8
    # Seat (horizontal rectangle)
    draw.rectangle([m*2, m*3, m*6, m*4], outline=FG, width=lw(4))
    # Back (vertical rectangle)
    draw.rectangle([m*5, m*1, m*6, m*4], outline=FG, width=lw(4))
    # Front leg
    draw.line([m*2, m*4, m*2, m*7], fill=FG, width=lw(4))
    # Back leg
    draw.line([m*5, m*4, m*5, m*7], fill=FG, width=lw(4))
    # Footrest brace
    draw.line([m*2, m*6, m*5, m*6], fill=FG, width=lw(2))

def draw_sofa(draw: ImageDraw.ImageDraw):
    """Simple front-view sofa."""
    m = SIZE // 8
    # Main body
    draw.rectangle([m, m*4, m*7, m*6], outline=FG, width=lw(4))
    # Left arm
    draw.rectangle([m, m*3, m*2, m*6], outline=FG, width=lw(4))
    # Right arm
    draw.rectangle([m*6, m*3, m*7, m*6], outline=FG, width=lw(4))
    # Back cushion
    draw.rectangle([m*2, m*2, m*6, m*4], outline=FG, width=lw(3))
    # Seat divider
    draw.line([m*4, m*4, m*4, m*6], fill=FG, width=lw(2))
    # Legs
    for x in [m*2, m*6]:
        draw.rectangle([x, m*6, x+m//2, m*7], outline=FG, width=lw(3))

def draw_table(draw: ImageDraw.ImageDraw):
    """Simple side-view table."""
    m = SIZE // 8
    # Tabletop
    draw.rectangle([m, m*3, m*7, m*4], outline=FG, width=lw(4))
    # Legs
    draw.line([m*2, m*4, m*2, m*7], fill=FG, width=lw(4))
    draw.line([m*6, m*4, m*6, m*7], fill=FG, width=lw(4))
    # Stretcher
    draw.line([m*2, m*6, m*6, m*6], fill=FG, width=lw(2))

def draw_bed(draw: ImageDraw.ImageDraw):
    """Simple top-view bed."""
    m = SIZE // 8
    # Frame
    draw.rectangle([m, m*2, m*7, m*7], outline=FG, width=lw(4))
    # Headboard
    draw.rectangle([m, m*2, m*7, m*3], outline=FG, width=lw(4))
    # Pillow
    draw.ellipse([m*2, m*3, m*4, m*4], outline=FG, width=lw(3))
    draw.ellipse([m*4, m*3, m*6, m*4], outline=FG, width=lw(3))
    # Blanket fold
    draw.line([m, m*5, m*7, m*5], fill=FG, width=lw(2))

def draw_mirror(draw: ImageDraw.ImageDraw):
    """Simple oval mirror with frame."""
    cx, cy = SIZE//2, SIZE//2
    rx, ry = SIZE//3, SIZE*2//5
    draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], outline=FG, width=lw(5))
    draw.ellipse([cx-rx+lw(12), cy-ry+lw(12), cx+rx-lw(12), cy+ry-lw(12)],
                 outline=FG, width=lw(2))
    # Stand
    draw.line([cx, cy+ry, cx, cy+ry+lw(40)], fill=FG, width=lw(5))
    draw.line([cx-lw(30), cy+ry+lw(40), cx+lw(30), cy+ry+lw(40)],
              fill=FG, width=lw(5))

def draw_pillow(draw: ImageDraw.ImageDraw):
    """Simple rectangular pillow."""
    m = SIZE // 6
    draw.rounded_rectangle([m, m*2, m*5, m*4], radius=lw(40),
                            outline=FG, width=lw(5))
    # Stitching line
    draw.rounded_rectangle([m+lw(20), m*2+lw(20), m*5-lw(20), m*4-lw(20)],
                            radius=lw(30), outline=FG, width=lw(2))

def draw_hammer(draw: ImageDraw.ImageDraw):
    """Simple hammer side view."""
    m = SIZE // 8
    # Handle
    draw.rectangle([m*2, m*3, m*3, m*7], outline=FG, width=lw(3))
    # Head
    draw.rectangle([m*2, m*2, m*6, m*3+lw(20)], outline=FG, width=lw(4))
    # Claw notch
    draw.line([m*5, m*2, m*5+lw(30), m], fill=FG, width=lw(3))
    draw.line([m*6, m*2, m*6+lw(30), m], fill=FG, width=lw(3))

def draw_wrench(draw: ImageDraw.ImageDraw):
    """Simple open-end wrench."""
    m = SIZE // 8
    # Handle
    draw.rectangle([m, m*4, m*6, m*5], outline=FG, width=lw(3))
    # Head opening (U shape)
    draw.line([m*6, m*3, m*7, m*3], fill=FG, width=lw(4))
    draw.line([m*7, m*3, m*7, m*6], fill=FG, width=lw(4))
    draw.line([m*7, m*6, m*6, m*6], fill=FG, width=lw(4))

def draw_screwdriver(draw: ImageDraw.ImageDraw):
    """Simple screwdriver."""
    m = SIZE // 8
    # Handle (rounded rect)
    draw.rounded_rectangle([m*2, m*2, m*4, m*5], radius=lw(20),
                            outline=FG, width=lw(4))
    # Shaft
    draw.rectangle([m*3, m*5, m*3+lw(10), m*7], outline=FG, width=lw(3))
    # Tip
    draw.polygon([(m*3-lw(8), m*7), (m*3+lw(18), m*7),
                  (m*3+lw(5), m*7+lw(30))], outline=FG, fill=FG)

def draw_saw(draw: ImageDraw.ImageDraw):
    """Simple hand saw."""
    m = SIZE // 8
    # Blade (long rectangle)
    draw.rectangle([m, m*4, m*7, m*5], outline=FG, width=lw(3))
    # Teeth (triangles along bottom)
    for i in range(8):
        x = m + i * (m*6//8)
        draw.polygon([(x, m*5), (x+m*6//16, m*5+lw(30)),
                      (x+m*6//8, m*5)], fill=FG)
    # Handle (top of blade)
    draw.rounded_rectangle([m*5, m*2, m*7, m*4], radius=lw(15),
                            outline=FG, width=lw(3))

def draw_pliers(draw: ImageDraw.ImageDraw):
    """Simple pliers."""
    m = SIZE // 8
    cx = SIZE // 2
    # Upper handle
    draw.line([m, m*2, cx, m*4], fill=FG, width=lw(6))
    # Lower handle
    draw.line([m, m*6, cx, m*4], fill=FG, width=lw(6))
    # Pivot
    draw.ellipse([cx-lw(12), m*4-lw(12), cx+lw(12), m*4+lw(12)],
                 outline=FG, fill=FG)
    # Upper jaw
    draw.line([cx, m*4, m*7, m*3], fill=FG, width=lw(5))
    # Lower jaw
    draw.line([cx, m*4, m*7, m*5], fill=FG, width=lw(5))

def draw_level(draw: ImageDraw.ImageDraw):
    """Spirit level."""
    m = SIZE // 8
    # Body
    draw.rectangle([m, m*3, m*7, m*5], outline=FG, width=lw(4))
    # Bubble vial
    draw.ellipse([m*3, m*3+lw(10), m*5, m*5-lw(10)], outline=FG, width=lw(3))
    # Bubble
    draw.ellipse([m*4-lw(20), m*4-lw(15), m*4+lw(20), m*4+lw(15)],
                 outline=FG, width=lw(2))

def draw_bottle(draw: ImageDraw.ImageDraw):
    """Simple bottle outline."""
    m = SIZE // 8
    cx = SIZE // 2
    # Body
    draw.rounded_rectangle([m*2, m*4, m*6, m*7], radius=lw(30),
                            outline=FG, width=lw(4))
    # Neck (trapezoid)
    draw.polygon([(m*3, m*4), (m*5, m*4), (m*4+lw(20), m*2),
                  (m*4-lw(20), m*2)], outline=FG, width=lw(4))
    # Cap
    draw.rectangle([m*4-lw(25), m*2-lw(20), m*4+lw(25), m*2],
                   outline=FG, width=lw(4))

def draw_box(draw: ImageDraw.ImageDraw):
    """Simple cardboard box perspective view."""
    m = SIZE // 8
    # Front face
    draw.rectangle([m*2, m*4, m*6, m*7], outline=FG, width=lw(4))
    # Top face (parallelogram)
    draw.polygon([(m*2, m*4), (m*4, m*2), (m*7, m*2),
                  (m*6, m*4)], outline=FG, width=lw(4))
    # Right face
    draw.polygon([(m*6, m*4), (m*7, m*2), (m*7, m*6),
                  (m*6, m*7)], outline=FG, width=lw(4))
    # Top flaps
    draw.line([m*3, m*2, m*3+lw(20), m*4], fill=FG, width=lw(2))
    draw.line([m*5, m*2, m*5+lw(20), m*4], fill=FG, width=lw(2))

def draw_tin_can(draw: ImageDraw.ImageDraw):
    """Simple tin can."""
    m = SIZE // 8
    cx = SIZE // 2
    rx = m*2
    # Body
    draw.rectangle([cx-rx, m*2, cx+rx, m*7], outline=FG, width=lw(4))
    # Top ellipse
    draw.ellipse([cx-rx, m*2-lw(20), cx+rx, m*2+lw(20)],
                 outline=FG, width=lw(4))
    # Bottom ellipse
    draw.ellipse([cx-rx, m*7-lw(20), cx+rx, m*7+lw(20)],
                 outline=FG, width=lw(4))
    # Label line
    draw.line([cx-rx, m*4, cx+rx, m*4], fill=FG, width=lw(2))
    draw.line([cx-rx, m*5, cx+rx, m*5], fill=FG, width=lw(2))

def draw_jar(draw: ImageDraw.ImageDraw):
    """Simple mason jar."""
    m = SIZE // 8
    cx = SIZE // 2
    # Body (trapezoid wider at bottom)
    draw.polygon([(cx-m*2, m*3), (cx+m*2, m*3),
                  (cx+m*2+lw(20), m*7), (cx-m*2-lw(20), m*7)],
                 outline=FG, width=lw(4))
    # Neck
    draw.rectangle([cx-m, m*2, cx+m, m*3], outline=FG, width=lw(4))
    # Lid
    draw.rectangle([cx-m-lw(10), m*2-lw(15), cx+m+lw(10), m*2],
                   outline=FG, width=lw(4))
    # Lid band
    draw.line([cx-m*2, m*3+lw(10), cx+m*2, m*3+lw(10)], fill=FG, width=lw(2))

def draw_bag(draw: ImageDraw.ImageDraw):
    """Simple shopping bag."""
    m = SIZE // 8
    cx = SIZE // 2
    # Body
    draw.rectangle([m*2, m*3, m*6, m*7], outline=FG, width=lw(4))
    # Handles
    draw.arc([m*3, m*2, m*4, m*4], 180, 0, fill=FG, width=lw(4))
    draw.arc([m*4, m*2, m*5, m*4], 180, 0, fill=FG, width=lw(4))

def draw_spray_can(draw: ImageDraw.ImageDraw):
    """Simple spray can."""
    m = SIZE // 8
    cx = SIZE // 2
    # Body cylinder
    draw.rectangle([cx-m, m*3, cx+m, m*7], outline=FG, width=lw(4))
    # Shoulder (trapezoid)
    draw.polygon([(cx-m, m*3), (cx+m, m*3),
                  (cx+m//2, m*2), (cx-m//2, m*2)], outline=FG, width=lw(3))
    # Nozzle
    draw.rectangle([cx+m//2, m*2-lw(5), cx+m*2, m*2+lw(10)],
                   outline=FG, width=lw(3))
    # Cap
    draw.ellipse([cx-m//2, m*2-lw(20), cx+m//2, m*2],
                 outline=FG, width=lw(3))

def draw_bicycle(draw: ImageDraw.ImageDraw):
    """Simple bicycle side view."""
    m = SIZE // 8
    r = m * 2
    # Rear wheel
    draw.ellipse([m, m*4-r, m+r*2, m*4+r], outline=FG, width=lw(4))
    # Front wheel
    draw.ellipse([m*5, m*4-r, m*5+r*2, m*4+r], outline=FG, width=lw(4))
    # Frame (triangle)
    cx_r = m + r  # rear wheel center x
    cx_f = m*5 + r  # front wheel center x
    mid_y = m*4
    draw.line([cx_r, mid_y, cx_f, mid_y], fill=FG, width=lw(3))  # chain stay
    draw.line([cx_r, mid_y, (cx_r+cx_f)//2, m*2], fill=FG, width=lw(3))  # seat tube
    draw.line([(cx_r+cx_f)//2, m*2, cx_f, mid_y], fill=FG, width=lw(3))  # fork
    # Seat
    draw.line([(cx_r+cx_f)//2-lw(20), m*2, (cx_r+cx_f)//2+lw(20), m*2],
              fill=FG, width=lw(4))
    # Handlebar
    draw.line([cx_f, m*2, cx_f+lw(15), m*2-lw(20)], fill=FG, width=lw(4))

def draw_car(draw: ImageDraw.ImageDraw):
    """Simple car side view."""
    m = SIZE // 8
    # Body lower (rectangle)
    draw.rounded_rectangle([m, m*5, m*7, m*7], radius=lw(20),
                            outline=FG, width=lw(4))
    # Cabin
    draw.polygon([(m*2, m*5), (m*3, m*3), (m*6, m*3),
                  (m*7, m*5)], outline=FG, width=lw(4))
    # Wheels
    draw.ellipse([m+lw(10), m*6, m*2+lw(10), m*7+lw(20)],
                 outline=FG, fill=BG, width=lw(4))
    draw.ellipse([m*5+lw(10), m*6, m*6+lw(10), m*7+lw(20)],
                 outline=FG, fill=BG, width=lw(4))
    # Window
    draw.polygon([(m*3+lw(5), m*5-lw(5)), (m*3+lw(15), m*3+lw(10)),
                  (m*6-lw(5), m*3+lw(10)), (m*6-lw(5), m*5-lw(5))],
                 outline=FG, width=lw(2))

def draw_motorcycle(draw: ImageDraw.ImageDraw):
    """Simple motorcycle."""
    m = SIZE // 8
    r = m + lw(20)
    # Rear wheel
    draw.ellipse([m, m*4, m+r*2, m*4+r*2], outline=FG, width=lw(4))
    # Front wheel
    draw.ellipse([m*5, m*4, m*5+r*2, m*4+r*2], outline=FG, width=lw(4))
    # Body
    draw.polygon([(m+r, m*5), (m*3, m*3), (m*5, m*3),
                  (m*5+r, m*5)], outline=FG, width=lw(4))
    # Seat
    draw.rectangle([m*3, m*3, m*5, m*3+lw(15)], outline=FG, width=lw(3))
    # Handlebar
    draw.line([m*5, m*3, m*6, m*2], fill=FG, width=lw(4))

def draw_bus(draw: ImageDraw.ImageDraw):
    """Simple bus side view."""
    m = SIZE // 8
    # Body
    draw.rounded_rectangle([m, m*3, m*7, m*6], radius=lw(15),
                            outline=FG, width=lw(4))
    # Windows
    for i in range(4):
        x = m*2 + i*m
        draw.rectangle([x, m*3+lw(10), x+m-lw(10), m*4+lw(10)],
                        outline=FG, width=lw(2))
    # Door
    draw.rectangle([m, m*4, m*2, m*6], outline=FG, width=lw(2))
    # Wheels
    draw.ellipse([m+lw(10), m*6-lw(10), m*2+lw(10), m*7+lw(10)],
                 outline=FG, fill=BG, width=lw(4))
    draw.ellipse([m*5+lw(10), m*6-lw(10), m*6+lw(10), m*7+lw(10)],
                 outline=FG, fill=BG, width=lw(4))

def draw_truck(draw: ImageDraw.ImageDraw):
    """Simple truck side view."""
    m = SIZE // 8
    # Cargo box
    draw.rectangle([m*3, m*3, m*7, m*6], outline=FG, width=lw(4))
    # Cab
    draw.rounded_rectangle([m, m*4, m*3, m*6], radius=lw(10),
                            outline=FG, width=lw(4))
    # Windshield
    draw.rectangle([m+lw(10), m*4+lw(10), m*3-lw(5), m*5],
                   outline=FG, width=lw(2))
    # Wheels
    draw.ellipse([m+lw(5), m*6-lw(5), m*2+lw(5), m*7+lw(10)],
                 outline=FG, fill=BG, width=lw(4))
    draw.ellipse([m*5, m*6-lw(5), m*6, m*7+lw(10)],
                 outline=FG, fill=BG, width=lw(4))

def draw_wheel(draw: ImageDraw.ImageDraw):
    """Spoked wheel."""
    cx, cy = SIZE//2, SIZE//2
    r_outer = SIZE*3//8
    r_inner = SIZE//6
    r_hub   = SIZE//16
    draw.ellipse([cx-r_outer, cy-r_outer, cx+r_outer, cy+r_outer],
                 outline=FG, width=lw(5))
    draw.ellipse([cx-r_hub, cy-r_hub, cx+r_hub, cy+r_hub],
                 outline=FG, fill=FG)
    # Spokes
    for angle in range(0, 360, 45):
        rad = math.radians(angle)
        x1 = cx + int(r_hub * math.cos(rad))
        y1 = cy + int(r_hub * math.sin(rad))
        x2 = cx + int(r_outer * math.cos(rad))
        y2 = cy + int(r_outer * math.sin(rad))
        draw.line([x1, y1, x2, y2], fill=FG, width=lw(3))

def draw_laptop(draw: ImageDraw.ImageDraw):
    """Simple laptop open view."""
    m = SIZE // 8
    # Screen
    draw.rectangle([m*2, m*2, m*6, m*5], outline=FG, width=lw(4))
    # Screen bezel
    draw.rectangle([m*2+lw(15), m*2+lw(15), m*6-lw(15), m*5-lw(15)],
                   outline=FG, width=lw(2))
    # Base (trapezoid, wider)
    draw.polygon([(m, m*5), (m*7, m*5), (m*7+lw(10), m*6),
                  (m-lw(10), m*6)], outline=FG, width=lw(4))
    # Keyboard lines
    for i in range(3):
        y = m*5 + lw(15) + i*lw(12)
        draw.line([m+lw(10), y, m*7-lw(10), y], fill=FG, width=lw(1))

def draw_phone(draw: ImageDraw.ImageDraw):
    """Simple smartphone front view."""
    m = SIZE // 8
    cx = SIZE // 2
    # Body
    draw.rounded_rectangle([m*2, m, m*6, m*7], radius=lw(30),
                            outline=FG, width=lw(5))
    # Screen
    draw.rectangle([m*2+lw(15), m+lw(50), m*6-lw(15), m*7-lw(80)],
                   outline=FG, width=lw(2))
    # Camera
    draw.ellipse([cx-lw(15), m+lw(15), cx+lw(15), m+lw(40)],
                 outline=FG, width=lw(2))
    # Home button
    draw.ellipse([cx-lw(20), m*7-lw(60), cx+lw(20), m*7-lw(20)],
                 outline=FG, width=lw(2))

def draw_desktop_computer(draw: ImageDraw.ImageDraw):
    """Simple desktop tower computer."""
    m = SIZE // 8
    # Tower
    draw.rectangle([m*2, m*2, m*5, m*7], outline=FG, width=lw(4))
    # Drive bays
    draw.rectangle([m*2+lw(10), m*3, m*5-lw(10), m*4], outline=FG, width=lw(2))
    draw.rectangle([m*2+lw(10), m*4+lw(5), m*5-lw(10), m*5+lw(5)],
                   outline=FG, width=lw(2))
    # Power button
    draw.ellipse([m*3, m*2+lw(15), m*3+lw(25), m*2+lw(40)],
                 outline=FG, width=lw(2))

def draw_keyboard(draw: ImageDraw.ImageDraw):
    """Simple keyboard top view."""
    m = SIZE // 8
    # Body
    draw.rounded_rectangle([m, m*3, m*7, m*6], radius=lw(10),
                            outline=FG, width=lw(4))
    # Key rows
    for row in range(4):
        y = m*3 + lw(20) + row*lw(20)
        draw.line([m+lw(10), y, m*7-lw(10), y], fill=FG, width=lw(2))
    # Key columns (simplified)
    for col in range(12):
        x = m + lw(10) + col*lw(45)
        draw.line([x, m*3+lw(10), x, m*6-lw(10)], fill=FG, width=lw(1))

def draw_headphones(draw: ImageDraw.ImageDraw):
    """Simple headphones front view."""
    m = SIZE // 8
    cx = SIZE // 2
    # Headband arc
    draw.arc([m, m*2, m*7, m*5], 180, 0, fill=FG, width=lw(5))
    # Left ear cup
    draw.ellipse([m, m*4, m*3, m*7], outline=FG, width=lw(4))
    # Right ear cup
    draw.ellipse([m*5, m*4, m*7, m*7], outline=FG, width=lw(4))
    # Left stem
    draw.line([m*2, m*3, m*2, m*4], fill=FG, width=lw(4))
    # Right stem
    draw.line([m*6, m*3, m*6, m*4], fill=FG, width=lw(4))

def draw_faucet(draw: ImageDraw.ImageDraw):
    """Simple faucet side view."""
    m = SIZE // 8
    # Spout (horizontal pipe)
    draw.rectangle([m, m*4, m*5, m*5], outline=FG, width=lw(4))
    # Spout tip (angled down)
    draw.polygon([(m*4, m*4), (m*5, m*4), (m*5, m*5),
                  (m*4+lw(20), m*6)], outline=FG, width=lw(3))
    # Valve (vertical pipe)
    draw.rectangle([m*3, m*2, m*4, m*5], outline=FG, width=lw(4))
    # Handle (T-bar)
    draw.line([m*2, m*2, m*5, m*2], fill=FG, width=lw(4))
    draw.rectangle([m*3+lw(5), m*2, m*4-lw(5), m*2+lw(20)],
                   outline=FG, width=lw(3))

def draw_showerhead(draw: ImageDraw.ImageDraw):
    """Simple showerhead."""
    m = SIZE // 8
    cx = SIZE // 2
    # Arm (pipe)
    draw.line([m*2, m*2, m*5, m*4], fill=FG, width=lw(5))
    # Head (circle)
    draw.ellipse([m*4, m*4, m*7, m*7], outline=FG, width=lw(4))
    # Spray holes
    for i in range(3):
        for j in range(3):
            x = m*4 + lw(30) + i*lw(20)
            y = m*4 + lw(30) + j*lw(20)
            draw.ellipse([x, y, x+lw(8), y+lw(8)], outline=FG, fill=FG)

def draw_toilet(draw: ImageDraw.ImageDraw):
    """Simple toilet side view."""
    m = SIZE // 8
    cx = SIZE // 2
    # Tank
    draw.rectangle([m*2, m*2, m*6, m*4], outline=FG, width=lw(4))
    # Bowl (oval)
    draw.ellipse([m*2, m*4, m*6, m*7], outline=FG, width=lw(4))
    # Seat
    draw.arc([m*2+lw(10), m*4+lw(10), m*6-lw(10), m*6],
             180, 0, fill=FG, width=lw(3))

def draw_bathtub(draw: ImageDraw.ImageDraw):
    """Simple bathtub side view."""
    m = SIZE // 8
    # Tub body
    draw.arc([m, m*3, m*7, m*7], 180, 0, fill=FG, width=lw(5))
    draw.line([m, m*5, m*7, m*5], fill=FG, width=lw(5))
    # Rim
    draw.line([m, m*3, m*7, m*3], fill=FG, width=lw(4))
    # Faucet spout
    draw.line([m*5, m*3, m*5, m*4], fill=FG, width=lw(4))
    draw.line([m*5, m*4, m*6, m*4], fill=FG, width=lw(4))
    # Drain
    draw.ellipse([cx-lw(15) if (cx:=SIZE//2) else 0,
                  m*5+lw(10), cx+lw(15), m*5+lw(30)],
                 outline=FG, width=lw(2))

def draw_fan(draw: ImageDraw.ImageDraw):
    """Ceiling fan top view."""
    cx, cy = SIZE//2, SIZE//2
    r = SIZE*3//8
    hub_r = SIZE//12
    # Blades (4)
    for angle in [0, 90, 180, 270]:
        rad = math.radians(angle)
        rad2 = math.radians(angle + 30)
        x1 = cx + int(hub_r * math.cos(rad))
        y1 = cy + int(hub_r * math.sin(rad))
        x2 = cx + int(r * math.cos(rad))
        y2 = cy + int(r * math.sin(rad))
        x3 = cx + int(r * math.cos(rad2))
        y3 = cy + int(r * math.sin(rad2))
        draw.polygon([x1, y1, x2, y2, x3, y3], outline=FG, width=lw(3))
    # Hub
    draw.ellipse([cx-hub_r, cy-hub_r, cx+hub_r, cy+hub_r],
                 outline=FG, fill=FG)

def draw_bucket(draw: ImageDraw.ImageDraw):
    """Simple bucket."""
    m = SIZE // 8
    cx = SIZE // 2
    # Body (trapezoid wider at top)
    draw.polygon([(m*2, m*3), (m*6, m*3),
                  (m*5+lw(10), m*7), (m*3-lw(10), m*7)],
                 outline=FG, width=lw(4))
    # Handle arc
    draw.arc([m*3, m*2, m*5, m*4], 180, 0, fill=FG, width=lw(4))

def draw_syringe(draw: ImageDraw.ImageDraw):
    """Simple syringe."""
    m = SIZE // 8
    # Barrel
    draw.rectangle([m*2, m*3, m*6, m*5], outline=FG, width=lw(4))
    # Needle
    draw.polygon([(m*6, m*3+lw(15)), (m*6, m*5-lw(15)),
                  (m*7+lw(10), m*4)], outline=FG, width=lw(3))
    # Plunger
    draw.rectangle([m, m*3+lw(5), m*2, m*5-lw(5)], outline=FG, width=lw(3))
    draw.rectangle([m-lw(10), m*4-lw(10), m+lw(5), m*4+lw(10)],
                   outline=FG, fill=FG)
    # Graduation marks
    for i in range(4):
        x = m*3 + i*m//2
        draw.line([x, m*3, x, m*3+lw(15)], fill=FG, width=lw(2))

def draw_stethoscope(draw: ImageDraw.ImageDraw):
    """Simple stethoscope."""
    m = SIZE // 8
    cx = SIZE // 2
    # Earpieces (top)
    draw.line([m*2, m*2, m*3, m*3], fill=FG, width=lw(4))
    draw.line([m*6, m*2, m*5, m*3], fill=FG, width=lw(4))
    # Ear tips
    draw.ellipse([m*2-lw(10), m*2-lw(10), m*2+lw(10), m*2+lw(10)],
                 fill=FG)
    draw.ellipse([m*6-lw(10), m*2-lw(10), m*6+lw(10), m*2+lw(10)],
                 fill=FG)
    # Tubing
    draw.arc([m*2, m*2, m*6, m*5], 0, 180, fill=FG, width=lw(4))
    # Tube to chest piece
    draw.line([cx, m*5, cx, m*6], fill=FG, width=lw(4))
    # Chest piece (circle)
    draw.ellipse([cx-m, m*6, cx+m, m*7+m//2], outline=FG, width=lw(4))

def draw_thermometer(draw: ImageDraw.ImageDraw):
    """Simple thermometer."""
    m = SIZE // 8
    cx = SIZE // 2
    # Tube
    draw.rectangle([cx-lw(15), m*2, cx+lw(15), m*6], outline=FG, width=lw(4))
    # Bulb
    draw.ellipse([cx-m//2, m*6, cx+m//2, m*7+m//2], outline=FG, width=lw(4))
    # Mercury level
    draw.rectangle([cx-lw(6), m*4, cx+lw(6), m*6+lw(5)],
                   outline=FG, fill=FG)
    # Scale marks
    for i in range(4):
        y = m*2+lw(20) + i*lw(25)
        draw.line([cx+lw(15), y, cx+lw(30), y], fill=FG, width=lw(2))

def draw_pill(draw: ImageDraw.ImageDraw):
    """Simple capsule/pill."""
    m = SIZE // 8
    cx = SIZE // 2
    r = m
    # Left half (rounded)
    draw.pieslice([m*2, m*4-r, cx, m*4+r], 90, 270, outline=FG,
                  fill=FG)
    # Right half
    draw.pieslice([cx, m*4-r, m*6, m*4+r], 270, 90, outline=FG,
                  fill=BG, width=lw(4))
    draw.rectangle([cx, m*4-r, m*6, m*4+r], outline=None, fill=BG)
    # Outline
    draw.rounded_rectangle([m*2, m*4-r, m*6, m*4+r], radius=r,
                            outline=FG, width=lw(4))
    # Dividing line
    draw.line([cx, m*4-r, cx, m*4+r], fill=FG, width=lw(3))

def draw_bandage(draw: ImageDraw.ImageDraw):
    """Simple bandage/plaster."""
    m = SIZE // 8
    cx, cy = SIZE//2, SIZE//2
    # Main strip (rounded rectangle)
    draw.rounded_rectangle([m, cy-m, m*7, cy+m], radius=m,
                            outline=FG, width=lw(4))
    # Center pad
    draw.rectangle([m*3, cy-m//2, m*5, cy+m//2], outline=FG, width=lw(3))
    # Cross stitching on ends
    for x in [m*2, m*6]:
        draw.line([x-lw(10), cy-m+lw(10), x+lw(10), cy+m-lw(10)],
                  fill=FG, width=lw(2))
        draw.line([x+lw(10), cy-m+lw(10), x-lw(10), cy+m-lw(10)],
                  fill=FG, width=lw(2))

def draw_microscope(draw: ImageDraw.ImageDraw):
    """Simple microscope."""
    m = SIZE // 8
    cx = SIZE // 2
    # Base
    draw.rectangle([m*2, m*7-lw(20), m*6, m*7], outline=FG, fill=FG)
    # Arm (vertical)
    draw.rectangle([cx-lw(15), m*3, cx+lw(15), m*7-lw(20)],
                   outline=FG, width=lw(3))
    # Stage (horizontal)
    draw.rectangle([m*2, m*5, m*6, m*5+lw(15)], outline=FG, fill=FG)
    # Tube (angled)
    draw.line([cx, m*3, cx+lw(20), m*2], fill=FG, width=lw(8))
    # Eyepiece
    draw.rectangle([cx+lw(15), m*2-lw(15), cx+lw(35), m*2+lw(5)],
                   outline=FG, width=lw(3))
    # Objective
    draw.line([cx, m*5, cx, m*4], fill=FG, width=lw(6))

def draw_dna(draw: ImageDraw.ImageDraw):
    """DNA double helix schematic."""
    m = SIZE // 8
    cx = SIZE // 2
    # Two vertical strands with cross-rungs
    for i in range(8):
        y = m + i * m
        offset = int(m//2 * math.sin(math.radians(i * 45)))
        # Left strand point
        lx = cx - m + offset
        # Right strand point
        rx = cx + m - offset
        draw.ellipse([lx-lw(8), y-lw(8), lx+lw(8), y+lw(8)], fill=FG)
        draw.ellipse([rx-lw(8), y-lw(8), rx+lw(8), y+lw(8)], fill=FG)
        # Rung
        if i > 0:
            prev_y = m + (i-1)*m
            prev_off = int(m//2 * math.sin(math.radians((i-1)*45)))
            draw.line([cx-m+prev_off, prev_y, lx, y], fill=FG, width=lw(2))
            draw.line([cx+m-prev_off, prev_y, rx, y], fill=FG, width=lw(2))
        draw.line([lx, y, rx, y], fill=FG, width=lw(2))

def draw_lightbulb(draw: ImageDraw.ImageDraw):
    """Simple light bulb."""
    m = SIZE // 8
    cx = SIZE // 2
    r = m * 2
    # Globe
    draw.arc([cx-r, m*2, cx+r, m*2+r*2], 0, 180, fill=FG, width=lw(5))
    draw.line([cx-r, m*2+r, cx+r, m*2+r], fill=FG, width=lw(5))
    # Base (screw thread)
    for i in range(3):
        y = m*2+r + i*lw(20)
        draw.line([cx-r+lw(10), y, cx+r-lw(10), y], fill=FG, width=lw(3))
    # Base cap
    draw.rectangle([cx-r+lw(20), m*2+r+lw(60), cx+r-lw(20), m*2+r+lw(80)],
                   outline=FG, fill=FG)
    # Filament
    draw.line([cx, m*2+lw(20), cx-lw(15), m*2+r-lw(20)], fill=FG, width=lw(2))
    draw.line([cx-lw(15), m*2+r-lw(20), cx+lw(15), m*2+r-lw(20)],
              fill=FG, width=lw(2))

def draw_candle(draw: ImageDraw.ImageDraw):
    """Simple candle."""
    m = SIZE // 8
    cx = SIZE // 2
    # Body
    draw.rectangle([cx-m, m*4, cx+m, m*7], outline=FG, width=lw(4))
    # Top oval
    draw.ellipse([cx-m, m*4-lw(10), cx+m, m*4+lw(10)],
                 outline=FG, width=lw(3))
    # Wick
    draw.line([cx, m*4-lw(10), cx, m*3], fill=FG, width=lw(3))
    # Flame
    draw.ellipse([cx-lw(15), m*2, cx+lw(15), m*3], outline=FG, width=lw(3))
    # Drips
    draw.arc([cx+m-lw(20), m*4, cx+m, m*5], 0, 90, fill=FG, width=lw(3))

def draw_flashlight(draw: ImageDraw.ImageDraw):
    """Simple flashlight."""
    m = SIZE // 8
    cx = SIZE // 2
    # Body (handle)
    draw.rectangle([m*3, m*4, m*5, m*7], outline=FG, width=lw(4))
    # Head (wider)
    draw.polygon([(m*2, m*2), (m*6, m*2), (m*5, m*4),
                  (m*3, m*4)], outline=FG, width=lw(4))
    # Lens
    draw.ellipse([m*2+lw(10), m*2+lw(10), m*6-lw(10), m*3],
                 outline=FG, width=lw(3))
    # Button
    draw.ellipse([m*4, m*6, m*4+lw(20), m*6+lw(20)],
                 outline=FG, fill=FG)

def draw_lantern(draw: ImageDraw.ImageDraw):
    """Simple lantern."""
    m = SIZE // 8
    cx = SIZE // 2
    # Handle (arc)
    draw.arc([cx-m, m, cx+m, m*3], 0, 180, fill=FG, width=lw(4))
    # Top
    draw.polygon([(m*2, m*3), (m*6, m*3), (m*5, m*2),
                  (m*3, m*2)], outline=FG, width=lw(3))
    # Body
    draw.rectangle([m*2, m*3, m*6, m*7], outline=FG, width=lw(4))
    # Glass panels (vertical lines)
    for x in [m*3, m*4, m*5]:
        draw.line([x, m*3, x, m*7], fill=FG, width=lw(2))
    # Bottom base
    draw.rectangle([m*2-lw(10), m*7, m*6+lw(10), m*7+lw(15)],
                   outline=FG, fill=FG)

def draw_sparkle(draw: ImageDraw.ImageDraw):
    """Sparkle / star burst."""
    cx, cy = SIZE//2, SIZE//2
    r_long = SIZE*3//8
    r_short = SIZE//6
    for angle in range(0, 360, 45):
        rad = math.radians(angle)
        rad2 = math.radians(angle + 22.5)
        x1 = cx + int(r_long * math.cos(rad))
        y1 = cy + int(r_long * math.sin(rad))
        x2 = cx + int(r_short * math.cos(rad2))
        y2 = cy + int(r_short * math.sin(rad2))
        draw.line([cx, cy, x1, y1], fill=FG, width=lw(4))

def draw_star(draw: ImageDraw.ImageDraw):
    """5-pointed star."""
    cx, cy = SIZE//2, SIZE//2
    r_outer = SIZE*3//8
    r_inner = SIZE*3//16
    pts = []
    for i in range(10):
        angle = math.radians(i * 36 - 90)
        r = r_outer if i % 2 == 0 else r_inner
        pts.extend([cx + int(r * math.cos(angle)),
                    cy + int(r * math.sin(angle))])
    draw.polygon(pts, outline=FG, width=lw(4))

# ── Replacement spec ─────────────────────────────────────────────────────────

REPLACEMENTS = [
    # D6 — Furniture
    ("D6", "D6_gen_chair",    draw_chair,         "Chair line drawing (generated)"),
    ("D6", "D6_gen_sofa",     draw_sofa,          "Sofa line drawing (generated)"),
    ("D6", "D6_gen_table",    draw_table,         "Table line drawing (generated)"),
    ("D6", "D6_gen_bed",      draw_bed,           "Bed line drawing (generated)"),
    ("D6", "D6_gen_mirror",   draw_mirror,        "Mirror line drawing (generated)"),
    ("D6", "D6_gen_pillow",   draw_pillow,        "Pillow line drawing (generated)"),

    # D8 — Tools
    ("D8", "D8_gen_hammer",   draw_hammer,        "Hammer line drawing (generated)"),
    ("D8", "D8_gen_wrench",   draw_wrench,        "Wrench line drawing (generated)"),
    ("D8", "D8_gen_screwdriver", draw_screwdriver, "Screwdriver line drawing (generated)"),
    ("D8", "D8_gen_saw",      draw_saw,           "Hand saw line drawing (generated)"),
    ("D8", "D8_gen_pliers",   draw_pliers,        "Pliers line drawing (generated)"),
    ("D8", "D8_gen_level",    draw_level,         "Spirit level line drawing (generated)"),

    # D9 — Packaging
    ("D9", "D9_gen_bottle",   draw_bottle,        "Bottle line drawing (generated)"),
    ("D9", "D9_gen_box",      draw_box,           "Box line drawing (generated)"),
    ("D9", "D9_gen_tin_can",  draw_tin_can,       "Tin can line drawing (generated)"),
    ("D9", "D9_gen_jar",      draw_jar,           "Jar line drawing (generated)"),
    ("D9", "D9_gen_bag",      draw_bag,           "Bag line drawing (generated)"),
    ("D9", "D9_gen_spray_can", draw_spray_can,    "Spray can line drawing (generated)"),

    # D12 — Transportation
    ("D12", "D12_gen_bicycle",  draw_bicycle,     "Bicycle line drawing (generated)"),
    ("D12", "D12_gen_car",      draw_car,         "Car line drawing (generated)"),
    ("D12", "D12_gen_motorcycle", draw_motorcycle, "Motorcycle line drawing (generated)"),
    ("D12", "D12_gen_bus",      draw_bus,         "Bus line drawing (generated)"),
    ("D12", "D12_gen_truck",    draw_truck,       "Truck line drawing (generated)"),
    ("D12", "D12_gen_wheel",    draw_wheel,       "Wheel line drawing (generated)"),

    # D14 — Electronics
    ("D14", "D14_gen_laptop",   draw_laptop,      "Laptop line drawing (generated)"),
    ("D14", "D14_gen_phone",    draw_phone,       "Phone line drawing (generated)"),
    ("D14", "D14_gen_desktop",  draw_desktop_computer, "Desktop computer (generated)"),
    ("D14", "D14_gen_keyboard", draw_keyboard,    "Keyboard line drawing (generated)"),
    ("D14", "D14_gen_headphones", draw_headphones, "Headphones line drawing (generated)"),
    ("D14", "D14_gen_tablet",   draw_phone,       "Tablet line drawing (generated)"),  # reuse phone shape

    # D23 — Fluid handling
    ("D23", "D23_gen_faucet",   draw_faucet,      "Faucet line drawing (generated)"),
    ("D23", "D23_gen_showerhead", draw_showerhead, "Showerhead line drawing (generated)"),
    ("D23", "D23_gen_toilet",   draw_toilet,      "Toilet line drawing (generated)"),
    ("D23", "D23_gen_bathtub",  draw_bathtub,     "Bathtub line drawing (generated)"),
    ("D23", "D23_gen_fan",      draw_fan,         "Ceiling fan line drawing (generated)"),
    ("D23", "D23_gen_bucket",   draw_bucket,      "Bucket line drawing (generated)"),

    # D24 — Medical instruments
    ("D24", "D24_gen_syringe",  draw_syringe,     "Syringe line drawing (generated)"),
    ("D24", "D24_gen_stethoscope", draw_stethoscope, "Stethoscope line drawing (generated)"),
    ("D24", "D24_gen_thermometer", draw_thermometer, "Thermometer line drawing (generated)"),
    ("D24", "D24_gen_pill",     draw_pill,        "Pill line drawing (generated)"),
    ("D24", "D24_gen_bandage",  draw_bandage,     "Bandage line drawing (generated)"),
    ("D24", "D24_gen_microscope", draw_microscope, "Microscope line drawing (generated)"),

    # D26 — Lighting
    ("D26", "D26_gen_lightbulb", draw_lightbulb,  "Light bulb line drawing (generated)"),
    ("D26", "D26_gen_candle",   draw_candle,      "Candle line drawing (generated)"),
    ("D26", "D26_gen_flashlight", draw_flashlight, "Flashlight line drawing (generated)"),
    ("D26", "D26_gen_lantern",  draw_lantern,     "Lantern line drawing (generated)"),
    ("D26", "D26_gen_sparkle",  draw_sparkle,     "Sparkle/light burst (generated)"),
    ("D26", "D26_gen_star",     draw_star,        "Star/light shape (generated)"),
]

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Fix A FINAL: Generating replacement images with PIL")
    log.info("=" * 60)

    df = pd.read_csv(MANIFEST_PATH)
    log.info(f"Manifest loaded: {len(df)} rows")

    # Remove all bad rows
    def is_bad(row):
        return any(pat in str(row["id"]) for pat in BAD_PATTERNS)

    bad_mask = df.apply(is_bad, axis=1)
    bad_df   = df[bad_mask]
    log.info(f"Bad rows to remove: {len(bad_df)}")
    for _, row in bad_df.iterrows():
        png = Path(row["png_path"])
        if png.exists():
            png.unlink()
        log.info(f"  Removed: {row['id']}")

    df = df[~bad_mask].reset_index(drop=True)
    log.info(f"Rows after cleanup: {len(df)}")

    # Generate replacements
    new_rows = []
    for uspc, stem, draw_fn, description in REPLACEMENTS:
        png_path = PROCESSED_DIR / f"{stem}.png"
        log.info(f"[{uspc}] Generating {stem}.png")

        if not png_path.exists():
            img, draw = new_canvas()
            draw_fn(draw)
            img.save(png_path, format="PNG")

        new_rows.append({
            "id": stem, "label": "negative",
            "source_type": "negative_opensource",
            "uspc_class": uspc, "locarno_class": None,
            "grant_date": None, "invention_title": description,
            "png_path": str(png_path.relative_to(ROOT)),
            "drawing_page": 1, "text_masked": False,
        })

    # Save manifest
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(MANIFEST_PATH, index=False)

    counts = df.groupby(["uspc_class", "label"]).size().unstack(fill_value=0)
    log.info(f"\nDone. Total rows: {len(df)} | Generated: {len(new_rows)}")
    log.info("\nPer-class breakdown:\n" + counts.to_string())
    log.info("\nNext: run 06_select_candidates.py")
    log.info("=" * 60)

if __name__ == "__main__":
    main()