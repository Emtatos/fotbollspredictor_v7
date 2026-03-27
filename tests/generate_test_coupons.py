"""
Generate realistic synthetic Swedish betting coupon images for testing.
These simulate the stryktipset/europatipset layout used by Svenska Spel.
"""
import os
import sys

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Pillow required: pip install Pillow")
    sys.exit(1)


def _get_font(size: int):
    """Get a font, falling back to default if no TTF available."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def _get_bold_font(size: int):
    """Get a bold font."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except (OSError, IOError):
        return _get_font(size)


def generate_coupon_image(
    matches: list,
    filename: str,
    title: str = "Stryktipset",
    width: int = 900,
    add_noise: bool = False,
    rotation: float = 0.0,
):
    """
    Generate a coupon image.

    matches: list of dicts with keys:
        nr, home, away, s1, sx, s2, o1, ox, o2
    """
    row_height = 40
    header_height = 80
    col_header_height = 45
    padding = 20
    height = header_height + col_header_height + len(matches) * row_height + padding * 2

    img = Image.new("RGB", (width, height), color="#FFFFFF")
    draw = ImageDraw.Draw(img)

    font = _get_font(16)
    font_small = _get_font(13)
    font_bold = _get_bold_font(18)
    font_title = _get_bold_font(24)

    # Title bar
    draw.rectangle([(0, 0), (width, header_height)], fill="#1B5E20")
    draw.text((width // 2 - 80, 20), title, fill="white", font=font_title)
    draw.text((width // 2 - 60, 52), "Omgang 12 - 2025", fill="#C8E6C9", font=font_small)

    # Column headers
    y_header = header_height + 5
    # (removed stray rectangle call)

    # Define column positions
    col_nr = 15
    col_match = 50
    col_s1 = 450
    col_sx = 520
    col_s2 = 590
    col_o1 = 660
    col_ox = 730
    col_o2 = 800

    # Column header background
    draw.rectangle([(0, header_height), (width, header_height + col_header_height)], fill="#E8F5E9")

    # Column headers
    draw.text((col_nr, y_header + 12), "Nr", fill="#333", font=font_bold)
    draw.text((col_match, y_header + 12), "Match", fill="#333", font=font_bold)
    draw.text((col_s1, y_header + 3), "Sv. folket", fill="#666", font=font_small)
    draw.text((col_s1, y_header + 20), "1", fill="#333", font=font_bold)
    draw.text((col_sx, y_header + 20), "X", fill="#333", font=font_bold)
    draw.text((col_s2, y_header + 20), "2", fill="#333", font=font_bold)
    draw.text((col_o1, y_header + 3), "Odds", fill="#666", font=font_small)
    draw.text((col_o1, y_header + 20), "1", fill="#333", font=font_bold)
    draw.text((col_ox, y_header + 20), "X", fill="#333", font=font_bold)
    draw.text((col_o2, y_header + 20), "2", fill="#333", font=font_bold)

    # Separator line
    y_start = header_height + col_header_height
    draw.line([(0, y_start), (width, y_start)], fill="#999", width=2)

    # Match rows
    for i, m in enumerate(matches):
        y = y_start + i * row_height + 5

        # Alternating row background
        if i % 2 == 0:
            draw.rectangle(
                [(0, y_start + i * row_height), (width, y_start + (i + 1) * row_height)],
                fill="#FAFAFA",
            )

        # Row separator
        draw.line(
            [(0, y_start + (i + 1) * row_height), (width, y_start + (i + 1) * row_height)],
            fill="#DDD",
            width=1,
        )

        # Row number
        draw.text((col_nr, y + 8), str(m["nr"]), fill="#333", font=font)

        # Match (home - away)
        match_text = f"{m['home']} - {m['away']}"
        draw.text((col_match, y + 8), match_text, fill="#111", font=font)

        # Streck
        if m.get("s1") is not None:
            draw.text((col_s1, y + 8), f"{m['s1']}", fill="#555", font=font)
        if m.get("sx") is not None:
            draw.text((col_sx, y + 8), f"{m['sx']}", fill="#555", font=font)
        if m.get("s2") is not None:
            draw.text((col_s2, y + 8), f"{m['s2']}", fill="#555", font=font)

        # Odds
        if m.get("o1") is not None:
            draw.text((col_o1, y + 8), f"{m['o1']:.2f}", fill="#333", font=font)
        if m.get("ox") is not None:
            draw.text((col_ox, y + 8), f"{m['ox']:.2f}", fill="#333", font=font)
        if m.get("o2") is not None:
            draw.text((col_o2, y + 8), f"{m['o2']:.2f}", fill="#333", font=font)

    # Add slight rotation if specified
    if rotation != 0:
        img = img.rotate(rotation, expand=True, fillcolor="#FFFFFF")

    # Add noise if specified
    if add_noise:
        import random
        pixels = img.load()
        for _ in range(int(width * height * 0.01)):
            x = random.randint(0, img.width - 1)
            y = random.randint(0, img.height - 1)
            r, g, b = pixels[x, y]
            noise = random.randint(-20, 20)
            pixels[x, y] = (
                max(0, min(255, r + noise)),
                max(0, min(255, g + noise)),
                max(0, min(255, b + noise)),
            )

    img.save(filename, "PNG")
    return filename


# ---- Coupon 1: Clean, complete stryktipset (13 matches) ----
COUPON_1_MATCHES = [
    {"nr": 1, "home": "Arsenal", "away": "Liverpool", "s1": 42, "sx": 28, "s2": 30, "o1": 2.15, "ox": 3.40, "o2": 3.50},
    {"nr": 2, "home": "Man Utd", "away": "Chelsea", "s1": 35, "sx": 30, "s2": 35, "o1": 2.80, "ox": 3.30, "o2": 2.65},
    {"nr": 3, "home": "Wolves", "away": "Brighton", "s1": 38, "sx": 30, "s2": 32, "o1": 2.50, "ox": 3.20, "o2": 2.90},
    {"nr": 4, "home": "Newcastle", "away": "Tottenham", "s1": 45, "sx": 27, "s2": 28, "o1": 2.10, "ox": 3.50, "o2": 3.40},
    {"nr": 5, "home": "Aston Villa", "away": "West Ham", "s1": 50, "sx": 26, "s2": 24, "o1": 1.85, "ox": 3.60, "o2": 4.20},
    {"nr": 6, "home": "Everton", "away": "Nott'm Forest", "s1": 33, "sx": 30, "s2": 37, "o1": 3.00, "ox": 3.30, "o2": 2.40},
    {"nr": 7, "home": "Bournemouth", "away": "Crystal Palace", "s1": 40, "sx": 29, "s2": 31, "o1": 2.30, "ox": 3.35, "o2": 3.10},
    {"nr": 8, "home": "Fulham", "away": "Brentford", "s1": 36, "sx": 31, "s2": 33, "o1": 2.70, "ox": 3.25, "o2": 2.70},
    {"nr": 9, "home": "Leicester", "away": "Ipswich", "s1": 48, "sx": 27, "s2": 25, "o1": 1.95, "ox": 3.50, "o2": 3.90},
    {"nr": 10, "home": "Southampton", "away": "Leeds", "s1": 34, "sx": 29, "s2": 37, "o1": 2.85, "ox": 3.35, "o2": 2.55},
    {"nr": 11, "home": "Sheff Utd", "away": "QPR", "s1": 52, "sx": 26, "s2": 22, "o1": 1.75, "ox": 3.60, "o2": 4.50},
    {"nr": 12, "home": "Derby", "away": "Preston", "s1": 41, "sx": 29, "s2": 30, "o1": 2.25, "ox": 3.40, "o2": 3.20},
    {"nr": 13, "home": "Luton", "away": "Plymouth", "s1": 39, "sx": 30, "s2": 31, "o1": 2.40, "ox": 3.30, "o2": 3.00},
]

# ---- Coupon 2: Partial data (some missing odds/streck), shorter ----
COUPON_2_MATCHES = [
    {"nr": 1, "home": "AIK", "away": "Djurgarden", "s1": 38, "sx": 29, "s2": 33, "o1": 2.55, "ox": 3.30, "o2": 2.80},
    {"nr": 2, "home": "Hammarby", "away": "Malmo FF", "s1": 42, "sx": 28, "s2": 30, "o1": 2.20, "ox": 3.40, "o2": 3.30},
    {"nr": 3, "home": "IFK Goteborg", "away": "IFK Norrkoping", "s1": 45, "sx": 27, "s2": 28, "o1": 2.10, "ox": 3.50, "o2": 3.40},
    {"nr": 4, "home": "IF Elfsborg", "away": "Hacken", "s1": 40, "sx": 29, "s2": 31, "o1": 2.35, "ox": 3.30, "o2": 3.05},
    {"nr": 5, "home": "Kalmar FF", "away": "Mjallby", "s1": 44, "sx": 28, "s2": 28, "o1": None, "ox": None, "o2": None},
    {"nr": 6, "home": "Sirius", "away": "Varnamo", "s1": 50, "sx": 26, "s2": 24, "o1": None, "ox": None, "o2": None},
    {"nr": 7, "home": "Brommapojkarna", "away": "Halmstad", "s1": 35, "sx": 30, "s2": 35, "o1": 2.75, "ox": 3.25, "o2": 2.65},
    {"nr": 8, "home": "Degerfors", "away": "Sundsvall", "s1": 43, "sx": 28, "s2": 29, "o1": 2.20, "ox": 3.40, "o2": 3.30},
]

# ---- Coupon 3: Noisy image with rotation, some OCR-challenging names ----
COUPON_3_MATCHES = [
    {"nr": 1, "home": "Man City", "away": "Nott'm Forest", "s1": 65, "sx": 18, "s2": 17, "o1": 1.35, "ox": 5.00, "o2": 8.50},
    {"nr": 2, "home": "Sheff Wed", "away": "West Brom", "s1": 36, "sx": 30, "s2": 34, "o1": 2.70, "ox": 3.25, "o2": 2.70},
    {"nr": 3, "home": "Brighton", "away": "Wolverhampton", "s1": 48, "sx": 27, "s2": 25, "o1": 1.95, "ox": 3.50, "o2": 3.90},
    {"nr": 4, "home": "Spurs", "away": "Bournemouth", "s1": 52, "sx": 26, "s2": 22, "o1": 1.80, "ox": 3.55, "o2": 4.30},
    {"nr": 5, "home": "Blackburn", "away": "Stockport", "s1": 44, "sx": 28, "s2": 28, "o1": 2.15, "ox": 3.40, "o2": 3.40},
    {"nr": 6, "home": "Cambridge Utd", "away": "Charlton", "s1": 37, "sx": 30, "s2": 33, "o1": 2.60, "ox": 3.30, "o2": 2.80},
]


def generate_all(output_dir: str = "tests/fixtures/coupons"):
    """Generate all test coupon images."""
    os.makedirs(output_dir, exist_ok=True)

    # Coupon 1: Clean, complete, 13 matches
    f1 = generate_coupon_image(
        COUPON_1_MATCHES,
        os.path.join(output_dir, "coupon_clean_13matches.png"),
        title="Stryktipset",
    )
    print(f"Generated: {f1}")

    # Coupon 2: Swedish league, 8 matches, some missing odds
    f2 = generate_coupon_image(
        COUPON_2_MATCHES,
        os.path.join(output_dir, "coupon_allsvenskan_partial.png"),
        title="Europatipset",
        width=850,
    )
    print(f"Generated: {f2}")

    # Coupon 3: Noisy, slight rotation, 6 matches
    f3 = generate_coupon_image(
        COUPON_3_MATCHES,
        os.path.join(output_dir, "coupon_noisy_rotated.png"),
        title="Stryktipset",
        add_noise=True,
        rotation=1.5,
    )
    print(f"Generated: {f3}")

    return [f1, f2, f3]


if __name__ == "__main__":
    files = generate_all()
    print(f"\nGenerated {len(files)} test coupon images.")
