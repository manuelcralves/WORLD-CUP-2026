"""Generate the trophy app icons (PNG) for the PWA / home-screen, drawn to match
the SVG favicon. Run once: `python make_icons.py`."""
from PIL import Image, ImageDraw

GOLD = (255, 203, 92, 255)
DARK = (10, 14, 20, 255)


def make(px: int, path: str):
    ss = 4                      # supersample for crisp edges
    S = px * ss
    sc = S / 64.0               # SVG is on a 0..64 grid
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def R(*v):                  # scale a list of coords
        return [x * sc for x in v]

    d.rounded_rectangle(R(2, 2, 62, 62), radius=14 * sc, fill=DARK)
    cx, cy, r = 32, 24, 13.5
    lw = max(1, int(1.7 * sc))
    # globe (with equator/meridian grid lines)
    d.ellipse(R(cx - r, cy - r, cx + r, cy + r), fill=GOLD)
    d.line(R(cx - r, cy, cx + r, cy), fill=DARK, width=lw)
    d.ellipse(R(cx - r, cy - r * 0.5, cx + r, cy + r * 0.5), outline=DARK, width=lw)
    d.line(R(cx, cy - r, cx, cy + r), fill=DARK, width=lw)
    d.ellipse(R(cx - r * 0.5, cy - r, cx + r * 0.5, cy + r), outline=DARK, width=lw)
    # stem + two-tier base
    d.rectangle(R(30.5, 37, 33.5, 46), fill=GOLD)
    d.rounded_rectangle(R(23, 46, 41, 50), radius=2 * sc, fill=GOLD)
    d.rounded_rectangle(R(18, 50, 46, 55), radius=2.5 * sc, fill=GOLD)

    img.resize((px, px), Image.LANCZOS).save(path)
    print("wrote", path)


if __name__ == "__main__":
    make(512, "icon-512.png")
    make(192, "icon-192.png")
    make(180, "apple-touch-icon.png")
