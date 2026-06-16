"""Generate the trophy app icons (PNG) for the PWA / home-screen, drawn to match
the SVG favicon. Run once: `python make_icons.py`."""
from PIL import Image, ImageDraw

GOLD = (255, 211, 77, 255)
DARK = (12, 16, 24, 255)


def make(px: int, path: str):
    ss = 4                      # supersample for crisp edges
    S = px * ss
    sc = S / 64.0               # SVG is on a 0..64 grid
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def R(*v):                  # scale a list of coords
        return [x * sc for x in v]

    d.rounded_rectangle(R(2, 2, 62, 62), radius=14 * sc, fill=DARK)
    lw = int(3.6 * sc)
    # handles (left/right) — C-shaped arcs hugging the cup
    d.arc(R(6, 13, 22, 31), start=90, end=270, fill=GOLD, width=lw)
    d.arc(R(42, 13, 58, 31), start=270, end=90, fill=GOLD, width=lw)
    # cup: top bar + rounded bowl
    d.rectangle(R(18, 14, 46, 21), fill=GOLD)
    d.pieslice(R(18, 7, 46, 35), start=0, end=180, fill=GOLD)
    # stem + two-tier base
    d.rectangle(R(30, 33, 34, 42), fill=GOLD)
    d.rounded_rectangle(R(22, 42, 42, 46), radius=2 * sc, fill=GOLD)
    d.rounded_rectangle(R(17, 46, 47, 51), radius=2 * sc, fill=GOLD)

    img.resize((px, px), Image.LANCZOS).save(path)
    print("wrote", path)


if __name__ == "__main__":
    make(512, "icon-512.png")
    make(192, "icon-192.png")
    make(180, "apple-touch-icon.png")
