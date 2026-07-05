"""Generate the 1200x630 social card (og:image) for the site.

An authentic backdrop: the atlas sprite sheet itself (real JWST galaxy
thumbnails), darkened, with a left-to-right gradient so the title stays legible.
Colours mirror the site theme. Output: docs/card.png.

    python -m experiments.build_og_card
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from core import config

W, H = 1200, 630
BG = (6, 7, 13)            # --bg
INK = (233, 236, 246)      # --ink
MUTED = (136, 147, 171)    # --muted
CYAN = (90, 209, 230)      # --cyan
SPRITES = config.WEB_ATLAS_DIR / "sprites.jpg"
OUT = config.WEB_DIR / "card.png"

FONTS = {                  # macOS system fonts (fall back to PIL default)
    "bold": "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "reg": "/System/Library/Fonts/Supplemental/Arial.ttf",
    "mono": "/System/Library/Fonts/Menlo.ttc",
}


def font(kind: str, size: int):
    try:
        return ImageFont.truetype(FONTS[kind], size)
    except OSError:
        return ImageFont.load_default()


def backdrop() -> Image.Image:
    """A darkened crop of the real atlas sprite sheet."""
    base = Image.new("RGB", (W, H), BG)
    if SPRITES.exists():
        sheet = Image.open(SPRITES).convert("RGB")
        # take a wide band and downscale so many little galaxies show
        crop = sheet.crop((0, 0, sheet.width, int(sheet.width * H / W)))
        crop = crop.resize((W, H), Image.LANCZOS)
        base = Image.blend(base, crop, 0.55)          # let galaxies glow through
    # global darkening so text reads
    base = Image.blend(base, Image.new("RGB", (W, H), BG), 0.30)

    # left-to-right gradient: opaque bg on the left, clear on the right
    grad = Image.new("L", (W, 1))
    for x in range(W):
        t = x / W
        grad.putpixel((x, 0), int(240 * max(0.0, 1.0 - (t / 0.72))))
    grad = grad.resize((W, H))
    shade = Image.new("RGB", (W, H), BG)
    base = Image.composite(shade, base, grad)
    # subtle top+bottom darkening
    vg = Image.new("L", (1, H))
    for y in range(H):
        edge = min(y, H - y) / (H / 2)
        vg.putpixel((0, y), int(120 * max(0.0, 1.0 - edge) ** 1.5))
    vg = vg.resize((W, H))
    base = Image.composite(Image.new("RGB", (W, H), BG), base, vg)
    return base


def main() -> None:
    img = backdrop()
    d = ImageDraw.Draw(img)
    x = 68

    d.text((x, 92), "M A C H I N E   L E A R N I N G   ·   J W S T   D E E P   F I E L D S",
           font=font("mono", 20), fill=CYAN)
    d.text((x, 132), "Lynceus", font=font("bold", 132), fill=INK)
    d.text((x, 300),
           "Pulling the faint and the hidden out of", font=font("reg", 36), fill=MUTED)
    d.text((x, 344),
           "the James Webb deep sky.", font=font("reg", 36), fill=MUTED)

    # accent rule + demo list
    d.rectangle((x, 428, x + 96, 432), fill=CYAN)
    d.text((x, 452), "morphology  ·  self-supervised atlas  ·  anomaly hunt  ·  photo-z",
           font=font("mono", 23), fill=INK)
    d.text((x, 520), "mikebertin.github.io/lynceus  —  live, in your browser, no build step",
           font=font("mono", 20), fill=MUTED)

    img.save(OUT, "PNG", optimize=True)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {W}x{H})")


if __name__ == "__main__":
    main()
