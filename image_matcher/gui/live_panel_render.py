"""
A `LivePanel` (lásd `live_panel.py`) kártyáinak TISZTA, widget-mentes
képösszeállító logikája – szándékosan nem importál `tkinter`-t, hogy
unittest-tel valódi ablak/display nélkül tesztelhető legyen (ugyanaz a
minta, mint `argv_builder.py`-nál: a tiszta logika és a widget-wrapper
külön fájlban).

Betűtípus: SEM abszolút Windows-elérési utat, SEM az OS-specifikus
rendszer-betűkészlet meglétét nem feltételezi – néhány gyakori TrueType
nevet próbál (ha a `PIL.ImageFont.truetype()` a keresési útján
megtalálja), de a végső, MINDIG működő tartalék a Pillow-ba csomagolt
skálázható alapértelmezett betűtípus (`ImageFont.load_default(size=...)`,
Pillow 10+) – ez teszi ezt a modult ténylegesen platform-függetlenné.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

CARD_SIZE = (320, 190)

PANEL_BG = (255, 255, 255)
BORDER_IDLE = (200, 200, 200)
BORDER_FOUND = (46, 160, 67)
BORDER_NOT_FOUND = (209, 60, 45)
BORDER_NEAR_MISS = (191, 135, 0)
BADGE_TEXT_COLOR = (10, 10, 10)
PLACEHOLDER_COLOR = (190, 190, 190)

_STATUS_COLOR = {
    "found": BORDER_FOUND,
    "exists": BORDER_FOUND,
    "not_found": BORDER_NOT_FOUND,
    "near_miss": BORDER_NEAR_MISS,
    "error": BORDER_NOT_FOUND,
}


def status_color(status: Optional[str]) -> Tuple[int, int, int]:
    return _STATUS_COLOR.get(status or "", BORDER_IDLE)


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold
        else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # régebbi Pillow: load_default() nem fogad size-t
        return ImageFont.load_default()


def _fit_thumbnail(image_path: Path, box_w: int, box_h: int) -> Optional[Image.Image]:
    try:
        img = Image.open(image_path)
        img.load()
        img = img.convert("RGB")
    except Exception:
        return None
    scale = max(box_w / img.width, box_h / img.height)
    new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    img = img.resize(new_size, Image.LANCZOS)
    left = (img.width - box_w) // 2
    top = (img.height - box_h) // 2
    return img.crop((left, top, left + box_w, top + box_h))


def build_card(
    image_path: Optional[Path],
    border_color: Tuple[int, int, int] = BORDER_IDLE,
    badge_text: Optional[str] = None,
    badge_color: Optional[Tuple[int, int, int]] = None,
    size: Tuple[int, int] = CARD_SIZE,
) -> Image.Image:
    """
    Egy `size` méretű kártyát épít: lekerekített sarkú, `border_color`
    keret, benne a kép (ha `image_path` megadott és betölthető, kitöltve/
    körbevágva), egyébként egy semleges "?" placeholder. `badge_text`
    esetén egy `badge_color` hátterű jelvény kerül a jobb felső sarokba.
    """
    w, h = size
    border_w = 3
    img = Image.new("RGB", size, PANEL_BG)
    draw = ImageDraw.Draw(img)

    inner_margin = border_w + 2
    inner_box = (inner_margin, inner_margin, w - inner_margin, h - inner_margin)
    thumb = _fit_thumbnail(image_path, inner_box[2] - inner_box[0], inner_box[3] - inner_box[1]) \
        if image_path is not None else None
    if thumb is not None:
        img.paste(thumb, (inner_box[0], inner_box[1]))
    else:
        draw.text((w / 2, h / 2), "?", font=_load_font(64, bold=True), fill=PLACEHOLDER_COLOR, anchor="mm")

    draw.rounded_rectangle((border_w // 2, border_w // 2, w - 1 - border_w // 2, h - 1 - border_w // 2),
                            radius=12, outline=border_color, width=border_w)

    if badge_text:
        font = _load_font(14, bold=True)
        pad_x, pad_y = 10, 5
        text_bbox = draw.textbbox((0, 0), badge_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        badge_w = text_w + pad_x * 2
        badge_h = text_h + pad_y * 2
        margin = 10
        bx0, by0 = w - margin - badge_w, margin
        bx1, by1 = bx0 + badge_w, by0 + badge_h
        draw.rounded_rectangle((bx0, by0, bx1, by1), radius=5, fill=badge_color or border_color)
        draw.text(((bx0 + bx1) / 2, (by0 + by1) / 2 - text_bbox[1] / 2), badge_text,
                   font=font, fill=BADGE_TEXT_COLOR, anchor="mm")

    return img
