"""Image factory for document fixtures.

The face is a real, photorealistic GAN-generated headshot (a person who does
not exist) fetched at *generation time* from thispersondoesnotexist.com, then
composited into a UK driving-licence layout and committed to the repo — so the
live demo still runs fully offline. Using a synthetic, non-existent person is
the correct choice for a privacy/redaction tool: no real likeness is ever used.

The UK has no national ID card (the scheme was scrapped in 2011), so the
identity document modelled here is the **DVLA driving-licence photocard** — the
real UK identity document — rendered with its standard numbered fields, a
valid-format driver number, the real face in the photo box and a handwritten
signature in field 7.

Each builder returns an `ImageAsset` whose `regions` carry gold bounding boxes
(in image pixel space) for the vision track to score against:
ID, FACE, SIGNATURE, TEXT_PII.
"""

from __future__ import annotations

import math
import os
import urllib.request

from PIL import Image, ImageDraw, ImageFont

from dataset.schema import ImageAsset, LabeledImageRegion

_FACE_URL = "https://thispersondoesnotexist.com/"

# Cursive/handwriting fonts to try, in order. Rendered output is committed as a
# PNG, so this only needs to resolve on the generating machine.
_SIGNATURE_FONTS = [
    "/System/Library/Fonts/Supplemental/SnellRoundhand.ttc",
    "/System/Library/Fonts/Supplemental/Brush Script.ttf",
    "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
]
_SANS_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_SANS_BOLD_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _font(candidates, px: int) -> ImageFont.FreeTypeFont:
    for candidate in candidates:
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, px)
            except OSError:
                continue
    return ImageFont.load_default()


def _fetch_face_image(size: int = 600) -> Image.Image:
    """Download a photorealistic synthetic face as an in-memory square image."""
    req = urllib.request.Request(_FACE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    import io
    return Image.open(io.BytesIO(data)).convert("RGB").resize((size, size), Image.LANCZOS)


def _dvla_driver_number(surname: str, first_name: str, dob_iso: str) -> str:
    """Generate a structurally-valid 16-char DVLA driver number from the
    persona. Format: 5 surname chars, decade digit, month, day, year digit,
    two initials, a constant 9, and two check characters.
    """
    year, month, day = (int(p) for p in dob_iso.split("-"))
    sn = (surname.upper() + "99999")[:5]
    decade = str((year // 10) % 10)
    yr = str(year % 10)
    initials = (first_name[:1].upper() + "9")[:2]
    check = "9A"
    return f"{sn}{decade}{month:02d}{day:02d}{yr}{initials}9{check}"


def _render_signature(full_name: str, size: tuple = (440, 150)) -> Image.Image:
    """Render a handwritten signature on a transparent image."""
    w, h = size
    img = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    font = _font(_SIGNATURE_FONTS, px=64)
    bbox = draw.textbbox((0, 0), full_name, font=font)
    tx = (w - (bbox[2] - bbox[0])) // 2 - bbox[0]
    ty = (h - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((tx, ty), full_name, fill=(15, 23, 110, 255), font=font)
    return img


def _draw_guilloche(draw: ImageDraw.ImageDraw, box, colour, lines: int = 26) -> None:
    """Draw a subtle interference/guilloché security pattern inside `box`."""
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    for i in range(lines):
        pts = []
        phase = i * 0.5
        for px in range(0, int(w), 6):
            t = px / w * math.tau * 3
            py = y0 + h / 2 + math.sin(t + phase) * (h / 2 - 4) * math.cos(phase * 0.3)
            pts.append((x0 + px, py))
        draw.line(pts, fill=colour, width=1)


def make_driving_licence(path: str, full_name: str, dob_iso: str, address: str,
                         persona_id: str | None = None,
                         size: tuple = (1012, 638)) -> ImageAsset:
    """Composite a realistic UK DVLA driving-licence photocard.

    Embeds the real GAN face in the photo box and a handwritten signature in
    field 7. Emits four gold regions: ID (whole card), FACE (photo), SIGNATURE,
    and TEXT_PII (the printed personal-details block).
    """
    w, h = size
    surname = full_name.split()[-1]
    first_name = full_name.split()[0]
    dd, mm, yyyy = dob_iso.split("-")[2], dob_iso.split("-")[1], dob_iso.split("-")[0]
    dob_uk = f"{dd}.{mm}.{yyyy}"
    driver_no = _dvla_driver_number(surname, first_name, dob_iso)

    # Card base with the lilac/pink wash of a UK licence.
    img = Image.new("RGB", size, (236, 222, 234))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, w, int(h * 0.42)], fill=(231, 206, 222))
    draw.rectangle([0, int(h * 0.42), w, h], fill=(238, 228, 236))
    _draw_guilloche(draw, (0, 0, w, h), (224, 205, 220), lines=30)

    pad = int(w * 0.03)
    draw.rectangle([4, 4, w - 5, h - 5], outline=(150, 120, 150), width=2)

    # Header band.
    header_h = int(h * 0.12)
    draw.rectangle([0, 0, w, header_h], fill=(123, 36, 92))
    flag_box = (pad, int(header_h * 0.2), pad + int(header_h * 0.9), int(header_h * 0.85))
    draw.rectangle(flag_box, fill=(0, 33, 99))
    draw.line([flag_box[0:2], flag_box[2:4]], fill=(255, 255, 255), width=3)
    draw.line([(flag_box[2], flag_box[1]), (flag_box[0], flag_box[3])], fill=(255, 255, 255), width=3)
    draw.line([((flag_box[0]+flag_box[2])//2, flag_box[1]), ((flag_box[0]+flag_box[2])//2, flag_box[3])], fill=(200, 16, 46), width=4)
    draw.line([(flag_box[0], (flag_box[1]+flag_box[3])//2), (flag_box[2], (flag_box[1]+flag_box[3])//2)], fill=(200, 16, 46), width=4)
    title_font = _font(_SANS_BOLD_FONTS, px=int(header_h * 0.5))
    draw.text((flag_box[2] + pad, int(header_h * 0.25)), "DRIVING LICENCE", fill=(255, 255, 255), font=title_font)

    # Photo box (FACE) on the left.
    photo_x0 = pad
    photo_y0 = header_h + pad
    photo_w = int(w * 0.26)
    photo_h = int(photo_w * 1.25)
    face = _fetch_face_image(size=max(photo_w, photo_h))
    face = face.resize((photo_w, photo_h), Image.LANCZOS)
    img.paste(face, (photo_x0, photo_y0))
    draw.rectangle([photo_x0, photo_y0, photo_x0 + photo_w, photo_y0 + photo_h],
                   outline=(120, 90, 120), width=2)
    face_box = (photo_x0, photo_y0, photo_x0 + photo_w, photo_y0 + photo_h)

    # Numbered personal-details fields (TEXT_PII).
    label_font = _font(_SANS_FONTS, px=15)
    value_font = _font(_SANS_BOLD_FONTS, px=20)
    fx = photo_x0 + photo_w + pad
    fy = header_h + pad
    line_gap = int((h - fy - pad) / 7)

    fields = [
        ("1.", surname.upper()),
        ("2.", first_name.upper()),
        ("3.", f"{dob_uk}  UNITED KINGDOM"),
        ("4a.", "01.05.2019"),
        ("4b.", "30.04.2029"),
        ("4c.", "DVLA"),
        ("5.", driver_no),
    ]
    text_top = fy
    for i, (num, value) in enumerate(fields):
        ly = fy + i * line_gap
        draw.text((fx, ly), num, fill=(90, 60, 90), font=label_font)
        draw.text((fx + int(w * 0.05), ly - 2), value, fill=(30, 20, 40), font=value_font)
    text_box = (fx, text_top - 2, w - pad, fy + len(fields) * line_gap)

    # Field 8 (address) under the photo.
    addr_y = photo_y0 + photo_h + int(pad * 0.4)
    draw.text((photo_x0, addr_y), "8.", fill=(90, 60, 90), font=label_font)
    draw.text((photo_x0 + 24, addr_y), address, fill=(30, 20, 40), font=_font(_SANS_FONTS, px=16))
    text_box = (min(text_box[0], photo_x0), text_box[1], max(text_box[2], w - pad), addr_y + 22)

    # Signature (field 7) bottom-right.
    sig = _render_signature(full_name)
    sig_w = int(w * 0.30)
    sig_h = int(sig.height * (sig_w / sig.width))
    sig = sig.resize((sig_w, sig_h), Image.LANCZOS)
    sig_x = w - pad - sig_w
    sig_y = h - pad - sig_h
    img.paste(sig, (sig_x, sig_y), sig)
    draw.text((sig_x, sig_y - 18), "7.", fill=(90, 60, 90), font=label_font)
    sig_box = (sig_x, sig_y, sig_x + sig_w, sig_y + sig_h)

    img.save(path)
    return ImageAsset(
        image_id=os.path.splitext(os.path.basename(path))[0],
        path=path,
        width=w,
        height=h,
        regions=[
            LabeledImageRegion(label="ID", bbox=(4, 4, w - 5, h - 5), persona_id=persona_id),
            LabeledImageRegion(label="FACE", bbox=face_box, persona_id=persona_id),
            LabeledImageRegion(label="TEXT_PII", bbox=text_box, persona_id=persona_id),
            LabeledImageRegion(label="SIGNATURE", bbox=sig_box, persona_id=persona_id),
        ],
    )
