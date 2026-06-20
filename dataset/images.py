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


def make_id_card(path: str, full_name: str, dob_iso: str, address: str,
                 persona_id: str | None = None,
                 size: tuple = (640, 400)) -> ImageAsset:
    """A clean photo-ID card: solid header band, a real face in the photo box
    on the left, and a printed personal-details block on the right.

    Emits three gold regions: ID (whole card), FACE (photo), and TEXT_PII (the
    printed details block). The signature is emitted as its own image so the
    vision track gets a clean, isolated SIGNATURE target.
    """
    w, h = size
    surname = full_name.split()[-1]
    first_name = full_name.split()[0]
    dd, mm, yyyy = dob_iso.split("-")[2], dob_iso.split("-")[1], dob_iso.split("-")[0]
    dob_uk = f"{dd}.{mm}.{yyyy}"
    doc_number = _dvla_driver_number(surname, first_name, dob_iso)

    img = Image.new("RGB", size, (228, 236, 246))
    draw = ImageDraw.Draw(img)

    # Card border (ID region) and solid header band.
    draw.rectangle([4, 4, w - 5, h - 5], outline=(40, 60, 110), width=3)
    header_h = int(h * 0.16)
    draw.rectangle([4, 4, w - 5, header_h], fill=(40, 60, 110))
    title_font = _font(_SANS_BOLD_FONTS, px=int(header_h * 0.42))
    draw.text((20, int(header_h * 0.28)), "UNITED KINGDOM  -  IDENTITY CARD",
              fill=(255, 255, 255), font=title_font)

    # Photo box (FACE) on the left, holding the real face.
    pad = int(w * 0.035)
    photo_x0 = pad
    photo_y0 = header_h + pad
    photo_w = int(w * 0.30)
    photo_h = int(photo_w * 1.2)
    face = _fetch_face_image(size=max(photo_w, photo_h))
    face = face.resize((photo_w, photo_h), Image.LANCZOS)
    img.paste(face, (photo_x0, photo_y0))
    draw.rectangle([photo_x0, photo_y0, photo_x0 + photo_w, photo_y0 + photo_h],
                   outline=(40, 60, 110), width=2)
    face_box = (photo_x0, photo_y0, photo_x0 + photo_w, photo_y0 + photo_h)

    # Printed personal-details block (TEXT_PII) on the right.
    label_font = _font(_SANS_FONTS, px=18)
    text_x = photo_x0 + photo_w + pad
    text_top = header_h + pad + 6
    line_h = 34
    lines = [
        f"Surname:  {surname}",
        f"Given names:  {first_name}",
        f"Date of birth:  {dob_uk}",
        f"Document No:  {doc_number}",
        "Nationality:  British",
    ]
    for i, line in enumerate(lines):
        draw.text((text_x, text_top + i * line_h), line, fill=(20, 30, 60), font=label_font)
    text_box = (text_x, text_top, w - pad, text_top + line_h * len(lines))

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
        ],
    )


def make_signature(path: str, full_name: str, persona_id: str | None = None,
                   size: tuple = (520, 200)) -> ImageAsset:
    """A standalone handwritten signature on a white card (SIGNATURE target)."""
    w, h = size
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    sig = _render_signature(full_name, size=(int(w * 0.84), int(h * 0.7)))
    sx = (w - sig.width) // 2
    sy = (h - sig.height) // 2 - int(h * 0.05)
    img.paste(sig, (sx, sy), sig)
    draw.line([(int(w * 0.08), int(h * 0.78)), (int(w * 0.92), int(h * 0.78))],
              fill=(120, 120, 120), width=2)
    img.save(path)
    sig_box = (sx, sy, sx + sig.width, sy + sig.height)
    return ImageAsset(
        image_id=os.path.splitext(os.path.basename(path))[0],
        path=path,
        width=w,
        height=h,
        regions=[LabeledImageRegion(label="SIGNATURE", bbox=sig_box, persona_id=persona_id)],
    )
