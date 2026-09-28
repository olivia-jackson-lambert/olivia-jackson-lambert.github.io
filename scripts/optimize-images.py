"""Post-render: lighter, steadier images on the built site.

For every page in _site, each local PNG/JPEG <img> gets a WebP copy (max 1600px
wide) next to it and its src is pointed at that copy. Every image also gets its
intrinsic width/height (so the page does not jump as images arrive) and lazy
loading. Lightbox links keep pointing at the original full-size file.
Only touches the built output; the sources in the repo are unchanged.
"""
import re
import sys
from pathlib import Path

from PIL import Image

SITE = Path(__file__).resolve().parent.parent / "_site"
MAXW = 1600
IMG = re.compile(r"<img\b[^>]*>", re.I)
SRC = re.compile(r'\bsrc="([^"]+)"', re.I)

cache = {}


def to_webp(path: Path):
    if path in cache:
        return cache[path]
    out = path.with_suffix(".webp")
    im = Image.open(path)
    w, h = im.size
    if not out.exists() or out.stat().st_mtime < path.stat().st_mtime:
        rgb = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB")
        if w > MAXW:
            rgb = rgb.resize((MAXW, round(h * MAXW / w)), Image.LANCZOS)
        rgb.save(out, "WEBP", quality=90, method=6)
    cache[path] = (out, w, h)
    return cache[path]


def fix(tag: str, page: Path) -> str:
    m = SRC.search(tag)
    if not m:
        return tag
    src = m.group(1)
    if re.match(r"(https?:|data:|//)", src):
        return tag
    file = (SITE / src.lstrip("/")) if src.startswith("/") else (page.parent / src)
    file = file.resolve()
    if not file.exists():
        return tag
    new = tag
    if file.suffix.lower() in (".png", ".jpg", ".jpeg"):
        out, w, h = to_webp(file)
        new = new.replace(m.group(0), 'src="%s.webp"' % src[: len(src) - len(file.suffix)])
    else:
        with Image.open(file) as im:
            w, h = im.size
    if "width=" not in new:
        new = new.replace("<img", '<img width="%d" height="%d"' % (w, h), 1)
    if "loading=" not in new and "fetchpriority" not in new:
        new = new.replace("<img", '<img loading="lazy" decoding="async"', 1)
    return new


def main():
    pages = [p for p in SITE.rglob("*.html") if "site_libs" not in p.parts]
    n = 0
    for page in pages:
        html = page.read_text(encoding="utf-8")
        out = IMG.sub(lambda m: fix(m.group(0), page), html)
        if out != html:
            page.write_text(out, encoding="utf-8")
            n += 1
    print(f"optimize-images: updated {n} pages, {len(cache)} images converted")


if __name__ == "__main__":
    sys.exit(main())
