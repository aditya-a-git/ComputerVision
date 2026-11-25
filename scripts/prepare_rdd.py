"""Prepare one resumable, immutable selection per output directory."""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import requests
import yaml
from PIL import Image, ImageDraw
from core.data.rdd import MAPPING, NAMES, public, read, sample


def fetch(url, remaining):
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname != "cdn.ul.run":
        raise ValueError("unexpected source")
    expires = parse_qs(parts.query).get("Expires")
    if expires and int(expires[0]) <= time.time():
        raise ValueError("signed URL expired")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(5, 12)) as response:
                if response.status_code in (401, 403, 404):
                    raise ValueError(f"HTTP {response.status_code}")
                response.raise_for_status()
                data = bytearray()
                for chunk in response.iter_content(8192):
                    remaining[0] -= len(chunk)
                    if remaining[0] < 0 or len(data) + len(chunk) > 16_000_000:
                        raise ValueError("byte budget reached")
                    data.extend(chunk)
                return bytes(data)
        except requests.RequestException:
            if attempt == 2:
                raise ValueError("network request failed") from None
            time.sleep(2 ** attempt)


def verify(data, row):
    with Image.open(io.BytesIO(data)) as im:
        if im.size != (row["width"], row["height"]) or im.format not in ("JPEG", "PNG"):
            raise ValueError("image format or dimensions mismatch")
        im.load()
        return im.convert("RGB")


def labels(row):
    return "".join(" ".join(map(str, b)) + "\n" for b in row["boxes"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag, default in (("train",2000),("val",400),("seed",42),("max-images",2400),("max-download-mb",256)):
        p.add_argument("--" + flag, type=int, default=default)
    p.add_argument("--out", type=Path, default=Path("data/processed/rdd"))
    p.add_argument("--local", type=Path, help="Directory of original export images with exact export filenames")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if min(a.train,a.val,a.max_images,a.max_download_mb) < 0:
        p.error("counts must be nonnegative")
    total = a.train + a.val
    cap = min(a.max_images,total)
    val = round(cap*a.val/total) if total else 0
    rows = sample(read(),cap-val,val,a.seed)
    manifest = [public(r) for r in rows]
    print(f"Selected {len(rows)} images, {val} validation requested. Download limit {a.max_download_mb} MB.")
    if a.dry_run:
        return
    root = a.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state = root / "selection.json"
    content = json.dumps({"images":manifest,"mapping":MAPPING,"seed":a.seed},sort_keys=True)
    if state.exists() and state.read_text() != content:
        p.error("different selection already exists here. Choose a fresh --out directory")
    state.write_text(content)
    remaining = [a.max_download_mb * 1_000_000]
    skipped, previews, hashes = [], [], {}
    for row in rows:
        name, split = row["file"], row["split"]
        image = root / "images" / split / name
        label = root / "labels" / split / (Path(name).stem+".txt")
        try:
            if image.exists():
                data = image.read_bytes()
            elif a.local:
                data = (a.local / name).read_bytes()
            elif remaining[0] > 0:
                data = fetch(row["url"], remaining)
            else:
                raise ValueError("byte budget reached")
            im = verify(data,row)
            label.parent.mkdir(parents=True,exist_ok=True)
            image.parent.mkdir(parents=True,exist_ok=True)
            label.write_text(labels(row))
            if not image.exists():
                temp = image.with_suffix(".part")
                temp.write_bytes(data)
                temp.replace(image)
            hashes[name] = hashlib.sha256(data).hexdigest()
            if len(previews)<8:
                draw = ImageDraw.Draw(im)
                for c,x,y,w,h in row["boxes"]:
                    draw.rectangle(((x-w/2)*im.width,(y-h/2)*im.height,(x+w/2)*im.width,(y+h/2)*im.height),outline="red",width=3)
                im.thumbnail((200,150))
                previews.append(im)
        except (ValueError,OSError) as exc:
            skipped.append({"file":name,"reason":str(exc)})
    if previews:
        sheet = Image.new("RGB",(200*min(4,len(previews)),150*((len(previews)+3)//4)))
        for i,im in enumerate(previews):
            sheet.paste(im,(200*(i%4),150*(i//4)))
        sheet.save(root/"preview.jpg")
    (root/"report.json").write_text(json.dumps({"skipped":skipped,"sha256":hashes},indent=2))
    if skipped and any((root/"images"/r["split"]/r["file"]).exists() and r["file"] not in hashes for r in rows):
        raise SystemExit("Existing image failed validation. Repair the reported file before training.")
    (root/"data.yaml").write_text(yaml.safe_dump(dict(path=str(root),train="images/train",val="images/val",names=NAMES)))
    print(f"Verified {len(hashes)} pairs. Skipped {len(skipped)}. See {root}/report.json")


if __name__ == "__main__":
    main()
