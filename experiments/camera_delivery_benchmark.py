"""Compare old PNG/base64 and binary JPEG costs on saved real observations.

Offline encoding/payload measurements only: these are not capture, network,
render FPS or glass-to-glass latency measurements.
"""

import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import __version__ as pillow_version


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input_dir.glob("*.png"))[:8]:
        image = Image.open(path).convert("RGB")
        timings = {"png_base64_ms": [], "jpeg_ms": []}
        for _ in range(3):
            start = time.perf_counter()
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            old = base64.b64encode(stream.getvalue())
            timings["png_base64_ms"].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            stream = io.BytesIO()
            image.save(stream, format="JPEG", quality=80)
            jpeg = stream.getvalue()
            timings["jpeg_ms"].append((time.perf_counter() - start) * 1000)
        rows.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": image.size,
                "old_base64_bytes": len(old),
                "jpeg_bytes": len(jpeg),
                **{k: float(np.median(v)) for k, v in timings.items()},
            }
        )
    if not rows:
        raise ValueError("no saved observation PNGs")
    result = {
        "scope": __doc__,
        "pillow_version": pillow_version,
        "frames": rows,
        "median": {
            key: float(np.median([row[key] for row in rows]))
            for key in ("old_base64_bytes", "jpeg_bytes", "png_base64_ms", "jpeg_ms")
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["median"]))


if __name__ == "__main__":
    main()
