#!/usr/bin/env python3
"""Real-ESRGAN x4 upscale of the Etsy Jewel Tone masters on a Kaggle T4.

Input : /kaggle/input/etsy-jewel-tone-masters/*.png  (1280x1920)
Output: /kaggle/working/upscaled/*.png               (5120x7680) + run-summary.txt
"""
import glob
import os
import subprocess
import sys
import time
import traceback

WORK = "/kaggle/working"
OUT = os.path.join(WORK, "upscaled")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
TIMINGS = []


def phase(name):
    TIMINGS.append((name, time.time() - T0))
    print(f"[{time.time() - T0:7.1f}s] {name}", flush=True)


def main():
    phase("start")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "spandrel"],
                   check=True)
    phase("spandrel installed")

    import urllib.request
    weights = "/tmp/RealESRGAN_x4plus.pth"
    if not os.path.exists(weights):
        urllib.request.urlretrieve(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.1.0/RealESRGAN_x4plus.pth", weights)
    phase(f"weights downloaded ({os.path.getsize(weights) / 1e6:.0f} MB)")

    import numpy as np
    import torch
    from PIL import Image
    from spandrel import ModelLoader

    desc = ModelLoader().load_from_file(weights)
    model = desc.model.eval().cuda()
    scale = desc.scale
    print("model scale:", scale, "| device: cuda", flush=True)
    phase("model loaded")

    def upscale(img_np):
        """Tiled x4 inference, 512px tiles with 16px context padding."""
        h, w, _ = img_np.shape
        tile, pad = 512, 16
        out = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                x0, y0 = max(x - pad, 0), max(y - pad, 0)
                x1, y1 = min(x + tile + pad, w), min(y + tile + pad, h)
                patch = img_np[y0:y1, x0:x1]
                t = torch.from_numpy(patch.transpose(2, 0, 1)).float()
                t = t.unsqueeze(0).cuda() / 255.0
                with torch.no_grad():
                    o = model(t)[0].float().cpu().numpy().transpose(1, 2, 0)
                o = np.clip(o, 0, 1)
                cx0, cy0 = (x - x0) * scale, (y - y0) * scale
                th, tw = min(tile, h - y) * scale, min(tile, w - x) * scale
                out[y * scale:y * scale + th, x * scale:x * scale + tw] = \
                    o[cy0:cy0 + th, cx0:cx0 + tw]
        return (out * 255).round().astype(np.uint8)

    srcs = sorted(glob.glob("/kaggle/input/**/design-*.png", recursive=True))
    if not srcs:
        for root, _dirs, files in os.walk("/kaggle/input"):
            print("INPUT TREE:", root, files[:8], flush=True)
    print("inputs:", [os.path.basename(s) for s in srcs], flush=True)
    for src in srcs:
        t_img = time.time()
        img = np.array(Image.open(src).convert("RGB"))
        up = upscale(img)
        dst = os.path.join(OUT, os.path.basename(src))
        Image.fromarray(up).save(dst)
        print(f"{os.path.basename(src)}: {img.shape[1]}x{img.shape[0]} -> "
              f"{up.shape[1]}x{up.shape[0]} in {time.time() - t_img:.1f}s "
              f"({os.path.getsize(dst) / 1e6:.1f} MB)", flush=True)
        phase(f"done {os.path.basename(src)}")

    with open(os.path.join(WORK, "run-summary.txt"), "w") as fh:
        fh.write(f"total wall time: {(time.time() - T0) / 60:.1f} min\n")
        fh.write("model: RealESRGAN_x4plus (official weights), tiled 512/pad16, fp32\n")
        fh.write("input 1280x1920 -> output 5120x7680 per design\n")
        for name, el in TIMINGS:
            fh.write(f"{name}: {el / 60:.1f} min\n")
    phase("ALL DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        print(err, flush=True)
        with open(os.path.join(WORK, "error.txt"), "w") as fh:
            fh.write(err)
        sys.exit(0)
