#!/usr/bin/env python3
"""SDXL batch image generation for Pinterest (Afrohemian Week 7: Rattan & Cane).

ComfyUI headless + Juggernaut XL v9. Generates 6 vertical images
(832x1248, ~2:3) into /kaggle/working/pins/ + run-summary.txt.
"""
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request

WORK = "/kaggle/working"
ROOT = "/tmp/sdxlwork"
COMFY = os.path.join(ROOT, "ComfyUI")
T0 = time.time()
TIMINGS = []


def phase(name):
    TIMINGS.append((name, time.time() - T0))
    print(f"[{time.time() - T0:7.1f}s] {name}", flush=True)


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


STYLE = ("afrohemian interior, african boho home decor, warm earthy tones, "
         "terracotta and ochre accents, mudcloth cushions, woven textures, "
         "natural materials, soft window light, photorealistic interior "
         "photography, cozy and calm, high detail")
NEG = ("people, faces, text, watermark, logo, cartoon, illustration, "
       "low quality, blurry, deformed furniture, oversaturated")

PROMPTS = [
    ("w7-1-rattan-living-room",
     "bright living room with a natural rattan peacock chair, cane webbing "
     "sideboard, jute rug, clay pots with pampas grass, " + STYLE),
    ("w7-2-cane-bedroom",
     "serene bedroom with a rattan headboard with cane webbing panels, "
     "linen bedding in warm cream, mudcloth throw pillows, woven pendant "
     "light, wooden nightstand with a small brass lamp, " + STYLE),
    ("w7-3-rattan-dining",
     "dining corner with six woven rattan chairs around a rustic wooden "
     "table, cane pendant lights above, terracotta vase centerpiece with "
     "dried grasses, woven placemats, " + STYLE),
    ("w7-4-cane-entryway",
     "welcoming entryway with a cane webbing console table, round rattan "
     "mirror, woven seagrass baskets below, a mudcloth runner rug, clay "
     "vases and a small olive tree in a terracotta pot, " + STYLE),
    ("w7-5-rattan-reading-nook",
     "cozy reading nook with a hanging rattan egg chair with cream cushions, "
     "small cane side table with books and a cup of tea, woven wall baskets "
     "as decor, soft afternoon light through linen curtains, " + STYLE),
    ("w7-6-cane-living-detail",
     "living room detail of a cane webbing cabinet styled with stacked "
     "art books, a brass candle holder, a small woven basket and a "
     "terracotta sculpture, mudcloth cushion on a linen sofa beside it, "
     + STYLE),
]


def download(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        return
    print("downloading", url, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dst, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)


def main():
    phase("start")
    os.makedirs(ROOT, exist_ok=True)
    if not os.path.isdir(COMFY):
        sh(["git", "clone", "-q", "--depth", "1",
            "https://github.com/comfyanonymous/ComfyUI.git", COMFY])
    sh([sys.executable, "-m", "pip", "install", "-q", "-r",
        os.path.join(COMFY, "requirements.txt")])
    phase("ComfyUI installed")

    ckpt_dir = os.path.join(COMFY, "models", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, "juggernautXL_v9.safetensors")
    download("https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/"
             "main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors", ckpt)
    phase(f"checkpoint ready ({os.path.getsize(ckpt) / 1e9:.1f} GB)")

    out_dir = os.path.join(COMFY, "output")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    server = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188",
         "--disable-pinned-memory"],
        cwd=COMFY, env=env,
        stdout=open("/tmp/comfy.log", "w"), stderr=subprocess.STDOUT)
    try:
        import urllib.error
        for _ in range(120):
            try:
                urllib.request.urlopen("http://127.0.0.1:8188/system_stats",
                                       timeout=5)
                break
            except Exception:
                time.sleep(5)
        else:
            raise RuntimeError("ComfyUI server did not come up")
        phase("server up")

        def queue(prompt_graph):
            data = json.dumps({"prompt": prompt_graph}).encode()
            req = urllib.request.Request(
                "http://127.0.0.1:8188/prompt", data=data,
                headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req).read())["prompt_id"]

        def wait_done(pid, timeout=1800):
            start = time.time()
            while time.time() - start < timeout:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:8188/history/{pid}") as r:
                    hist = json.loads(r.read())
                if pid in hist:
                    status = hist[pid]["status"]
                    if status.get("status_str") == "error":
                        raise RuntimeError(str(status))
                    return hist[pid]
                time.sleep(10)
            raise TimeoutError(pid)

        os.makedirs(os.path.join(WORK, "pins"), exist_ok=True)
        for i, (name, prompt) in enumerate(PROMPTS):
            graph = {
                "1": {"class_type": "CheckpointLoaderSimple",
                      "inputs": {"ckpt_name": "juggernautXL_v9.safetensors"}},
                "2": {"class_type": "CLIPTextEncode",
                      "inputs": {"text": prompt, "clip": ["1", 1]}},
                "3": {"class_type": "CLIPTextEncode",
                      "inputs": {"text": NEG, "clip": ["1", 1]}},
                "4": {"class_type": "EmptyLatentImage",
                      "inputs": {"width": 832, "height": 1248, "batch_size": 1}},
                "5": {"class_type": "KSampler",
                      "inputs": {"seed": 777000 + i, "steps": 30, "cfg": 6.5,
                                 "sampler_name": "dpmpp_2m",
                                 "scheduler": "karras", "denoise": 1.0,
                                 "model": ["1", 0], "positive": ["2", 0],
                                 "negative": ["3", 0], "latent_image": ["4", 0]}},
                "6": {"class_type": "VAEDecode",
                      "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
                "7": {"class_type": "SaveImage",
                      "inputs": {"filename_prefix": name, "images": ["6", 0]}},
            }
            t_img = time.time()
            hist = wait_done(queue(graph))
            for node in hist["outputs"].values():
                for im in node.get("images", []):
                    src = os.path.join(out_dir, im.get("subfolder", ""),
                                       im["filename"])
                    dst = os.path.join(WORK, "pins", name + ".png")
                    with open(src, "rb") as a, open(dst, "wb") as b:
                        b.write(a.read())
                    print(f"{name}: {time.time() - t_img:.0f}s", flush=True)
            phase(f"done {name}")
    finally:
        server.terminate()

    with open(os.path.join(WORK, "run-summary.txt"), "w") as fh:
        fh.write(f"total wall time: {(time.time() - T0) / 60:.1f} min\n")
        fh.write("model: Juggernaut XL v9, 832x1248, 30 steps dpmpp_2m karras, cfg 6.5\n")
        fh.write("batch: Afrohemian Week 7 theme - Rattan & Cane (6 images)\n")
        for name, el in TIMINGS:
            fh.write(f"{name}: {el / 60:.1f} min\n")
    phase("ALL DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        print(err, flush=True)
        tail = ""
        if os.path.exists("/tmp/comfy.log"):
            tail = open("/tmp/comfy.log").read()[-3000:]
        with open(os.path.join(WORK, "error.txt"), "w") as fh:
            fh.write(err + "\n--- comfy.log tail ---\n" + tail)
        sys.exit(0)
