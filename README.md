# Kaggle Experiments

Working scripts from my Kaggle runs — written to run as Kaggle kernels (GPU T4), so paths follow the Kaggle layout (`/kaggle/input/...` in, `/kaggle/working/...` out). Each script is self-contained and logs its own summary.

## 1. Credit Card Fraud — LightGBM baseline

**File:** `fraud_detection_lightgbm_baseline.py`
**Competition:** [Playground Series S6E11 — Predicting Credit Card Fraud](https://www.kaggle.com/competitions/playground-series-s6e11)

- LightGBM, 5-fold stratified CV, early stopping per fold.
- Out-of-fold ROC AUC **0.94428** · Public leaderboard **0.94635**.
- This baseline anchored a larger 14-model ensemble that reached **0.95163** public.

## 2. Real-ESRGAN 4x upscale for print masters

**File:** `realesrgan_upscale.py`

- RealESRGAN_x4plus loaded via spandrel, tiled inference on a Kaggle T4.
- Upscales 1280×1920 print-art masters to **5120×7680** (leopard piece: 4480×8960).
- Real run: 6 masters in **10.8 minutes** total — visibly sharper than Lanczos with no colour shift.
- Input: a private Kaggle dataset of PNG masters at `/kaggle/input/etsy-jewel-tone-masters/`.

## 3. SDXL batch generation for Pinterest

**File:** `sdxl_batch_generation.py`

- ComfyUI headless + Juggernaut XL v9, 832×1248 vertical frames, 30 steps, dpmpp_2m / karras, CFG 6.5.
- Generates a 6-image themed batch for a Pinterest channel in about **6.5 minutes** on a T4.
- Every frame is QC'd by a human-style pass afterwards; weak frames get regenerated with a warmer colour grade.

## Notes

- Scripts catch their own exceptions and write `error.txt` to `/kaggle/working` — on Kaggle a "COMPLETE" kernel can still have failed, so the error file is the real signal.
- Competition data is never redistributed here; attach the competition as a data source when you run the kernels.
