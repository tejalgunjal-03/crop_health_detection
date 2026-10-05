# UAV-Based Intelligent Maize Crop Health Monitoring Using Data Fusion

BE final-year project. Crop: **maize only**. Runs on CPU (Intel i5, 8 GB RAM, no NVIDIA GPU).

> **Honesty statement.** No metric in this repository is hand-written. Accuracy / IoU / Dice values exist only in
> `results/reports/*.json` **after you run the training scripts**. If you have not trained, there are no numbers.

---
## 1. Objective
Help a farmer (a) find suspicious regions in a large maize field from UAV imagery, and (b) get a final crop-health
assessment from a close-up leaf photo combined with local weather.

## 2. System architecture

**Workflow A – large field (UAV)**
```
UAV RGB image -> tiling -> U-Net -> stitched suspicious-region map -> farmer inspects region
      -> close-up leaf image -> leaf classifier --+
      -> GPS/browser location -> weather API -----+--> DATA FUSION (Leaf + Weather) -> Final assessment
```
**Workflow B – small field (direct leaf):** leaf image + location/weather -> fusion -> final assessment. UAV not needed.

## 3. UAV role  (read this before your viva)
The UAV module only **localises suspicious/affected regions**. It is **not** the diagnosis and is **never an input to
fusion**. It is shown in a separate "Field Monitoring Information" box.
> The UAV module is demonstrated using UAV-acquired RGB imagery from the provided dataset. The project does not claim to operate a physical drone.

## 4. Data-fusion rule
`fuse(leaf, weather, cfg)` in `backend/app/ml/fusion/engine.py` accepts **only** `LeafAnalysis` and `WeatherData`.
There is no UAV parameter. `backend/tests/test_fusion.py` asserts the signature, and `test_api.py` spies on the call.
Rules and thresholds are in `configs/fusion_config.yaml` and are **project-defined, not universal agronomic standards**.

Logic (all steps are written into `evidence.rules_applied` in every result):
1. `leaf_risk = Σ probability_i × base_risk_i` (per-class base risk from config)
2. If leaf is diseased and weather exists: `score += weather_weight × (favourable_fraction − 0.5)` where the fraction is the share of temperature / humidity / rainfall criteria inside the configured range for that disease
3. `score += heat_stress_bonus` if temperature ≥ `heat_stress_c`
4. Label from score: Healthy / Mild Stress / Moderate Stress / Severe/Critical. If top leaf confidence < `min_reliable_confidence` → **Inconclusive** (extra label so the system does not guess)
5. "Severity" is derived from the label, **not** lesion area (area is not measured).

Recommendations are generic and safe (inspect, monitor, check irrigation, consult an expert). No pesticide names or doses.

## 5. Datasets (as actually inspected – see `results/reports/`)
**Leaf** (`archive(1).zip`, folder `data/`): 4,188 valid images, 0 corrupt.
Blight 1,146 · Common_Rust 1,306 · Gray_Leaf_Spot 574 · Healthy 1,162 (imbalance max/min = 2.28).
Formats: 4,183 JPEG-decoded as RGB, 4 PNG (RGBA), 1 CMYK image (handled via `.convert("RGB")`). 3,852 images are 256×256, the rest vary widely (266 distinct sizes).
**2 exact-duplicate pairs exist, each filed under two different classes (Blight vs Gray Leaf Spot) – label noise.** The split code keeps one copy; no file is deleted.

**UAV** (Representative Subset v1.0): 1,000 RGB JPG + 1,000 PNG masks, all matched, all 224×224, none corrupt; 500 "water" + 500 "rust" patches (from `meta/subset_manifest.csv`). 1,000 `.npy` multispectral files exist in the zip but are **not extracted or used**.
Mask values found: **0,1,2,3,4**. The dataset README does not define them. Mapping used (**inferred**, see `configs/uav_classes.yaml`):

| value | meaning | evidence |
|---|---|---|
| 0 | soil / background | in both groups; `masks_viz` dark grey |
| 1 | low water stress | only in water group; viz orange |
| 2 | high water stress | only in water group; viz red |
| 3 | healthy vegetation | in both groups, largest (45% of pixels); viz green |
| 4 | rust | only in rust group; viz purple |

Values 0, 3, 4 are well supported. Whether 1 = *low* and 2 = *high* is a colour-convention guess: **confirm with the dataset authors.**
Pixel share: 0=28.5%, 1=5.2%, 2=12.1%, 3=45.3%, 4=8.9%.

### Dataset preparation
Extract so that these exist (the scripts auto-detect nesting):
```
datasets/leaf/data/{Blight,Common_Rust,Gray_Leaf_Spot,Healthy}/*.jpg
datasets/uav/<any folder name>/{images,masks,masks_viz,meta}/
```
UAV: extract everything except `*.npy` (saves ~600 MB). Windows PowerShell: extract with 7-Zip `7z x file.zip -x!*.npy`.
Then: `python scripts/inspect_leaf_dataset.py` and `python scripts/inspect_uav_dataset.py`.

## 6. Models
* **Leaf:** MobileNetV3-Small (ImageNet transfer learning), 224×224, class-weighted cross-entropy, AdamW + cosine LR, on-the-fly augmentation (train only), stratified 70/15/15 split, seed 42, early stopping on validation macro-F1.
* **UAV:** small U-Net (base 16 channels, 4 down/up levels, trained from scratch), 224×224×3 in, 5-class mask out, loss = weighted CE + Dice, batch 4, stratified (water/rust) 70/15/15 split, best epoch by validation mIoU.
* **Large images:** automatically tiled in 224×224 tiles with 50% overlap; probabilities blended with a triangular window and stitched. Stitching is unit-tested to reproduce a per-pixel rule exactly for odd image sizes. Max image size is limited by `UAV_MAX_PIXELS` (default 24 MP) to protect RAM.

## 7. Requirements
Windows 10/11 · Python 3.10–3.12 · Node.js 18+ · ~5 GB free disk (without duplicating datasets) · no CUDA.

## 8. Installation (Windows, from the project root)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r backend\requirements.txt
copy .env.example .env        # then edit .env: set WEATHER_API_KEY and a random SECRET_KEY
cd frontend; npm install; cd ..
```
Weather key: free account at openweathermap.org (new keys can take up to ~2 hours to activate).

## 9. Training  (separate from the API – never runs on upload)
```powershell
python scripts\train_leaf.py --epochs 12                 # add --no-pretrained if offline
python scripts\train_uav.py  --epochs 30
python scripts\evaluate_models.py
```
Outputs: `models/leaf/best_model.pth`, `models/uav/best_model.pth`, and in `results/reports/`: `*_test_metrics.json`, `*_training_history.json`, `leaf_confusion_matrix.png`, `*_splits.json`.
Useful flags: `--batch-size`, `--lr`, `--threads`, `--seed`. If RAM is tight use `--batch-size 16` (leaf) or `2` (UAV).
Checkpoints are git-ignored; keep them in `models/leaf/` and `models/uav/`.

## 10. Run
```powershell
# terminal 1 (backend)
cd backend
uvicorn app.main:app --reload --port 8000
# terminal 2 (frontend)
cd frontend
npm run dev          # open http://localhost:5173
```
Docs for the API: http://localhost:8000/docs . Health: `GET /api/health` shows whether models and the weather key are present.

## 11. API (all except register/login need `Authorization: Bearer <token>`)
| Method | Path | Purpose |
|---|---|---|
| POST | /api/auth/register, /api/auth/login | account, returns token |
| POST | /api/sessions | start a MonitoringSession (`UAV` or `DIRECT_LEAF`) |
| POST | /api/uav/analyze | form: `session_id`, `file` -> overlay, regions (localisation only) |
| GET | /api/uav/result/{id} | stored UAV result |
| POST | /api/leaf/analyze | form: `session_id`, `file` -> disease, confidence, probabilities |
| GET | /api/leaf/result/{id} | stored leaf result |
| GET | /api/weather?lat&lon or ?q=City,CC [&session_id] | weather; with `session_id` it is saved to the session |
| POST | /api/fusion/analyze | `{session_id}` -> fusion of that session's Leaf + Weather |
| GET | /api/results/history, /api/results/{id} | reports |
| GET | /api/health | model/key presence |

Errors are `{ "error": {"code", "message"} }`. Missing model -> HTTP 503 `Trained model not found. Please train the model first.`

## 12. Database (SQLite; set `DATABASE_URL` for PostgreSQL later)
`users` → `crop_fields` → `monitoring_sessions` (mode, pointers to uav_analysis / leaf_analysis / weather / final result)
· `uav_images` → `uav_analyses` (optional) · `leaf_images` → `leaf_analyses` · `weather_data` · `crop_health_results` (links **leaf + weather only**) → `recommendations`.
In DIRECT_LEAF mode the UAV pointer is NULL.

## 13. Folder structure
```
backend/app/{api,core,database,schemas,services,ml/{leaf,uav,fusion}}   backend/tests/
frontend/src/{pages,components}   configs/   scripts/   datasets/   models/   results/   uploads/
```

## 14. Tests
```powershell
cd backend
pytest -q
```
* `test_fusion`, `test_image_validation`, `test_weather`, `test_tiling`: no model/internet needed.
* `test_api`: full workflows with the ML predictors and the weather HTTP call stubbed (wiring + "fusion = Leaf + Weather" rule), plus real missing-model and invalid-image behaviour.
Manual acceptance (UAV workflow and Direct Leaf workflow) is in section 16.

## 15. Troubleshooting
* *"Trained model not found"* – run the training scripts first; check paths in `.env`.
* *Weather "not configured / invalid key"* – set `WEATHER_API_KEY`; wait for activation; the app still works without weather (assessment says weather was unavailable).
* *Location denied* – use "Enter Location Manually" (e.g. `Pune,IN`). Browsers allow location only on `localhost` or HTTPS.
* *Out of memory in training* – lower `--batch-size`, close other apps.
* *Slow UAV analysis* – large images mean many tiles; try a smaller image.

## 16. Manual acceptance checklist
**UAV workflow:** Dashboard → UAV → upload field image → Analyse → overlay + regions shown with "inspect highlighted region" → Continue → upload leaf → Use My Current Location → weather shown → Analyse leaf → Get final assessment → result shows `LEAF + WEATHER = FINAL`, and UAV overlay in a separate dashed "Field Monitoring Information" box.
**Direct workflow:** Dashboard → Direct Leaf → upload leaf → location/weather → analyse → final assessment; no UAV anywhere.

## 17. Known limitations
1. No physical UAV; UAV is demonstrated with dataset imagery. 2. UAV model is RGB-only; multispectral `.npy` not used.
3. The UAV model is trained on 224×224 patches; its behaviour on arbitrary farmer-uploaded field photos (different altitude, camera, resolution) is **unverified** – the "suspicious pixel share" is a model estimate, not a measured area.
4. The mask class meanings are inferred (see §5). 5. The patch file names carry tile coordinates (`y…_x…`) from larger orthomosaics, and neighbouring tiles may exist in different splits, so UAV test scores may be optimistic; the full orthomosaics are not provided.
6. Leaf photos in the dataset are mostly 256×256 (probably controlled/curated); accuracy on real phone photos is not guaranteed. 7. Two duplicate images with conflicting labels exist.
8. Fusion thresholds are project-defined. 9. Weather depends on the API and network; location on browser permission or manual input.
10. Rainfall is the API's last-hour value, a weak proxy for leaf wetness. 11. ML performance depends on dataset quality and training.

## 18. Future work
Physical UAV integration, real-time drone data, multispectral imagery, satellite data, IoT soil-moisture sensors, larger field-scale datasets, cloud deployment, mobile app, better segmentation, more crops/variables, learned multimodal fusion, real-time monitoring.
