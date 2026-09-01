# Foveated LiDAR Mapping for Real-Time Autonomous Navigation

> **SIH MVP** — a *distance-adaptive foveated semantic 2.5D* representation of
> LiDAR point clouds. Keep high resolution where it matters (near the vehicle),
> reduce it with distance — inspired by the human eye's fovea.

---

## 1. Problem statement

Autonomous navigation systems process 3D LiDAR point clouds with **millions of
points**. Processing everything at uniform high resolution is expensive in
compute, memory, and latency. Flattening to a traditional 2D occupancy grid is
cheap but throws away the **height information** needed to detect curbs,
potholes, uneven terrain, and overhanging obstacles.

## 2. Motivation

Human vision does not sample the whole field of view at maximum acuity. The
**fovea** gives sharp detail where you look; the periphery is coarse but
cheap. We apply the same idea to LiDAR mapping.

## 3. Proposed solution

```
Raw LiDAR (.bin)
   -> preprocessing        (load, remove invalid, range/z filter, optional voxel)
   -> semantic classification  (baseline geometric segmenter; DL-pluggable)
   -> distance-dependent variable-resolution projection
   -> semantic 2.5D map    (elevation + occupancy + class + confidence per cell)
   -> visualization
   -> benchmark vs a uniform 5 cm grid
```

The key innovation is **not** semantic segmentation alone. It is:

> **Distance-adaptive foveated semantic 2.5D representation.**

## 4. Architecture

```
                 configs/config.yaml   (single source of truth)
                          |
        +-----------------+------------------------------------+
        |                 |                                    |
   preprocessing     segmentation                          mapping
   (lidar_processor)  (base + baseline + pretrained*)   (uniform_grid,
        |                 |                              resolution_policy,
        |                 |                              foveated_grid, cell)
        +--------+--------+------------------+----------------+
                 |                           |
             pipeline.py  --------->   benchmarking/benchmark.py
                 |                           |
           visualization/plots.py      dashboard/app.py (Streamlit)

   * pretrained segmenter = interface/stub until a real model is wired in
```

## 5. Foveated mapping concept

Uniform mapping uses **5 cm everywhere** — a distant wall 90 m away is stored at
the same fidelity as a curb 2 m away, wasting cells. Foveated mapping scales the
cell size with radial distance `r = sqrt(x² + y²)`.

## 6. Resolution zones

| Zone      | Radial distance | Cell size |
|-----------|-----------------|-----------|
| Near      | 0 – 10 m        | **5 cm**  |
| Mid       | 10 – 30 m       | **15 cm** |
| Far       | 30 – 60 m       | **30 cm** |
| Very far  | 60 – 100 m      | **50 cm** |
| (discard) | > 100 m         | —         |

All boundaries and resolutions live in `configs/config.yaml` — never hard-coded.

## 7. 2.5D representation

The output is **not** a full 3D voxel grid. Each occupied 2D cell stores:

- x / y spatial position
- **elevation** (aggregated height — max by default)
- occupancy
- semantic class
- confidence
- point count

This preserves useful vertical information at a fraction of a 3D voxel map's cost.

## 8. Semantic segmentation

- **`BaselineSegmenter`** — geometric heuristics + clustering (no training
  required). Marked clearly as a *prototype classifier*.
- **`PretrainedSegmenter`** — interface/stub so a PointNet++ / sparse-CNN model
  can be plugged in later **without changing the mapping engine**. We do **not**
  fake neural-network accuracy.

Classes: `ground`, `vehicle`, `pedestrian`, `static_obstacle`, `unknown`.

---

## 9. Installation

```bash
# Python 3.10+  (developed on 3.12)
python -m venv .venv
# Windows:  .venv\Scripts\activate     Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

Open3D and PyTorch are **optional** (commented in `requirements.txt`). The MVP
runs on plain NumPy / SciPy / scikit-learn / Plotly / Streamlit on a normal
development machine — no GPU required.

## 10. Dataset preparation

Put KITTI / SemanticKITTI Velodyne frames in `data/raw/`:

```
data/raw/000000.bin      # float32 [N,4] = x, y, z, intensity
```

`.bin` files are gitignored (large + license). **No dataset yet?** A
deterministic **synthetic** frame generator is included so you can run and test
everything immediately:

```bash
python -m src.utils.synthetic --out data/raw/synthetic_000000.bin
```

## 11. Running the pipeline

Full pipeline (load → preprocess → segment → uniform map → foveated map →
benchmark), with per-stage timing:

```bash
# real KITTI frame
python -m src.pipeline --input data/raw/000000.bin

# or a deterministic synthetic frame (no dataset needed)
python -m src.pipeline --synthetic

# also write interactive HTML views + a JSON summary
python -m src.pipeline --input data/raw/000000.bin --viz --json outputs/run.json
```

There is also a lightweight Phase-2/3 smoke script (load + preprocess + raw view):

```bash
python -m scripts.load_and_visualize --synthetic
```

## 12. Running the dashboard

```bash
streamlit run dashboard/app.py
```

Then pick a frame (or "Use synthetic frame"), set options, and click
**Run Pipeline**. Sections: raw cloud → semantic cloud → foveated 2.5D map →
uniform-vs-foveated → performance metrics. No Python knowledge needed.

## 13. Benchmarking

Compare a uniform 5 cm grid against the foveated grid on the same frame(s):

```bash
python -m src.benchmarking.benchmark --synthetic
python -m src.benchmarking.benchmark --input data/raw/          # a directory
python -m src.benchmarking.benchmark --input data/raw/000000.bin --json out.json
```

All numbers are **real measurements**, never fabricated.

### Results on REAL KITTI data

Measured on **20 real KITTI frames** (raw drive `2011_09_26_drive_0001`,
~122k points/frame; see [data/DATASET.md](data/DATASET.md)):

| metric (mean over 20 frames) | uniform 5 cm | foveated |
|------------------------------|-------------:|---------:|
| cells                        |      ~75,041 |  ~45,501 |
| mapping FPS (1/mapping)      |        ~41   |    ~29   |

**Mean cell reduction ≈ 39%  →  mean logical-memory reduction ≈ 39%.**

Single real frame (frame 0) example:

| metric          | uniform 5 cm | foveated |
|-----------------|-------------:|---------:|
| cells           |       75,009 |   45,911 |
| logical storage |     ~4.69 MB*|  ~2.87 MB*|
| cell reduction  |            — |  **38.8%**|

*logical storage = cells × 64 B/cell (configurable).

Per-zone reduction grows with distance, and on real data the far field carries
real structure (roads/buildings extending out), so the **aggregate reduction is
~39%** — much larger than on the near-field-dense synthetic frame (~14%). This
is the honest, expected behavior: foveation pays off most when the scene has
mid/far content.

Foveation's win is fewer cells → less memory (which compounds for every
downstream consumer of the map — storage, transmission, planning). Mapping
*construction* latency is comparable between the two (both real-time, 29–41 FPS);
it is not the metric foveation optimizes. See §15.

> A deterministic **synthetic** frame is also included (`--synthetic`) so the
> pipeline runs with no dataset. Its reduction (~14%) is smaller by design
> because its scene is near-field-dense.

## 14. Metrics

`cell_reduction_percent`, `memory_reduction_percent`,
`latency_reduction_percent`, `FPS_improvement`, plus raw counts (points,
occupied cells). Process memory (`psutil`) is reported separately from a
logical per-cell storage estimate.

## 15. Limitations (current)

- Baseline segmenter is a **geometric prototype** (RANSAC ground + DBSCAN +
  geometry rules), not a trained network. Honestly flagged everywhere.
- **DBSCAN segmentation is the runtime bottleneck** (~2 s on a 119k-point
  frame, ~85% of total time). It is a *classifier* cost, not a *mapping* cost;
  slated for the Phase-11 optimization pass (or replaced by a trained model).
- Foveated cell/memory reduction depends on the scene: it is largest when the
  mid/far field carries structure. On near-field-dense frames the aggregate is
  modest (~14%) even though every zone individually shrinks.
- Foveated map **construction latency** is comparable to (not faster than)
  uniform — foveation optimizes cell count / memory, not build speed.
- Synthetic frames are for development/demo, clearly labeled — not real sensor
  data. The loader is format-identical for real KITTI `.bin`, so swapping in a
  real frame requires no code change.
- Single-frame pipeline is fully wired; multi-frame is supported by the
  benchmark CLI (directory input) and aggregate metrics.

## 16. Future improvements

PointNet++ / sparse-CNN segmenter, GPU acceleration, ROS2, live LiDAR, object
tracking, and **object-aware local refinement** (e.g. refine a distant 50 cm
zone to 10 cm when a pedestrian is detected). These are *designed for* but not
implemented in the MVP.

---

## 17. Environmental sensor monitoring (Grove Beginner Kit)

Rain, fog, smoke and dust distort a real LiDAR return (beam scattering /
attenuation). A physical Grove Beginner Kit + one added analog air-quality
sensor gives proxy signals for those conditions, classified into
**good / moderate / bad** and shown live on a separate dashboard page.

| Sensor (Grove Beginner Kit) | Pin  | Proxy for |
|---|---|---|
| Humidity (AHT20/DHT20)      | I2C (0x38) | Fog / rain likelihood |
| Air pressure (BMP280)       | I2C (0x77, falls back to 0x76) | A fast drop signals incoming rain/storm |
| Sound sensor                | A2   | Rain on surfaces / wind noise |
| 3-axis accelerometer (LIS3DHTR) | I2C (0x19) | Wind / mount instability (vibration) |
| Air Quality Sensor v1.3 *(added)* | A0 *(wired into the Sound Sensor's original socket; the on-board potentiometer was removed to make room)* | Smoke / dust / VOCs |
| OLED display 0.96"          | I2C  | On-device readout of every sensor at once, no laptop required |

**Setup:**

```bash
# 1. Flash the sketch (Arduino IDE) -- see the header comment in the .ino
#    for the exact library list.
arduino/env_lidar_monitor/env_lidar_monitor.ino

# 2. Run the serial bridge (separate terminal, project root, venv active)
python -m src.env_sensors.serial_bridge --port COM20

# 3. Open the dashboard -- the new page polls the bridge's live snapshot
streamlit run dashboard/app.py
# -> sidebar page "Environmental Sensors"
```

All thresholds live in `configs/config.yaml` under `environmental_sensors`
(never hard-coded), and the overall verdict is the *worst* rating among the
five proxy signals. These are honest proxies from a low-cost hobby kit, not
calibrated meteorological instruments — retune the limits against your own
deployment site before trusting the verdict operationally.

---

## Project layout

```
foveated-lidar/
├── configs/config.yaml          # all parameters (paths, ranges, resolutions)
├── data/{raw,processed,live}/   # frames in, maps out, live sensor snapshot (gitignored)
├── models/                      # pretrained weights (gitignored)
├── arduino/env_lidar_monitor/   # Grove Beginner Kit sketch (env sensors -> serial JSON)
├── src/
│   ├── config.py                # config loader + path resolver + logging
│   ├── preprocessing/           # LiDAR loader + preprocessing
│   ├── segmentation/            # base / baseline / pretrained + classes + factory
│   ├── mapping/                 # cell, uniform_grid, resolution_policy, foveated_grid, semantic_map
│   ├── benchmarking/            # benchmark + CLI (__main__)
│   ├── visualization/           # plots (raw, semantic, grid, zones, comparison)
│   ├── env_sensors/             # good/moderate/bad classifier + Arduino serial bridge
│   ├── utils/synthetic.py       # deterministic synthetic frames
│   └── pipeline.py              # end-to-end pipeline + CLI
├── scripts/load_and_visualize.py  # Phase 2/3 smoke script
├── dashboard/
│   ├── app.py                   # Streamlit demo dashboard (main page)
│   ├── theme.py                 # shared CSS design system for every page
│   └── pages/                   # Environmental Sensors (live monitoring page)
├── tests/                       # 90+ unit + integration tests
├── requirements.txt
└── README.md
```

## Implementation status

| Phase | What | Status |
|-------|------|--------|
| 1 | Setup, structure, config | ✅ done |
| 2 | LiDAR loader + visualize one frame | ✅ done |
| 3 | Preprocessing (range/z filter, invalid removal, voxel) | ✅ done |
| 4 | Uniform 2.5D grid | ✅ done |
| 5 | Foveated grid + resolution policy | ✅ done |
| 6 | Baseline semantic segmenter (+ pretrained stub) | ✅ done |
| 7 | Semantic 2.5D map | ✅ done |
| 8 | Benchmark (memory, cells, latency, FPS) | ✅ done |
| 9 | Central pipeline + visualizations + Streamlit dashboard | ✅ done |
| 10 | Full test + integration suite (90+ tests) | ✅ done |
| 11 | Optimization (DBSCAN segmentation) | ⏳ next |

## Tests

```bash
python -m pytest
```
