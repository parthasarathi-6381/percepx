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

## 11. Running the pipeline (current state)

Phase 2/3 smoke script — load ONE frame, preprocess, and write an interactive
3D HTML view:

```bash
# real KITTI frame
python -m scripts.load_and_visualize --input data/raw/000000.bin

# or generate + use a synthetic frame (no dataset needed)
python -m scripts.load_and_visualize --synthetic
```

Output: `outputs/raw_pointcloud.html` (open in a browser).

> The full `python -m src.pipeline` CLI, `benchmark`, and `streamlit run
> dashboard/app.py` commands arrive in later phases (see Roadmap).

## 12. Running the dashboard

_Coming in Phase 9._ Will be:

```bash
streamlit run dashboard/app.py
```

## 13. Benchmarking

_Coming in Phase 8._ Will compare a uniform 5 cm grid against the foveated grid
on the same frame(s): cell count, memory, mapping latency, total latency, FPS —
using **real measurements**, never fabricated.

## 14. Metrics

`cell_reduction_percent`, `memory_reduction_percent`,
`latency_reduction_percent`, `FPS_improvement`, plus raw counts (points,
occupied cells). Process memory (`psutil`) is reported separately from a
logical per-cell storage estimate.

## 15. Limitations (current)

- Baseline segmenter is a **geometric prototype**, not a trained network.
- Synthetic frames are for development/demo, clearly labeled — not real sensor
  data.
- Only single-frame processing is wired end-to-end so far (by design — Phase 2
  first).

## 16. Future improvements

PointNet++ / sparse-CNN segmenter, GPU acceleration, ROS2, live LiDAR, object
tracking, and **object-aware local refinement** (e.g. refine a distant 50 cm
zone to 10 cm when a pedestrian is detected). These are *designed for* but not
implemented in the MVP.

---

## Project layout

```
foveated-lidar/
├── configs/config.yaml          # all parameters (paths, ranges, resolutions)
├── data/{raw,processed}/        # frames in, maps out (gitignored)
├── models/                      # pretrained weights (gitignored)
├── src/
│   ├── config.py                # config loader + path resolver + logging
│   ├── preprocessing/           # LiDAR loader + preprocessing        [DONE]
│   ├── segmentation/            # base / baseline / pretrained         [Phase 6]
│   ├── mapping/                 # cell, uniform, policy, foveated       [Phase 4-5]
│   ├── benchmarking/            # benchmark                             [Phase 8]
│   ├── visualization/           # plots (raw done; more later)         [in progress]
│   ├── utils/synthetic.py       # deterministic synthetic frames       [DONE]
│   └── pipeline.py              # end-to-end CLI                        [Phase 9]
├── scripts/load_and_visualize.py  # Phase 2/3 smoke script            [DONE]
├── dashboard/app.py             # Streamlit demo                        [Phase 9]
├── tests/                       # unit tests                            [growing]
├── requirements.txt
└── README.md
```

## Implementation status

| Phase | What | Status |
|-------|------|--------|
| 1 | Setup, structure, config | ✅ done |
| 2 | LiDAR loader + visualize one frame | ✅ done |
| 3 | Preprocessing (range/z filter, invalid removal, voxel) | ✅ done |
| 4 | Uniform 2.5D grid | ⏳ next |
| 5 | Foveated grid + resolution policy | ⏳ |
| 6 | Baseline semantic segmenter | ⏳ |
| 7 | Semantic 2.5D map | ⏳ |
| 8 | Benchmark (memory, cells, latency, FPS) | ⏳ |
| 9 | Streamlit dashboard | ⏳ |
| 10 | Full test + integration suite | ⏳ |
| 11 | Optimization | ⏳ |

## Tests

```bash
python -m pytest
```
