# Foveated LiDAR Mapping — Complete Learning & Solution Guide

> **Purpose of this file.** A single, self-contained document that teaches you
> (or a teammate / a judge) the *entire* working of this project: the SIH problem,
> why the standard approaches fail, the novel idea we chose, the exact algorithms
> in the code, how data flows end-to-end, what the numbers mean, and how to
> present/defend it. Read top to bottom — no prior context needed.

---

## Part 0 — TL;DR (read this first)

**Problem (SIH):** Turn raw 3D LiDAR point clouds (millions of points) into a
compact **2.5D semantic map** for autonomous navigation, in real time, without
losing the height information that matters for safety (curbs, potholes,
overhangs).

**The tension:**
- Full 3D processing = accurate but **too heavy** (compute, memory, latency).
- Flat 2D occupancy grid = cheap but **loses height** → dangerous.

**Our novel idea — "foveated" mapping (like the human eye):**
Keep **high resolution near the vehicle** (where a 5 cm curb can hurt you) and
**progressively coarser resolution far away** (where 50 cm is plenty). This is a
**variable-resolution 2.5D grid**.

**What we actually built (working MVP):**
```
Raw LiDAR (.bin)
  → preprocess (filter range/height, drop invalid, optional voxel)
  → semantic segmentation (ground / vehicle / pedestrian / static / unknown)
  → project to a UNIFORM 2.5D grid   (the baseline to beat)
  → project to a FOVEATED 2.5D grid  (our contribution)
  → benchmark the two (cells, memory, latency, FPS) — REAL measured numbers
  → visualize in a Streamlit dashboard
```

**Headline result (real KITTI, 20 frames):** the foveated map uses **~39% fewer
cells → ~39% less memory** than a uniform 5 cm grid, while preserving elevation
and semantics everywhere — both maps build in real time (~29–41 FPS).

**The one sentence that makes us different:**
> *Distance-adaptive foveated semantic 2.5D representation* — not "segmentation",
> not "another occupancy grid". The novelty is the **adaptive spatial data
> structure** that spends memory where safety needs it and saves it everywhere else.

---

## Part 1 — The problem, precisely

### 1.1 What autonomous navigation needs from perception
A self-driving vehicle needs a map of its surroundings that answers:
- **Where can I drive?** (drivable surface vs not) — *terrain analysis*
- **What is around me?** (walls, poles = static; pedestrians, cars = dynamic) —
  *object detection / classification*
- **How tall / low is it?** (a curb is 15 cm; a pothole is negative; an overhang
  is above the road) — *height / elevation*, which a flat 2D map throws away.

### 1.2 Why the two obvious representations both fail

| Representation | Pro | Fatal con |
|---|---|---|
| **Raw 3D point cloud** (millions of pts) | Full fidelity | Millions of points per frame → compute + memory + latency blow up. Can't sustain real-time. |
| **3D voxel grid** at 5 cm over ±100 m | Structured, height kept | 4000×4000×(z) cells. Dense storage is *gigabytes*; even sparse it's huge. |
| **Flat 2D occupancy grid** | Cheap, fast | **Loses all height** → can't tell a paint line from a curb, can't see potholes or overhangs. Unsafe. |

### 1.3 The 2.5D compromise (what "2.5D" means)
A **2.5D grid** is a 2D grid (top-down, X–Y) where **each cell stores a height
value** (plus semantics). It's "2 and a half D": the footprint is 2D but each
cell carries elevation. This keeps the safety-critical vertical information at a
tiny fraction of a full 3D voxel grid's cost.

In this project each occupied cell stores (see [src/mapping/cell.py](src/mapping/cell.py)):

| Field | Meaning |
|---|---|
| `x, y` | world position of the cell (its origin corner) |
| `resolution` | the cell's size in meters (constant for uniform; **per-zone** for foveated) |
| `elevation` | aggregated height (default = **max z** of points in the cell) |
| `occupancy` | at least one point landed here |
| `semantic_class` | ground / vehicle / pedestrian / static_obstacle / unknown |
| `confidence` | classification confidence in [0,1] |
| `point_count` | how many points aggregated into this cell |
| `zone` | near / mid / far / very_far / uniform |

### 1.4 The remaining problem 2.5D alone doesn't solve
Even a 2.5D grid at a **uniform** 5 cm wastes memory: a wall 90 m away is stored
at the same 5 cm fidelity as a curb 2 m away, even though at 90 m the sensor is
too sparse to justify it and the planner doesn't need centimeter detail that far
out. **This is exactly the waste foveation removes.**

---

## Part 2 — The novel idea: foveation

### 2.1 The biological analogy
Your eye's **fovea** (center of the retina) has dense photoreceptors → sharp
detail where you look. The **periphery** is sparse → coarse but cheap. You don't
notice because your brain moves the fovea to whatever matters. Total data the eye
sends to the brain is *far* less than a uniform high-res sensor would.

**We apply the identical trade to LiDAR mapping:** high acuity near the vehicle,
coarse far away.

### 2.2 The resolution policy (the heart of the idea)
We split the world into radial **zones** by distance `r = √(x² + y²)` and give
each zone its own cell size. All values live in
[configs/config.yaml](configs/config.yaml) — **never hard-coded**.

| Zone | Radial distance | Cell size | Why |
|---|---|---|---|
| **near** | 0 – 10 m | **5 cm** | safety-critical: curbs, potholes, close pedestrians |
| **mid** | 10 – 30 m | **15 cm** | still important, but LiDAR is sparser here |
| **far** | 30 – 60 m | **30 cm** | planning/context, coarse is fine |
| **very_far** | 60 – 100 m | **50 cm** | situational awareness only |
| *(discard)* | > 100 m | — | out of range, **counted** not silently dropped |

**Why coarser far away is not just "cheaper" but also *correct*:** a spinning
LiDAR's points spread apart with distance (angular resolution is fixed, so arc
length = r·Δθ grows). At 90 m the beams are already meters apart — asking for 5 cm
cells there is asking for detail the sensor never captured. Foveation matches the
grid to the data density.

### 2.3 Why this is genuinely novel for SIH
- It is **not** "we ran PointNet on KITTI" (everyone does that).
- It is **not** "another occupancy grid".
- The contribution is a **variable-resolution 2.5D data structure** that:
  1. tiles space with **no gaps and no overlaps** across resolution changes
     (the hard part — alignment/aliasing errors are the classic failure),
  2. keeps semantics + elevation per cell,
  3. **provably** reduces memory (measured, not claimed),
  4. is **model-agnostic** — the mapping engine doesn't care whether labels come
     from a heuristic or a trained network (so it survives a model upgrade).

---

## Part 3 — Architecture & data flow

### 3.1 Module map
```
configs/config.yaml         ← single source of truth (all parameters)
        │
        ├── src/preprocessing/lidar_processor.py   load + clean the cloud
        ├── src/segmentation/                       classify points
        │     ├── base_segmenter.py   (interface + SegmentationResult)
        │     ├── baseline_segmenter.py  (geometric PROTOTYPE — works today)
        │     ├── pretrained_segmenter.py (stub for PointNet++ / sparse CNN)
        │     ├── classes.py          (class ids + colors)
        │     └── factory.py          (pick segmenter from config)
        ├── src/mapping/                             build the 2.5D maps
        │     ├── cell.py             (the 2.5D cell schema)
        │     ├── uniform_grid.py     (baseline: one resolution everywhere)
        │     ├── resolution_policy.py(distance → zone → cell size)
        │     ├── foveated_grid.py    (THE contribution: 4 zone-grids)
        │     └── semantic_map.py     (builders + config helpers)
        ├── src/benchmarking/benchmark.py            uniform vs foveated, measured
        ├── src/visualization/plots.py               Plotly figures
        ├── src/utils/synthetic.py                   deterministic fake frame
        └── src/pipeline.py                          wires it all together (CLI)

dashboard/app.py            ← Streamlit demo UI (calls the pipeline)
tests/                      ← 90+ unit + integration tests
```

### 3.2 End-to-end flow (what happens on one frame)
Driven by [src/pipeline.py](src/pipeline.py):

```
1. LOAD        LidarProcessor.load(.bin)  → (N,4) float32 [x,y,z,intensity]
2. PREPROCESS  drop NaN/Inf, crop range [0,100m] & height [-3,3m],
               optional voxel downsample                → clean cloud
3. SEGMENT     segmenter.segment(points) → labels[N], confidences[N]
4. UNIFORM MAP build_uniform_map(...)    → UniformGrid  @ 5 cm everywhere
5. FOVEATED MAP build_foveated_map(...)  → FoveatedGrid @ 5/15/30/50 cm
6. BENCHMARK   measure cells, memory, latency, FPS for both
7. (VIZ)       write HTML / show in dashboard
```
Every stage is **timed** (`load_time`, `preprocessing_time`,
`segmentation_time`, `uniform_mapping_time`, `foveated_mapping_time`,
`total_time`) so we can show exactly where the time goes.

---

## Part 4 — The algorithms, in the code

This is the part to understand deeply — judges will probe here.

### 4.1 Preprocessing — [src/preprocessing/lidar_processor.py](src/preprocessing/lidar_processor.py)
- **Load**: read the KITTI `.bin` as `float32`, reshape to `[N,4]` (x,y,z,
  intensity). `point_stride: 4` in config says 4 floats per point.
- **Remove invalid**: drop rows with NaN/Inf.
- **Range crop**: keep points with `min_range ≤ r ≤ max_range` (0–100 m).
- **Height crop**: keep `z_min ≤ z ≤ z_max` (−3…+3 m) — removes the sky/roofs
  and sub-ground noise.
- **Optional voxel downsample** of the *raw* cloud (off by default) — one
  representative point per voxel, independent of the mapping resolution.

### 4.2 The uniform grid — [src/mapping/uniform_grid.py](src/mapping/uniform_grid.py)
This is the **baseline** we beat, and the aggregation engine the foveated grid
reuses. Key ideas:

**Cell index (negative-safe):**
```
grid_x = floor(x / resolution)
grid_y = floor(y / resolution)
```
`floor` (not `int()` truncation) is essential — the sensor is at the origin and
points surround it, so half the coordinates are negative. `floor(-0.01/0.05) = -1`
puts a point just left of origin in cell −1, not cell 0. Truncation would fold
±cells together and corrupt the map.

**Sparse storage:** a dense ±100 m @ 5 cm array = 4000×4000 = 16 million cells,
but one scan only touches a small fraction. We store **only occupied cells** as
parallel NumPy arrays (structure-of-arrays), not one Python object per cell.

**Vectorized group-by (the fast path):** we assign every point to a cell, then
aggregate all points sharing a cell — with no Python loop:
1. Pack `(grid_x, grid_y)` into **one int64 key** (offset by the per-axis min so
   indices are non-negative, then `key = ux * span_y + uy`). This lets us use the
   fast 1-D `np.unique` instead of the slow `np.unique(axis=0)` lexsort.
2. `np.unique(packed, return_inverse=True)` → the list of occupied cells and, for
   each point, which cell it belongs to (`inverse`).
3. Aggregate per cell with `np.bincount` / `np.maximum.at` (scatter-reduce):
   - **elevation** = max (default) / mean / min of z
   - **point_count** = bincount
   - **semantic class** = per-cell **vote-by-confidence**: build an
     `(n_cells, n_classes)` matrix of summed confidence via a flat index, then
     `argmax` → the class with the most confidence "mass" wins the cell
   - **confidence** = mean (default) or max

This matters because latency is a headline metric: a Python loop over 100k points
would be ~100× slower and would make the benchmark meaningless.

### 4.3 The resolution policy — [src/mapping/resolution_policy.py](src/mapping/resolution_policy.py)
Maps a distance array to a **zone index** with a vectorized bucket:
```python
idx = np.searchsorted(self._edges, r, side="left")   # edges = [10,30,60,100]
idx = np.where(idx >= len(edges), -1, idx)            # beyond 100 m → -1 (discard)
```
**Boundary convention (documented + tested):** upper edges are **inclusive**.
A point exactly at 10 m is "near", at 30 m is "mid", at 60 m is "far", at 100 m is
"very_far", beyond 100 m is discarded. `side="left"` on inclusive edges is what
makes an exact-edge point fall *into* that zone.

Why care about the boundary? Because **the classic bug in variable-resolution
grids is a point at a zone edge getting dropped or double-counted.** We nailed the
convention down and wrote tests for it (§7).

### 4.4 The foveated grid — [src/mapping/foveated_grid.py](src/mapping/foveated_grid.py) (THE contribution)
**Design decision: one `UniformGrid` per zone**, each at its own cell size:
```
near_grid     @ 5  cm
mid_grid      @ 15 cm
far_grid      @ 30 cm
very_far_grid @ 50 cm
```
Why separate grids instead of one clever array?
- Different cell sizes live on **different integer lattices**. Mixing them in one
  array would require reconciling indices at every resolution change → the exact
  **alignment/aliasing errors** the problem statement warns about.
- Per-zone grids reuse the **already-tested** uniform aggregation path verbatim.
- Cell coordinates within a zone are unambiguous (all share that zone's res).

**Insert = route then aggregate:**
```python
r = sqrt(x² + y²)
zone_idx = policy.assign_zones(r)          # 0..3, or -1 = discard
discarded_out_of_range += (zone_idx == -1).sum()   # COUNTED, never silent
for zi, zone in enumerate(zones):
    mask = zone_idx == zi
    zone_grids[zone].insert(points[mask], labels[mask], conf[mask])
```
**No data loss guarantee:** the four zones **tile `[0, max_range]` with no gaps
and no overlap**, so every retained point lands in exactly one zone. The only
intentional drop is `r > 100 m`, which is counted and reported.

**Readout:** `to_arrays()` concatenates all zones' occupied cells and adds a
`zone_index` so the dashboard can color by zone / show the changing cell size.
`stats()` reports per-zone cell + point counts (used in the benchmark table).

### 4.5 Semantic segmentation — [src/segmentation/](src/segmentation/)
There are **two** implementations behind one interface (`BaseSegmenter` →
`SegmentationResult(labels, confidences, method, is_prototype, meta)`):

**(a) `BaselineSegmenter` — a geometric PROTOTYPE that works today**
([baseline_segmenter.py](src/segmentation/baseline_segmenter.py)). No training,
no GPU. Three steps:
1. **Ground via RANSAC plane fit** (pure NumPy): seed candidate points from the
   low-z band (so we fit the *road*, not a roof), sample 3 points → plane, count
   inliers within `ground_distance_threshold`, keep the best plane over
   `ground_max_iterations`. **Reject the plane if it isn't roughly horizontal**
   (`|normal_z| < 0.7` → probably a wall) and fall back to a flat z-threshold.
   RANSAC is what makes this robust to a mildly sloped road.
2. **Cluster non-ground** with DBSCAN. **Speed trick:** cluster on a
   *voxel-downsampled* copy (one centroid per voxel), then propagate each voxel's
   cluster id back to all its points. Preserves cluster **shape** (so the geometry
   classifier is unaffected) while cutting DBSCAN's input ~10× on dense frames.
3. **Classify each cluster by explainable geometry** (footprint = largest
   horizontal span, height = vertical extent):
   - tall or large → `static_obstacle` (checked first so a tall thin pole isn't
     called a pedestrian)
   - narrow footprint + human-height band → `pedestrian`
   - car-sized footprint + low → `vehicle`
   - else → `unknown`
   Confidence reflects how cleanly the rule matched.

   > **Honesty rule we enforce:** this is flagged `is_prototype = True`
   > *everywhere* (CLI prints a NOTE, dashboard shows it). We **never** claim
   > neural-network accuracy from a heuristic.

**(b) `PretrainedSegmenter` — the DL slot** ([pretrained_segmenter.py](src/segmentation/pretrained_segmenter.py)):
an interface/stub so a **PointNet++ or Sparse CNN** can be dropped in **without
touching the mapping engine**. This is the "Deep Learning Model" the problem asks
for; the architecture is built to receive it (see §8).

Classes (config-defined ids + colors): `ground(0)`, `vehicle(1)`,
`pedestrian(2)`, `static_obstacle(3)`, `unknown(4)`.

### 4.6 Benchmark — [src/benchmarking/benchmark.py](src/benchmarking/benchmark.py)
Builds both grids on the **same** frame and measures:
- **cells** (uniform vs foveated) — the core win
- **logical memory** = `occupied_cells × bytes_per_cell` (64 B/cell, configurable)
  — an exact model of the map's data footprint
- **process RSS delta** via `psutil` — real but noisy (includes Python/allocator
  overhead), reported **separately** and labeled noisy
- **mapping latency** = best time over `repeat` runs (least-noisy estimate)
- **FPS** = 1 / mapping_time
- derived: `cell_reduction_%`, `memory_reduction_%`, `latency_reduction_%`,
  `fps_improvement`, and a **per-zone** breakdown.

**Intellectual honesty built into the numbers:** the code's own notes state that
foveation's win is **fewer cells → less memory**, and that construction latency is
*comparable* between the two (both real-time) — it is **not** the metric foveation
optimizes. We don't pretend foveation makes mapping faster; we show it makes it
**smaller**, which compounds for every downstream consumer (storage, transmission,
planning).

---

## Part 5 — Results (real, measured)

Measured on **20 real KITTI frames** (raw drive `2011_09_26_drive_0001`,
~122k points/frame):

| metric (mean over 20 frames) | uniform 5 cm | foveated |
|---|--:|--:|
| cells | ~75,041 | ~45,501 |
| mapping FPS | ~41 | ~29 |

**Mean cell reduction ≈ 39% → mean logical-memory reduction ≈ 39%.**

Single real frame (frame 0):

| metric | uniform 5 cm | foveated |
|---|--:|--:|
| cells | 75,009 | 45,911 |
| logical storage | ~4.69 MB | ~2.87 MB |
| cell reduction | — | **38.8%** |

**How to read this honestly (say this to judges):**
- The reduction **grows with distance** (each far zone shrinks a lot), so the
  aggregate depends on the scene: it's largest when the mid/far field carries
  real structure (roads/buildings extending out — true on real KITTI → ~39%).
- On a **near-field-dense synthetic** frame the aggregate is smaller (~14%) even
  though *every zone individually shrinks* — that's expected and we say so.
- Both maps run **real-time** (29–41 FPS). Foveation trades a little build time
  for a big, compounding memory saving — the right trade for an embedded planner.

---

## Part 6 — The dashboard (the demo)
[dashboard/app.py](dashboard/app.py) (Streamlit):
```
streamlit run dashboard/app.py
```
Pick a frame (or "Use synthetic frame"), set options, click **Run Pipeline**.
Sections: raw cloud → semantic cloud → foveated 2.5D map → uniform-vs-foveated →
performance metrics. The map is color-coded by class (or by zone), and the
metrics panel shows the measured cell/memory reduction and FPS. No Python
knowledge needed — this is the "real-time visualization" deliverable.

---

## Part 7 — Correctness: why we trust the numbers
90+ unit + integration tests (`python -m pytest`) cover the risky invariants:
- **Zone boundaries**: points exactly at 10/30/60/100 m land in the right zone;
  >100 m is discarded and **counted**.
- **No data loss**: sum of per-zone points + discarded == input points.
- **Negative coordinates**: `floor` indexing is correct on both sides of origin.
- **Aggregation**: elevation max/mean/min, vote-by-confidence class, counts.
- **End-to-end**: the pipeline runs on a synthetic frame and produces a valid map
  and a benchmark with the expected fields.

This is what lets us say **"all numbers are real measurements, never fabricated."**

---

## Part 8 — How this maps to the SIH deliverables

| SIH asked for | What we deliver | Where |
|---|---|---|
| **Deep Learning model** (PointNet++ / sparse CNN) for semantic segmentation | Working geometric **prototype** now + a clean **DL slot** (`PretrainedSegmenter`) that plugs in without touching mapping | [src/segmentation/](src/segmentation/) |
| **Terrain analysis** (drivable vs not) | `ground` class from RANSAC plane fit | §4.5 |
| **Object detection/classification** (static + dynamic) | `vehicle` / `pedestrian` / `static_obstacle` via cluster geometry | §4.5 |
| **Variable-resolution grid engine** (5 cm @10 m → 50 cm @100 m) | `FoveatedGrid` + `ResolutionPolicy`, alignment-safe, no data loss | §4.3–4.4 |
| **2.5D elevation + semantic layers** | per-cell elevation + class + confidence | [cell.py](src/mapping/cell.py) |
| **Real-time visualization dashboard** | Streamlit app, color-coded, live metrics | §6 |
| **Memory reduction vs uniform 3D/high-res** | measured **~39%** fewer cells/memory | §5 |
| **Performance metrics (FPS, accuracy across distance)** | measured FPS + per-zone breakdown + benchmark CLI | §4.6, §5 |

---

## Part 9 — Honest limitations (state these before a judge does)
- Baseline segmenter is a **geometric prototype**, not a trained network (flagged
  everywhere). Accuracy is "reasonable heuristic", not SOTA.
- **DBSCAN is the runtime bottleneck** (~85% of frame time). It's a *classifier*
  cost, not a *mapping* cost — replaced by the DL model or the Phase-11 optimizer.
- Foveated **reduction depends on the scene** (largest with mid/far structure).
- Foveated **build latency ≈ uniform** — foveation optimizes memory, not build
  speed. We say this plainly.
- Synthetic frames are for demo only, clearly labeled; the loader is
  format-identical to real KITTI so swapping in a real frame needs no code change.
- Single-frame pipeline is fully wired; multi-frame is via the benchmark CLI.

---

## Part 10 — Proposed roadmap (what "novel + optimized" becomes next)
Ordered by impact for the SIH goal (optimize compute + memory):

1. **Wire a real DL segmenter** into `PretrainedSegmenter` (PointNet++ or a
   **Sparse CNN** like MinkowskiEngine/SpConv). Sparse convs are the natural fit —
   they already exploit sparsity, matching our sparse grid philosophy.
2. **Object-aware local refinement (foveation++):** when a pedestrian/vehicle is
   detected in a coarse far zone, *locally* refine that patch to a finer cell size
   — spend detail on *objects*, not just on *near distance*. This is the eye
   "saccading" to something interesting. Genuinely novel, directly on-theme.
3. **Replace DBSCAN** with a faster clustering (connected-components on the grid,
   or a learned instance head) to kill the current bottleneck.
4. **GPU + streaming**: batch the group-by on GPU; process the LiDAR stream frame
   by frame with a rolling map.
5. **ROS2 node + live LiDAR** for on-vehicle deployment; **temporal fusion /
   tracking** across frames.

**The pitch line for novelty:** *"We don't just segment a point cloud — we build
an adaptive, foveated 2.5D world model that spends memory where safety demands it
and saves it everywhere else, and we prove the saving with real measurements."*

---

## Part 11 — How to run everything (cheat sheet)
```bash
# setup
python -m venv .venv && .venv\Scripts\activate       # Windows
pip install -r requirements.txt                       # NumPy/SciPy/sklearn/Plotly/Streamlit; no GPU needed

# full pipeline on a real frame (or synthetic)
python -m src.pipeline --input data/raw/000000.bin
python -m src.pipeline --synthetic
python -m src.pipeline --input data/raw/000000.bin --viz --json outputs/run.json

# benchmark (single file, a directory, or synthetic)
python -m src.benchmarking.benchmark --synthetic
python -m src.benchmarking.benchmark --input data/raw/

# dashboard (the demo)
streamlit run dashboard/app.py

# make a synthetic frame with no dataset
python -m src.utils.synthetic --out data/raw/synthetic_000000.bin

# tests
python -m pytest
```

---

## Part 12 — 60-second verbal walkthrough (for the pitch)
> "LiDAR gives millions of points per frame. Processing all of them in 3D is too
> slow; flattening to 2D loses the height you need to see a curb or a pothole. We
> keep a **2.5D** map — 2D footprint, height per cell — and we make it
> **foveated**, like the human eye: **5 cm cells near the car** where safety
> lives, coarsening to **50 cm at 100 m** where the LiDAR is sparse anyway. Our
> `ResolutionPolicy` buckets each point by distance into one of four zones with
> **inclusive, tested boundaries** so nothing is dropped or double-counted at a
> resolution change — that alignment problem is the classic trap and we solved it
> with one grid per zone. Each cell keeps elevation, a semantic class, and a
> confidence. On **real KITTI** this uses **~39% fewer cells and ~39% less
> memory** than a uniform 5 cm grid — measured, in a live dashboard — while both
> run real-time. The segmenter today is an explainable geometric prototype, and
> the architecture has a **drop-in slot for a PointNet++ / Sparse CNN** that needs
> zero changes to the mapping engine. Next we make foveation **object-aware** —
> refine resolution around a detected pedestrian even when it's far — which is the
> eye moving its fovea to what matters."
