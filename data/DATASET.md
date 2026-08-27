# Datasets

## Real data: KITTI raw

The pipeline consumes **KITTI Velodyne `.bin`** frames (`float32 [N,4]` =
`x, y, z, intensity`).

### Source used for real-data testing

- **KITTI raw, drive `2011_09_26_drive_0001`** (synced), downloaded from the
  official public KITTI S3 bucket:
  `https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip`
- Velodyne scans live inside the zip at
  `2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/*.bin`.
- Extracted frames are placed in `data/raw/` (gitignored).

> Note: the `small_gicp` BENCHMARK.md referenced a `KITTI00.tar.gz` subset on
> Google Drive; that link was quota/permission-blocked for programmatic
> download, so we used the official KITTI S3 mirror instead. Both are the same
> underlying KITTI Velodyne `.bin` format.

### License

The KITTI dataset is distributed under **CC BY-NC-SA 3.0** (Attribution,
NonCommercial, ShareAlike). Use for this project is **academic / research /
SIH demonstration only** — not for commercial purposes. Cite:

> A. Geiger, P. Lenz, C. Stiller, R. Urtasun. *Vision meets Robotics: The KITTI
> Dataset.* International Journal of Robotics Research (IJRR), 2013.

## Synthetic data

`src/utils/synthetic.py` generates a deterministic KITTI-format frame (dense
near-field, sparse far-field, with ground / vehicles / pedestrians / poles) so
the pipeline runs and is testable with no download. Clearly labeled as
synthetic wherever produced.
