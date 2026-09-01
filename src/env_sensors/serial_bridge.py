"""Serial bridge: Arduino (Grove Beginner Kit) -> JSON snapshot + CSV history.

Run standalone, alongside (not inside) the Streamlit process:

    python -m src.env_sensors.serial_bridge
    python -m src.env_sensors.serial_bridge --port COM5

Reads one JSON line per update from the Arduino sketch
(arduino/env_lidar_monitor/env_lidar_monitor.ino), derives the metrics that
need history (vibration magnitude, a pressure trend), and writes:

  - a live snapshot (overwritten each update) -> environmental_sensors.live.snapshot_path
  - an append-only history CSV                -> environmental_sensors.live.history_path

for the dashboard page to read. This script only measures and derives --
good/moderate/bad classification happens at render time in the dashboard
(src/env_sensors/classifier.py), so retuning thresholds in config.yaml never
requires restarting the bridge.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import serial  # pyserial

from src.config import load_config, resolve_path

CSV_FIELDS = [
    "timestamp_iso", "unix_t", "sound_raw", "temp_c", "humidity_pct",
    "pressure_hpa", "pressure_trend_hpa", "ax", "ay", "az",
    "vibration_g", "air_quality_raw",
]


def vibration_from_accel(ax: Optional[float], ay: Optional[float],
                          az: Optional[float]) -> Optional[float]:
    """Deviation of the acceleration magnitude from 1 g -- a simple proxy for
    wind- or mounting-induced vibration that would blur a LiDAR scan."""
    if ax is None or ay is None or az is None:
        return None
    magnitude = math.sqrt(ax * ax + ay * ay + az * az)
    return abs(magnitude - 1.0)


class PressureTrend:
    """Tracks pressure over a rolling time window to derive a drop magnitude."""

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self._history: Deque[Tuple[float, float]] = deque()

    def update(self, now_s: float, pressure_hpa: Optional[float]) -> Optional[float]:
        if pressure_hpa is not None:
            self._history.append((now_s, pressure_hpa))
        cutoff = now_s - self.window_s
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
        if len(self._history) < 2:
            return None
        oldest_pressure = self._history[0][1]
        latest_pressure = self._history[-1][1]
        return oldest_pressure - latest_pressure  # positive = pressure has been falling


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    # On Windows, os.replace() needs delete access to `path` at the instant it
    # runs -- a reader (this page's read_snapshot(), Defender, an indexer)
    # can hold a brief incompatible lock and the swap fails with WinError 5.
    # Retry through that; it's a race, not a real permissions problem.
    for attempt in range(20):
        try:
            tmp_path.replace(path)  # atomic on both Windows and POSIX
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def append_history_row(path: Path, row: Dict[str, Any], max_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        return  # nothing to trim on the file's first row

    with path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    if len(lines) > max_rows + 1:  # +1 for the header row
        header, body = lines[0], lines[1:]
        with path.open("w", encoding="utf-8") as fh:
            fh.writelines([header] + body[-max_rows:])


def run(port: str, baud_rate: int, cfg: Dict[str, Any]) -> None:
    live_cfg = cfg["environmental_sensors"]["live"]
    snapshot_path = resolve_path(live_cfg["snapshot_path"])
    history_path = resolve_path(live_cfg["history_path"])
    history_max_rows = int(live_cfg["history_max_rows"])
    trend = PressureTrend(float(live_cfg["pressure_trend_window_s"]))

    print(f"[env_sensors] opening {port} @ {baud_rate} baud ...")
    while True:
        try:
            with serial.Serial(port, baud_rate, timeout=2) as ser:
                print(f"[env_sensors] connected to {port}")
                time.sleep(2)  # let the Arduino finish its reset-on-open boot
                # No reset_input_buffer() here: it would silently discard the
                # setup()-time {"error": ...} lines (e.g. "BMP280 not found")
                # buffered during the boot wait above -- exactly the
                # diagnostics needed to catch a sensor init failure. Garbled
                # partial lines from the reset transient are already handled
                # safely by _read_loop()'s JSONDecodeError skip.
                _read_loop(ser, snapshot_path, history_path, history_max_rows, trend)
        except serial.SerialException as exc:
            print(f"[env_sensors] serial error: {exc} -- retrying in 3s", file=sys.stderr)
            time.sleep(3)
        except KeyboardInterrupt:
            print("\n[env_sensors] stopped.")
            return


def _read_loop(ser: serial.Serial, snapshot_path: Path, history_path: Path,
               history_max_rows: int, trend: PressureTrend) -> None:
    while True:
        raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not raw_line:
            continue
        try:
            reading = json.loads(raw_line)
        except json.JSONDecodeError:
            continue  # a partial line from mid-boot / serial noise -- skip it

        if "error" in reading:
            print(f"[env_sensors] Arduino reported: {reading['error']}", file=sys.stderr)
            continue

        now_s = time.time()
        pressure_hpa = reading.get("pressure_hpa")
        pressure_trend_hpa = trend.update(now_s, pressure_hpa)
        vibration_g = vibration_from_accel(
            reading.get("ax"), reading.get("ay"), reading.get("az"))

        snapshot = {
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "unix_t": now_s,
            "sound_raw": reading.get("sound"),
            "temp_c": reading.get("temp_c"),
            "humidity_pct": reading.get("hum_pct"),
            "pressure_hpa": pressure_hpa,
            "pressure_trend_hpa": pressure_trend_hpa,
            "ax": reading.get("ax"),
            "ay": reading.get("ay"),
            "az": reading.get("az"),
            "vibration_g": vibration_g,
            "air_quality_raw": reading.get("aq_raw"),
        }

        try:
            atomic_write_json(snapshot_path, snapshot)
            append_history_row(history_path, snapshot, history_max_rows)
        except OSError as exc:
            # A held file lock (Defender scan, indexer, an editor) shouldn't
            # take down the whole bridge -- drop this update and keep reading
            # serial so the next one goes through.
            print(f"[env_sensors] snapshot write failed: {exc} -- skipping this update",
                  file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=None,
                        help="Override environmental_sensors.serial.port from config.yaml")
    parser.add_argument("--baud", type=int, default=None,
                        help="Override environmental_sensors.serial.baud_rate from config.yaml")
    args = parser.parse_args()

    cfg = load_config()
    serial_cfg = cfg["environmental_sensors"]["serial"]
    port = args.port or serial_cfg["port"]
    baud_rate = args.baud or int(serial_cfg["baud_rate"])

    run(port, baud_rate, cfg)


if __name__ == "__main__":
    main()
