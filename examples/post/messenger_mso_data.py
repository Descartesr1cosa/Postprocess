"""Read and optionally average MESSENGER MAG MSO data from the 2008 PDS3 files.

Edit the settings below, then run this file directly.  The public function
``load_messenger_data`` is also used by the virtual-flight post-processing
module.
"""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# ============================================================
# User settings (used when this file is run directly)
# ============================================================
# Change this to the directory containing the 2008 PDS3 .LBL/.TAB files.
DATA_DIR = Path(r"E:\path\to\your\MESSENGER\2008")
OUTPUT_PATH = Path(__file__).parent / "messenger_mso_selected.csv"

# ISO 8601 UTC.  Set either value to None to use the available endpoint.
# TIME_START = "2008-01-14T17:00:00"
# TIME_STOP = "2008-01-14T18:00:00"
TIME_START = "2008-10-06T07:00:00"
TIME_STOP = "2008-10-06T10:00:00"

# None: retain individual MAG samples; positive seconds: average each window.
AVERAGE_WINDOW_SECONDS = None #30.0
# When not averaging, retain at most one raw sample every this many seconds.
SAMPLE_EVERY_SECONDS = 15.0


_START_RE = re.compile(r"^START_TIME\s*=\s*([^\s]+)", re.MULTILINE)
_STOP_RE = re.compile(r"^STOP_TIME\s*=\s*([^\s]+)", re.MULTILINE)
_COLUMNS = (
    "utc_unix_s", "year", "day_of_year", "utc_seconds_of_day",
    "x_mso_km", "y_mso_km", "z_mso_km", "bx_mso_nT", "by_mso_nT", "bz_mso_nT",
)


def _as_utc_datetime(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def available_time_range(data_dir: Path = DATA_DIR) -> tuple[datetime, datetime]:
    """Print and return the full UTC range advertised by the PDS labels."""
    starts: list[datetime] = []
    stops: list[datetime] = []
    for label in sorted(Path(data_dir).rglob("*.LBL")):
        text = label.read_text(encoding="ascii", errors="ignore")
        start, stop = _START_RE.search(text), _STOP_RE.search(text)
        if start and stop:
            starts.append(_as_utc_datetime(start.group(1)))
            stops.append(_as_utc_datetime(stop.group(1)))
    if not starts:
        raise FileNotFoundError(f"No usable .LBL files were found below {data_dir}")
    result = min(starts), max(stops)
    print("Available MESSENGER MAG MSO UTC range:")
    print("  start:", result[0].isoformat())
    print("  stop: ", result[1].isoformat())
    return result


def _labels_overlapping(data_dir: Path, start: datetime, stop: datetime) -> list[Path]:
    labels: list[Path] = []
    for label in sorted(Path(data_dir).rglob("*.LBL")):
        text = label.read_text(encoding="ascii", errors="ignore")
        first, last = _START_RE.search(text), _STOP_RE.search(text)
        if first and last and _as_utc_datetime(last.group(1)) >= start and _as_utc_datetime(first.group(1)) <= stop:
            labels.append(label)
    return labels


def _read_tab(tab_path: Path) -> np.ndarray:
    """Return raw columns year, doy, hour, minute, second, x,y,z,Bx,By,Bz."""
    return np.loadtxt(tab_path, dtype=np.float64, usecols=range(12), ndmin=2)


def _unix_seconds(raw: np.ndarray) -> np.ndarray:
    years, doys = raw[:, 0].astype(int), raw[:, 1].astype(int)
    result = raw[:, 4] + raw[:, 3] * 60.0 + raw[:, 2] * 3600.0
    for year, doy in np.unique(np.column_stack((years, doys)), axis=0):
        selected = (years == year) & (doys == doy)
        base = datetime(int(year), 1, 1, tzinfo=timezone.utc).timestamp() + (int(doy) - 1) * 86400.0
        result[selected] += base
    return result


def _sample_raw(data: np.ndarray, every_seconds: float | None) -> np.ndarray:
    if every_seconds is None or every_seconds <= 0 or data.shape[0] < 2:
        return data
    kept = np.empty(data.shape[0], dtype=bool)
    kept[0] = True
    last = data[0, 0]
    for index in range(1, data.shape[0]):
        kept[index] = data[index, 0] - last >= every_seconds
        if kept[index]:
            last = data[index, 0]
    return data[kept]


def _average(data: np.ndarray, start_unix_s: float, window_seconds: float) -> np.ndarray:
    if window_seconds <= 0:
        raise ValueError("average_window_seconds must be positive or None")
    bins = np.floor((data[:, 0] - start_unix_s) / window_seconds).astype(np.int64)
    unique_bins, inverse = np.unique(bins, return_inverse=True)
    counts = np.bincount(inverse)
    result = np.empty((unique_bins.size, data.shape[1]), dtype=np.float64)
    for column in range(data.shape[1]):
        result[:, column] = np.bincount(inverse, weights=data[:, column]) / counts
    return result


def load_messenger_data(
    time_start: str | datetime | None,
    time_stop: str | datetime | None,
    *,
    data_dir: Path = DATA_DIR,
    average_window_seconds: float | None = None,
    sample_every_seconds: float | None = None,
) -> np.ndarray:
    """Load selected MAG samples as columns listed in ``COLUMNS``.

    Positions are MSO km and field components are MSO nT.  Time is UTC Unix
    seconds, so it is easy to compare, average, and convert back to ISO UTC.
    """
    available_start, available_stop = available_time_range(data_dir)
    start = _as_utc_datetime(time_start) or available_start
    stop = _as_utc_datetime(time_stop) or available_stop
    if stop < start:
        raise ValueError("time_stop must not precede time_start")
    labels = _labels_overlapping(data_dir, start, stop)
    if not labels:
        raise ValueError("The requested interval does not overlap any MAG file")

    start_s, stop_s = start.timestamp(), stop.timestamp()
    selected_parts: list[np.ndarray] = []
    for label in labels:
        raw = _read_tab(label.with_suffix(".TAB"))
        unix_s = _unix_seconds(raw)
        keep = (unix_s >= start_s) & (unix_s <= stop_s)
        if not np.any(keep):
            continue
        raw, unix_s = raw[keep], unix_s[keep]
        seconds_of_day = raw[:, 2] * 3600.0 + raw[:, 3] * 60.0 + raw[:, 4]
        selected_parts.append(np.column_stack((
            unix_s, raw[:, 0], raw[:, 1], seconds_of_day, raw[:, 6:12],
        )))
    if not selected_parts:
        raise ValueError("No MAG samples were found in the requested interval")
    data = np.vstack(selected_parts)
    data = data[np.argsort(data[:, 0])]
    return _average(data, start_s, average_window_seconds) if average_window_seconds is not None else _sample_raw(data, sample_every_seconds)


def write_csv(data: np.ndarray, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(("utc_iso",) + _COLUMNS)
        for row in data:
            iso = datetime.fromtimestamp(row[0], tz=timezone.utc).isoformat(timespec="milliseconds")
            writer.writerow((iso,) + tuple(f"{value:.9g}" for value in row))
    return output_path


if __name__ == "__main__":
    samples = load_messenger_data(
        TIME_START, TIME_STOP, data_dir=DATA_DIR,
        average_window_seconds=AVERAGE_WINDOW_SECONDS,
        sample_every_seconds=SAMPLE_EVERY_SECONDS,
    )
    written = write_csv(samples, OUTPUT_PATH)
    print(f"Written {samples.shape[0]} rows: {written}")
