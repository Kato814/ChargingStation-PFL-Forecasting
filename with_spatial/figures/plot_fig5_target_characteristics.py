from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_style import CN, EN, apply_tick_fonts

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data/source_station_dataset"

STATIONS = [
    ("F01", 1011, "pure_fast"),
    ("F02", 1009, "pure_fast"),
    ("S01", 2213, "pure_slow"),
    ("S02", 1686, "pure_slow"),
    ("S03", 2255, "pure_slow"),
    ("S04", 1120, "pure_slow"),
    ("S05", 1566, "pure_slow"),
    ("S06", 1307, "pure_slow"),
    ("M01", 1599, "mixed_ac_dc"),
    ("M02", 1349, "mixed_ac_dc"),
]

TYPE_LABEL = {
    "pure_fast": "快充站",
    "pure_slow": "慢充站",
    "mixed_ac_dc": "混合站",
}
TYPE_COLOR = {
    "pure_fast": "#d87a1c",
    "pure_slow": "#3b70b7",
    "mixed_ac_dc": "#3b8f6b",
}
TYPE_STYLE = {
    "pure_fast": "-",
    "pure_slow": "--",
    "mixed_ac_dc": ":",
}


def _load_split() -> dict[int, str]:
    table = pd.read_csv(SOURCE_ROOT / "station_split.csv")
    return {int(row.station_id): str(row.split) for row in table.itertuples(index=False)}


def _daily_energy(split_map: dict[int, str]) -> pd.DataFrame:
    frames = []
    for split in ["train", "validation", "test"]:
        path = SOURCE_ROOT / split / "station_daily_features.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("station_daily_features.csv files are missing; restore the frozen source dataset first")
    daily = pd.concat(frames, ignore_index=True)

    rows = []
    for label, station_id, station_type in STATIONS:
        g = daily[daily["station_id"] == station_id]
        if len(g) != 181:
            raise ValueError(f"expected 181 daily records for station {station_id}, got {len(g)}")
        rows.append({
            "target_station": label,
            "station_id": station_id,
            "station_type": station_type,
            "mean_daily_energy_kwh": float(g["daily_total_load"].mean()),
        })
    return pd.DataFrame(rows)


def _station_normalized_profile(station_id: int, split: str) -> np.ndarray:
    path = SOURCE_ROOT / split / "station_processed_5min" / f"{station_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing restored 5-min source file: {path}")
    data = pd.read_csv(path, usecols=["time", "volume"])
    data["time"] = pd.to_datetime(data["time"])
    data["date"] = data["time"].dt.date
    data["slot15"] = data["time"].dt.hour * 4 + data["time"].dt.minute // 15

    # Three 5-min points form one 15-min interval. A mean is sufficient here because
    # only the normalized intraday shape is used; sum and mean differ by a constant factor.
    q15 = data.groupby(["date", "slot15"], as_index=False)["volume"].mean()
    profile = q15.groupby("slot15")["volume"].mean().reindex(range(96)).to_numpy(dtype=float)
    if np.isnan(profile).any():
        raise ValueError(f"incomplete 96-point mean profile for station {station_id}")
    peak = float(np.max(profile))
    if peak <= 0:
        raise ValueError(f"non-positive profile peak for station {station_id}")
    return profile / peak


def _type_profiles(split_map: dict[int, str]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    station_profiles: dict[int, np.ndarray] = {}
    for _, station_id, _ in STATIONS:
        station_profiles[station_id] = _station_normalized_profile(station_id, split_map[station_id])

    rows = []
    arrays = {}
    for station_type in ["pure_fast", "pure_slow", "mixed_ac_dc"]:
        ids = [sid for _, sid, tp in STATIONS if tp == station_type]
        stack = np.stack([station_profiles[sid] for sid in ids])
        mean = stack.mean(axis=0)
        low = stack.min(axis=0)
        high = stack.max(axis=0)
        arrays[station_type] = stack
        for slot in range(96):
            rows.append({
                "station_type": station_type,
                "time_h": slot / 4.0,
                "mean_normalized_load": float(mean[slot]),
                "min_normalized_load": float(low[slot]),
                "max_normalized_load": float(high[slot]),
            })
    return pd.DataFrame(rows), arrays


def plot_daily_energy(table: pd.DataFrame, output: Path, svg_output: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.15), dpi=300)
    colors = [TYPE_COLOR[v] for v in table["station_type"]]
    ax.bar(table["target_station"], table["mean_daily_energy_kwh"], color=colors, width=0.58)
    ax.set_yscale("log")
    ax.set_ylim(100, 10000)
    ax.set_yticks([100, 300, 1000, 3000, 10000])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True, length=3.0)
    apply_tick_fonts(ax, 7.6)
    ax.set_xlabel("站点", fontsize=8.8, fontproperties=CN)
    ax.set_ylabel("日均充电量 / kWh", fontsize=8.8, fontproperties=CN)
    for label in ax.get_xticklabels():
        label.set_fontproperties(EN)
        label.set_fontsize(7.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_type_profiles(table: pd.DataFrame, output: Path, svg_output: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.15), dpi=300)
    for station_type in ["pure_fast", "pure_slow", "mixed_ac_dc"]:
        g = table[table["station_type"] == station_type]
        x = g["time_h"].to_numpy()
        mean = g["mean_normalized_load"].to_numpy()
        low = g["min_normalized_load"].to_numpy()
        high = g["max_normalized_load"].to_numpy()
        color = TYPE_COLOR[station_type]
        ax.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)
        ax.plot(x, mean, color=color, linestyle=TYPE_STYLE[station_type], linewidth=1.15, label=TYPE_LABEL[station_type])
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True, length=3.0)
    apply_tick_fonts(ax, 7.6)
    ax.set_xlabel("时刻 / h", fontsize=8.8, fontproperties=CN)
    ax.set_ylabel("归一化负荷", fontsize=8.8, fontproperties=CN)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.19), ncol=3, frameon=False, prop=CN, fontsize=7.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paper Fig. 5 target-station load scales and normalized intraday profiles.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures_generated")
    args = parser.parse_args()

    split_map = _load_split()
    energy = _daily_energy(split_map)
    profiles, _ = _type_profiles(split_map)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    energy.to_csv(args.output_dir / "fig5a_station_daily_energy_data.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(args.output_dir / "fig5b_type_profiles_data.csv", index=False, encoding="utf-8-sig")

    plot_daily_energy(
        energy,
        args.output_dir / "fig5a_station_daily_energy.png",
        args.output_dir / "fig5a_station_daily_energy.svg",
    )
    plot_type_profiles(
        profiles,
        args.output_dir / "fig5b_type_profiles.png",
        args.output_dir / "fig5b_type_profiles.svg",
    )

    print(args.output_dir / "fig5a_station_daily_energy.png")
    print(args.output_dir / "fig5b_type_profiles.png")


if __name__ == "__main__":
    main()
