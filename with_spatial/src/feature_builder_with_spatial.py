from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SLOTS_PER_DAY = 96
SLOT_MINUTES = 15
EPS = 1e-12
BASELINE_CANDIDATES = ["previous_day", "mean7", "same_dow4", "same_daytype14", "similar3"]
DAYAHEAD_EXOGENOUS_COLUMNS = ["dayofweek", "is_weekend", "is_holiday", "is_public_holiday_cn", "is_spring_festival"]
STATIC_NUMERIC_COLUMNS = [
    "slow_count", "fast_count", "charge_count", "dc_power_ratio", "rated_power_sum_kw",
    "poi_1km_total", "poi_1km_business_and_residential", "poi_1km_food_and_beverage_services",
    "poi_1km_lifestyle_services", "poi_3km_total", "poi_3km_business_and_residential",
    "poi_3km_food_and_beverage_services", "poi_3km_lifestyle_services",
    "longitude", "latitude", "zone_charge_count", "zone_area",
]
PRICE_FEATURE_PREFIXES = ("target_s_price_", "target_e_price_", "target_total_price_")


@dataclass
class DayAheadClient:
    split: str
    station_id: int
    station_type: str
    x: np.ndarray
    y_residual: np.ndarray
    baseline: np.ndarray
    baseline_candidates: dict[str, np.ndarray]
    y_true: np.ndarray
    meta: pd.DataFrame

    def __len__(self) -> int:
        return int(len(self.x))


def safe_numeric(value: object, default: float = 0.0) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if pd.notna(out) else float(default)


def load_static_tables(data_root: Path) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    tables = {}
    station_types: set[str] = set()
    raw_types: set[str] = set()
    for split in ["train", "validation", "test"]:
        path = Path(data_root) / split / "station_static_features.csv"
        table = pd.read_csv(path, encoding="utf-8-sig")
        table["station_id"] = table["station_id"].astype(int)
        tables[split] = table
        if "station_type" in table.columns:
            station_types.update(table["station_type"].dropna().astype(str))
        if "raw_station_type" in table.columns:
            raw_types.update(table["raw_station_type"].dropna().astype(str))
    return tables, sorted(station_types), sorted(raw_types)


def load_daily_tables(data_root: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for split in ["train", "validation", "test"]:
        path = Path(data_root) / split / "station_daily_features.csv"
        if not path.exists():
            tables[split] = pd.DataFrame(columns=["station_id", "date"])
            continue
        table = pd.read_csv(path, encoding="utf-8-sig")
        table["station_id"] = table["station_id"].astype(int)
        table["date"] = pd.to_datetime(table["date"]).dt.strftime("%Y-%m-%d")
        keep = ["station_id", "date"] + [column for column in DAYAHEAD_EXOGENOUS_COLUMNS if column in table.columns]
        tables[split] = table[keep].copy()
    return tables


def daily_row_map(table: pd.DataFrame, station_id: int) -> dict[str, pd.Series]:
    if table.empty:
        return {}
    station_table = table[table["station_id"].astype(int).eq(int(station_id))].copy()
    return {str(row["date"]): row for _, row in station_table.iterrows()}


def station_15min_series(path: Path, split: str, static_row: pd.Series) -> pd.DataFrame:
    station_id = int(path.stem)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.sort_values("time").reset_index(drop=True)
    for column in ["volume", "duration", "busy", "idle", "fast_busy", "fast_idle", "slow_busy", "slow_idle", "s_price", "e_price"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["volume"] = frame["volume"].clip(lower=0.0)
    frame["duration"] = frame["duration"].clip(lower=0.0)
    frame["target_time"] = frame["time"].dt.floor(f"{SLOT_MINUTES}min")
    agg_map = {
        "volume": "sum",
        "duration": "sum",
        "busy": "mean",
        "idle": "mean",
        "fast_busy": "mean",
        "fast_idle": "mean",
        "slow_busy": "mean",
        "slow_idle": "mean",
        "s_price": "mean",
        "e_price": "mean",
    }
    keep_agg = {column: func for column, func in agg_map.items() if column in frame.columns}
    out = frame.groupby("target_time", as_index=False).agg(keep_agg)
    out["power_kw"] = out["volume"] * 4.0
    out["split"] = split
    out["station_id"] = station_id
    for column, value in static_row.items():
        if column != "station_id":
            out[column] = value
    out["date"] = out["target_time"].dt.date.astype(str)
    out["slot"] = out["target_time"].dt.hour * 4 + out["target_time"].dt.minute // 15
    return out.sort_values("target_time").reset_index(drop=True)


def complete_daily_profiles(series_15min: pd.DataFrame) -> list[dict[str, object]]:
    profiles: list[dict[str, object]] = []
    for date, group in series_15min.groupby("date", sort=True):
        group = group.sort_values("slot")
        if len(group) != SLOTS_PER_DAY or set(group["slot"].astype(int)) != set(range(SLOTS_PER_DAY)):
            continue
        profile = group["power_kw"].to_numpy(dtype=np.float32)
        s_price = group["s_price"].ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32) if "s_price" in group.columns else np.zeros(SLOTS_PER_DAY, dtype=np.float32)
        e_price = group["e_price"].ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32) if "e_price" in group.columns else np.zeros(SLOTS_PER_DAY, dtype=np.float32)
        target_times = pd.to_datetime(group["target_time"]).to_numpy()
        profiles.append(
            {
                "date": pd.Timestamp(date),
                "profile": profile,
                "s_price_profile": s_price,
                "e_price_profile": e_price,
                "total_price_profile": s_price + e_price,
                "target_times": target_times,
            }
        )
    return profiles


def scalar_feature_names(station_type_values: list[str], raw_type_values: list[str]) -> list[str]:
    names = [
        "target_month",
        "target_dayofweek",
        "target_is_weekend",
        "target_is_holiday",
        "target_is_public_holiday_cn",
        "target_is_spring_festival",
        "target_pre_spring_7",
        "target_post_spring_7",
        "target_spring_window_7",
        "target_is_special_day",
        "target_spring_distance_signed",
        "target_spring_distance_abs",
        "target_dayofyear_sin",
        "target_dayofyear_cos",
        "prev_total",
        "prev_mean",
        "prev_peak",
        "prev_min",
        "prev_std",
        "mean7_total",
        "mean7_mean",
        "mean7_peak",
        "mean7_min",
        "mean7_std",
        "same_dow4_total",
        "same_dow4_peak",
        "daytype14_total",
        "daytype14_peak",
        "similar3_total",
        "similar3_peak",
        "target_s_price_mean",
        "target_e_price_mean",
        "target_total_price_mean",
        "target_total_price_max",
        "history_days_used",
    ]
    names.extend([f"static_{name}" for name in STATIC_NUMERIC_COLUMNS])
    names.extend([f"dayahead_exog_{name}" for name in DAYAHEAD_EXOGENOUS_COLUMNS])
    names.extend([f"station_type_{value}" for value in station_type_values])
    names.extend([f"raw_station_type_{value}" for value in raw_type_values])
    return names


def feature_names(station_type_values: list[str], raw_type_values: list[str]) -> list[str]:
    names = [f"prev_day_slot_{slot:02d}" for slot in range(SLOTS_PER_DAY)]
    names.extend([f"mean7_slot_{slot:02d}" for slot in range(SLOTS_PER_DAY)])
    names.extend([f"same_dow4_slot_{slot:02d}" for slot in range(SLOTS_PER_DAY)])
    names.extend([f"daytype14_slot_{slot:02d}" for slot in range(SLOTS_PER_DAY)])
    names.extend([f"similar3_slot_{slot:02d}" for slot in range(SLOTS_PER_DAY)])
    names.extend([f"target_s_price_slot_{slot:02d}" for slot in range(SLOTS_PER_DAY)])
    names.extend([f"target_e_price_slot_{slot:02d}" for slot in range(SLOTS_PER_DAY)])
    names.extend(scalar_feature_names(station_type_values, raw_type_values))
    return names


def row_value(row: pd.Series | None, column: str, default: float = 0.0) -> float:
    if row is None or column not in row.index:
        return default
    return safe_numeric(row.get(column, default))


def day_type_for(date: pd.Timestamp, daily_row: pd.Series | None) -> str:
    if row_value(daily_row, "is_holiday", 0.0) > 0.5 or row_value(daily_row, "is_public_holiday_cn", 0.0) > 0.5 or row_value(daily_row, "is_spring_festival", 0.0) > 0.5:
        return "holiday"
    if int(date.dayofweek) >= 5:
        return "weekend"
    return "workday"


def spring_festival_dates(daily_by_date: dict[str, pd.Series]) -> list[pd.Timestamp]:
    dates = []
    for date_text, row in daily_by_date.items():
        if row_value(row, "is_spring_festival", 0.0) > 0.5:
            dates.append(pd.Timestamp(date_text))
    return sorted(dates)


def nearest_spring_distance(date: pd.Timestamp, spring_dates: list[pd.Timestamp]) -> int:
    if not spring_dates:
        return 999
    nearest = min(spring_dates, key=lambda value: abs((pd.Timestamp(date) - value).days))
    return int((pd.Timestamp(date) - nearest).days)


def special_day_context(date: pd.Timestamp, daily_row: pd.Series | None, spring_dates: list[pd.Timestamp]) -> dict[str, object]:
    signed_distance = nearest_spring_distance(date, spring_dates)
    abs_distance = abs(signed_distance) if signed_distance != 999 else 999
    is_holiday = float(row_value(daily_row, "is_holiday", 0.0) > 0.5)
    is_public_holiday = float(row_value(daily_row, "is_public_holiday_cn", 0.0) > 0.5)
    is_spring = float(row_value(daily_row, "is_spring_festival", 0.0) > 0.5)
    is_weekend = float(int(date.dayofweek) >= 5)
    pre_spring_7 = float(signed_distance < 0 and abs_distance <= 7)
    post_spring_7 = float(signed_distance > 0 and abs_distance <= 7)
    spring_window_7 = float(abs_distance <= 7)
    special_day = float(max(is_holiday, is_public_holiday, is_spring, spring_window_7) > 0.5)
    if is_spring > 0.5 or abs_distance <= 2:
        strategy_group = "spring_core"
    elif spring_window_7 > 0.5:
        strategy_group = "spring_adjacent"
    elif is_holiday > 0.5 or is_public_holiday > 0.5:
        strategy_group = "holiday"
    elif is_weekend > 0.5:
        strategy_group = "weekend"
    else:
        strategy_group = "workday"
    return {
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "is_public_holiday_cn": is_public_holiday,
        "is_spring_festival": is_spring,
        "spring_distance_signed": float(signed_distance),
        "spring_distance_abs": float(abs_distance),
        "pre_spring_7": pre_spring_7,
        "post_spring_7": post_spring_7,
        "spring_window_7": spring_window_7,
        "is_special_day": special_day,
        "strategy_group": strategy_group,
    }


def spring_phase_bin(context: dict[str, object]) -> str:
    signed = safe_numeric(context.get("spring_distance_signed", 999.0), 999.0)
    abs_distance = abs(signed) if signed != 999 else 999
    if safe_numeric(context.get("is_spring_festival", 0.0)) > 0.5 or abs_distance <= 2:
        return "spring_core"
    if signed < 0 and abs_distance <= 7:
        return "pre_spring_near"
    if signed > 0 and abs_distance <= 7:
        return "post_spring_near"
    if safe_numeric(context.get("is_holiday", 0.0)) > 0.5 or safe_numeric(context.get("is_public_holiday_cn", 0.0)) > 0.5:
        return "holiday_non_spring"
    return "regular"


def history_support_bin(history_days: int) -> str:
    if history_days < 3:
        return "history_lt3"
    if history_days < 7:
        return "history_3_6"
    return "history_ge7"


def volatility_bin(cv: float) -> str:
    if cv < 0.08:
        return "stable"
    if cv < 0.18:
        return "moderate"
    return "volatile"


def trend_bin(change_ratio: float) -> str:
    if change_ratio <= -0.10:
        return "falling"
    if change_ratio >= 0.10:
        return "rising"
    return "flat"


def history_reliability_context(history: list[np.ndarray]) -> dict[str, object]:
    history_days = len(history)
    if history_days <= 0:
        return {
            "recent7_total_mean": 0.0,
            "recent7_total_std": 0.0,
            "recent7_total_cv": 999.0,
            "recent7_total_change_ratio": 0.0,
            "recent7_last_to_mean_ratio": 0.0,
            "recent7_volatility_bin": "unknown",
            "recent7_trend_bin": "unknown",
            "history_support_bin": history_support_bin(0),
            "history_stability_bin": "unknown",
        }
    totals = np.asarray([float(np.asarray(profile, dtype=np.float64).sum()) for profile in history], dtype=np.float64)
    mean = float(totals.mean())
    std = float(totals.std())
    cv = float(std / (abs(mean) + EPS))
    change_ratio = float((totals[-1] - totals[0]) / (abs(mean) + EPS)) if len(totals) >= 2 else 0.0
    last_to_mean_ratio = float((totals[-1] - mean) / (abs(mean) + EPS))
    stability = volatility_bin(cv)
    return {
        "recent7_total_mean": mean,
        "recent7_total_std": std,
        "recent7_total_cv": cv,
        "recent7_total_change_ratio": change_ratio,
        "recent7_last_to_mean_ratio": last_to_mean_ratio,
        "recent7_volatility_bin": stability,
        "recent7_trend_bin": trend_bin(change_ratio),
        "history_support_bin": history_support_bin(history_days),
        "history_stability_bin": stability,
    }


def mean_profile_or_fallback(profiles: list[np.ndarray], fallback: np.ndarray) -> np.ndarray:
    if not profiles:
        return fallback.astype(np.float32)
    return np.mean(np.vstack(profiles).astype(np.float32), axis=0).astype(np.float32)


def exogenous_vector(date: pd.Timestamp, daily_row: pd.Series | None, item: dict[str, object]) -> np.ndarray:
    """Strict similarity-day exogenous vector used by the locked final experiment.

    Price and weather values are intentionally excluded from similar-day matching.
    """
    del item
    dayofweek = int(pd.Timestamp(date).dayofweek)
    return np.asarray(
        [
            np.sin(2.0 * np.pi * dayofweek / 7.0),
            np.cos(2.0 * np.pi * dayofweek / 7.0),
            float(dayofweek >= 5),
            row_value(daily_row, "is_holiday", 0.0),
            row_value(daily_row, "is_public_holiday_cn", 0.0),
            row_value(daily_row, "is_spring_festival", 0.0),
        ],
        dtype=np.float32,
    )


def similar_day_profile(
    forecast_date: pd.Timestamp,
    target_item: dict[str, object],
    history_dates: list[pd.Timestamp],
    by_date: dict[pd.Timestamp, dict[str, object]],
    daily_by_date: dict[str, pd.Series],
    fallback: np.ndarray,
    top_k: int = 3,
) -> np.ndarray:
    if not history_dates:
        return fallback.astype(np.float32)
    target_row = daily_by_date.get(forecast_date.strftime("%Y-%m-%d"))
    target_vec = exogenous_vector(forecast_date, target_row, target_item)
    candidates = []
    for history_date in history_dates:
        history_item = by_date[history_date]
        history_row = daily_by_date.get(history_date.strftime("%Y-%m-%d"))
        vec = exogenous_vector(history_date, history_row, history_item)
        distance = float(np.sqrt(np.nanmean(np.square(target_vec - vec))))
        if int(history_date.dayofweek) != int(forecast_date.dayofweek):
            distance += 0.35
        if day_type_for(history_date, history_row) != day_type_for(forecast_date, target_row):
            distance += 0.50
        candidates.append((distance, history_date))
    selected = [np.asarray(by_date[date]["profile"], dtype=np.float32) for _, date in sorted(candidates, key=lambda item: (item[0], item[1]))[:top_k]]
    return mean_profile_or_fallback(selected, fallback)


def scalar_features_for_date(
    forecast_date: pd.Timestamp,
    prev_profile: np.ndarray,
    mean7_profile: np.ndarray,
    same_dow4_profile: np.ndarray,
    daytype14_profile: np.ndarray,
    similar3_profile: np.ndarray,
    target_item: dict[str, object],
    history_days_used: int,
    static_row: pd.Series,
    daily_row: pd.Series | None,
    special_context: dict[str, object],
    station_type_values: list[str],
    raw_type_values: list[str],
) -> list[float]:
    dayofyear = int(forecast_date.dayofyear)
    station_type = str(static_row.get("station_type", ""))
    raw_type = str(static_row.get("raw_station_type", ""))
    s_price = np.asarray(target_item.get("s_price_profile", np.zeros(SLOTS_PER_DAY)), dtype=np.float32)
    e_price = np.asarray(target_item.get("e_price_profile", np.zeros(SLOTS_PER_DAY)), dtype=np.float32)
    total_price = np.asarray(target_item.get("total_price_profile", s_price + e_price), dtype=np.float32)
    values = [
        float(forecast_date.month),
        float(forecast_date.dayofweek),
        float(int(forecast_date.dayofweek >= 5)),
        row_value(daily_row, "is_holiday", 0.0),
        row_value(daily_row, "is_public_holiday_cn", 0.0),
        row_value(daily_row, "is_spring_festival", 0.0),
        safe_numeric(special_context.get("pre_spring_7", 0.0)),
        safe_numeric(special_context.get("post_spring_7", 0.0)),
        safe_numeric(special_context.get("spring_window_7", 0.0)),
        safe_numeric(special_context.get("is_special_day", 0.0)),
        safe_numeric(special_context.get("spring_distance_signed", 999.0)),
        safe_numeric(special_context.get("spring_distance_abs", 999.0)),
        math.sin(2 * math.pi * dayofyear / 366.0),
        math.cos(2 * math.pi * dayofyear / 366.0),
        float(prev_profile.sum()),
        float(prev_profile.mean()),
        float(prev_profile.max()),
        float(prev_profile.min()),
        float(prev_profile.std()),
        float(mean7_profile.sum()),
        float(mean7_profile.mean()),
        float(mean7_profile.max()),
        float(mean7_profile.min()),
        float(mean7_profile.std()),
        float(same_dow4_profile.sum()),
        float(same_dow4_profile.max()),
        float(daytype14_profile.sum()),
        float(daytype14_profile.max()),
        float(similar3_profile.sum()),
        float(similar3_profile.max()),
        float(np.nanmean(s_price)),
        float(np.nanmean(e_price)),
        float(np.nanmean(total_price)),
        float(np.nanmax(total_price)),
        float(history_days_used),
    ]
    values.extend(safe_numeric(static_row.get(name, 0.0)) for name in STATIC_NUMERIC_COLUMNS)
    values.extend(row_value(daily_row, name, 0.0) for name in DAYAHEAD_EXOGENOUS_COLUMNS)
    values.extend(1.0 if station_type == value else 0.0 for value in station_type_values)
    values.extend(1.0 if raw_type == value else 0.0 for value in raw_type_values)
    return values


def build_station_client(
    split: str,
    path: Path,
    static_table: pd.DataFrame,
    daily_table: pd.DataFrame,
    station_type_values: list[str],
    raw_type_values: list[str],
) -> tuple[DayAheadClient | None, dict[str, object]]:
    station_id = int(path.stem)
    static_match = static_table[static_table["station_id"].astype(int).eq(station_id)]
    if static_match.empty:
        raise ValueError(f"Station {station_id} missing from static table for split={split}")
    static_row = static_match.iloc[0]
    series_15min = station_15min_series(path, split, static_row)
    daily = complete_daily_profiles(series_15min)
    by_date = {item["date"]: item for item in daily}
    daily_by_date = daily_row_map(daily_table, station_id)
    spring_dates = spring_festival_dates(daily_by_date)
    dates = sorted(by_date)
    x_rows = []
    y_rows = []
    baseline_rows = []
    candidate_rows: dict[str, list[np.ndarray]] = {name: [] for name in BASELINE_CANDIDATES}
    target_rows = []
    meta_rows = []
    for forecast_date in dates:
        previous_date = forecast_date - pd.Timedelta(days=1)
        if previous_date not in by_date:
            continue
        history_dates = [day for day in dates if day < forecast_date]
        history = [by_date[day]["profile"] for day in history_dates][-7:]
        if not history:
            continue
        prev_profile = by_date[previous_date]["profile"].astype(np.float32)
        mean7_profile = np.mean(np.vstack(history).astype(np.float32), axis=0)
        same_dow_dates = [day for day in history_dates if int(day.dayofweek) == int(forecast_date.dayofweek)][-4:]
        same_dow4_profile = mean_profile_or_fallback([by_date[day]["profile"] for day in same_dow_dates], mean7_profile)
        target_daily_row = daily_by_date.get(forecast_date.strftime("%Y-%m-%d"))
        target_day_type = day_type_for(forecast_date, target_daily_row)
        target_special_context = special_day_context(forecast_date, target_daily_row, spring_dates)
        target_history_context = history_reliability_context(history)
        daytype_dates = [
            day
            for day in history_dates[-28:]
            if day_type_for(day, daily_by_date.get(day.strftime("%Y-%m-%d"))) == target_day_type
        ][-14:]
        daytype14_profile = mean_profile_or_fallback([by_date[day]["profile"] for day in daytype_dates], mean7_profile)
        target_item = by_date[forecast_date]
        similar3_profile = similar_day_profile(forecast_date, target_item, history_dates[-60:], by_date, daily_by_date, mean7_profile)
        target_profile = by_date[forecast_date]["profile"].astype(np.float32)
        x = np.concatenate(
            [
                prev_profile,
                mean7_profile.astype(np.float32),
                same_dow4_profile.astype(np.float32),
                daytype14_profile.astype(np.float32),
                similar3_profile.astype(np.float32),
                np.asarray(target_item["s_price_profile"], dtype=np.float32),
                np.asarray(target_item["e_price_profile"], dtype=np.float32),
                np.asarray(
                    scalar_features_for_date(
                        forecast_date,
                        prev_profile,
                        mean7_profile,
                        same_dow4_profile,
                        daytype14_profile,
                        similar3_profile,
                        target_item,
                        len(history),
                        static_row,
                        target_daily_row,
                        target_special_context,
                        station_type_values,
                        raw_type_values,
                    ),
                    dtype=np.float32,
                ),
            ]
        )
        x_rows.append(x)
        baseline_rows.append(prev_profile)
        candidate_rows["previous_day"].append(prev_profile)
        candidate_rows["mean7"].append(mean7_profile.astype(np.float32))
        candidate_rows["same_dow4"].append(same_dow4_profile.astype(np.float32))
        candidate_rows["same_daytype14"].append(daytype14_profile.astype(np.float32))
        candidate_rows["similar3"].append(similar3_profile.astype(np.float32))
        target_rows.append(target_profile)
        y_rows.append((target_profile - prev_profile).astype(np.float32))
        meta_rows.append(
            {
                "split": split,
                "station_id": station_id,
                "station_type": str(static_row.get("station_type", "")),
                "forecast_date": forecast_date.strftime("%Y-%m-%d"),
                "available_history_end": previous_date.strftime("%Y-%m-%d"),
                "history_days_used": len(history),
                "day_type": target_day_type,
                "strategy_group": str(target_special_context["strategy_group"]),
                "spring_phase_bin": spring_phase_bin(target_special_context),
                "is_weekend": float(target_special_context["is_weekend"]),
                "is_holiday": float(target_special_context["is_holiday"]),
                "is_public_holiday_cn": float(target_special_context["is_public_holiday_cn"]),
                "is_spring_festival": float(target_special_context["is_spring_festival"]),
                "spring_distance_signed": float(target_special_context["spring_distance_signed"]),
                "spring_distance_abs": float(target_special_context["spring_distance_abs"]),
                "pre_spring_7": float(target_special_context["pre_spring_7"]),
                "post_spring_7": float(target_special_context["post_spring_7"]),
                "spring_window_7": float(target_special_context["spring_window_7"]),
                "is_special_day": float(target_special_context["is_special_day"]),
                **target_history_context,
            }
        )
    audit = {
        "split": split,
        "station_id": station_id,
        "station_type": str(static_row.get("station_type", "")),
        "raw_5min_rows": int(len(series_15min) * 3),
        "complete_15min_days": int(len(daily)),
        "dayahead_samples": int(len(x_rows)),
        "first_sample_date": meta_rows[0]["forecast_date"] if meta_rows else "",
        "last_sample_date": meta_rows[-1]["forecast_date"] if meta_rows else "",
    }
    if not x_rows:
        return None, audit
    client = DayAheadClient(
        split=split,
        station_id=station_id,
        station_type=str(static_row.get("station_type", "")),
        x=np.vstack(x_rows).astype(np.float32),
        y_residual=np.vstack(y_rows).astype(np.float32),
        baseline=np.vstack(baseline_rows).astype(np.float32),
        baseline_candidates={name: np.vstack(rows).astype(np.float32) for name, rows in candidate_rows.items()},
        y_true=np.vstack(target_rows).astype(np.float32),
        meta=pd.DataFrame(meta_rows),
    )
    return client, audit


def _train_only_categories(static_tables: dict[str, pd.DataFrame]) -> tuple[list[str], list[str]]:
    train = static_tables["train"]

    def values(column: str) -> list[str]:
        if column not in train.columns:
            return []
        return sorted(set(train[column].dropna().astype(str)))

    return values("station_type"), values("raw_station_type")


def _load_static_tables(data_root: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        path = Path(data_root) / split / "station_static_features.csv"
        table = pd.read_csv(path, encoding="utf-8-sig")
        table["station_id"] = table["station_id"].astype(int)
        tables[split] = table
    return tables


def _load_daily_tables(data_root: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        path = Path(data_root) / split / "station_daily_features.csv"
        if not path.exists():
            tables[split] = pd.DataFrame(columns=["station_id", "date"])
            continue
        table = pd.read_csv(path, encoding="utf-8-sig")
        table["station_id"] = table["station_id"].astype(int)
        table["date"] = pd.to_datetime(table["date"]).dt.strftime("%Y-%m-%d")
        keep = ["station_id", "date", *(c for c in DAYAHEAD_EXOGENOUS_COLUMNS if c in table.columns)]
        tables[split] = table[keep].copy()
    return tables


def _filtered_feature_names(names: list[str]) -> tuple[tuple[str, ...], np.ndarray]:
    keep_mask = np.asarray(
        [not any(name.startswith(prefix) for prefix in PRICE_FEATURE_PREFIXES) for name in names],
        dtype=bool,
    )
    return tuple(name for name, keep in zip(names, keep_mask) if keep), keep_mask


def build_model_input_cache(data_root: Path) -> tuple[dict[int, dict], tuple[str, ...], pd.DataFrame]:
    """Build the exact 539-D model input cache directly from the packaged station dataset."""
    data_root = Path(data_root)
    static_tables = _load_static_tables(data_root)
    daily_tables = _load_daily_tables(data_root)
    station_type_values, raw_type_values = _train_only_categories(static_tables)
    all_names = feature_names(station_type_values, raw_type_values)
    filtered_names, keep_mask = _filtered_feature_names(all_names)

    clients: dict[int, dict] = {}
    audit_rows: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        station_dir = data_root / split / "station_processed_5min"
        for station_path in sorted(station_dir.glob("*.csv")):
            client, audit = build_station_client(
                split,
                station_path,
                static_tables[split],
                daily_tables[split],
                station_type_values,
                raw_type_values,
            )
            audit_rows.append(audit)
            if client is None:
                continue
            x = np.asarray(client.x)[:, keep_mask].astype(np.float64)
            clients[int(client.station_id)] = {
                "station_id": int(client.station_id),
                "station_type": str(client.station_type),
                "x": x,
                "y_residual": np.asarray(client.y_residual, dtype=np.float64),
                "baseline": np.asarray(client.baseline, dtype=np.float64),
                "y_true": np.asarray(client.y_true, dtype=np.float64),
                "forecast_date": client.meta["forecast_date"].astype(str).tolist(),
            }
    if len(filtered_names) != 539:
        raise RuntimeError(f"expected 539 filtered features, got {len(filtered_names)}")
    if len(clients) != 40:
        raise RuntimeError(f"expected 40 stations, got {len(clients)}")
    return clients, filtered_names, pd.DataFrame(audit_rows)


def save_model_input_cache(data_root: Path, output_path: Path, audit_path: Path | None = None) -> None:
    clients, feature_names_out, audit = build_model_input_cache(data_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump({"clients": clients, "feature_names": feature_names_out}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if audit_path is not None:
        audit_path = Path(audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(audit_path, index=False, encoding="utf-8-sig")


__all__ = [
    "build_model_input_cache",
    "save_model_input_cache",
    "feature_names",
    "scalar_feature_names",
]
