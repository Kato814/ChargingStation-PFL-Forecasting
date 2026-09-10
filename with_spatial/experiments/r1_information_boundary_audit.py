from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_revision_r1"


def main() -> None:
    schema_path = ROOT / "data/feature_schema_539.csv"
    if not schema_path.exists():
        raise FileNotFoundError(schema_path)
    schema = pd.read_csv(schema_path)
    if len(schema) != 539:
        raise AssertionError(f"expected 539 features, got {len(schema)}")

    names = schema["feature_name"].astype(str)

    # Exact frozen 539-D decomposition. These masks are deliberately explicit so
    # the reported dimensions must add to 539 and the similar-day profile cannot
    # be hidden inside a broader historical group.
    masks = {
        "previous_day_profile": names.str.startswith("prev_day_slot_"),
        "recent7_mean_profile": names.str.startswith("mean7_slot_"),
        "same_weekday_profile": names.str.startswith("same_dow4_slot_"),
        "same_daytype_profile": names.str.startswith("daytype14_slot_"),
        "similar_day_profile": names.str.startswith("similar3_slot_"),
        "target_date_calendar_special_scalars": schema["feature_index"].between(480, 493),
        "historical_load_scalar_summaries": schema["feature_index"].between(494, 510),
        "station_non_spatial_static": schema["feature_group"].eq("station_static"),
        "spatial_poi": schema["feature_group"].eq("spatial_poi"),
        "spatial_location_zone_static": schema["feature_group"].eq("spatial_static"),
        "day_ahead_calendar": schema["feature_group"].eq("calendar"),
        "station_type_indicators": schema["feature_group"].eq("station_type"),
    }

    rows = [
        ("previous_day_profile", "completed load curve of day d-1", False,
         "96-point previous-day same-time load; previous_date=d-1"),
        ("recent7_mean_profile", "loads from dates strictly before d", False,
         "mean of the most recent historical days; no day-d label"),
        ("same_weekday_profile", "loads from dates strictly before d", False,
         "same-day-of-week historical dates satisfy history_date < forecast_date"),
        ("same_daytype_profile", "loads from dates strictly before d", False,
         "same day-type historical dates satisfy history_date < forecast_date"),
        ("similar_day_profile", "historical loads before d + day-d calendar known in advance", False,
         "candidate load profiles are selected only from history_date < forecast_date; similarity uses calendar/holiday descriptors, not day-d true load"),
        ("target_date_calendar_special_scalars", "calendar/holiday descriptors known before day d", False,
         "month, weekday, weekend/holiday/Spring-Festival flags and derived calendar distance descriptors"),
        ("historical_load_scalar_summaries", "summary statistics computed from loads strictly before d", False,
         "totals/means/peaks/std of previous/recent/same-weekday/day-type/similar-day histories plus history length"),
        ("station_non_spatial_static", "static station attributes known before deployment", False,
         "equipment counts, DC-power ratio and rated power"),
        ("spatial_poi", "static POI descriptors known before deployment", False,
         "8 POI features within 1 km and 3 km"),
        ("spatial_location_zone_static", "static location/zone descriptors known before deployment", False,
         "longitude, latitude, zone charging-facility count and zone area"),
        ("day_ahead_calendar", "calendar/holiday information known before day d", False,
         "dayofweek/weekend/holiday/public-holiday/Spring-Festival flags"),
        ("station_type_indicators", "static station type known before deployment", False,
         "model/raw station-type one-hot indicators"),
    ]

    out_rows = []
    covered = pd.Series(False, index=schema.index)
    for item, availability, uses_day_d_true_load, basis in rows:
        mask = masks[item]
        overlap = covered & mask
        if overlap.any():
            raise AssertionError(f"overlapping feature masks for {item}")
        covered |= mask
        out_rows.append({
            "item": item,
            "dimension": int(mask.sum()),
            "availability": availability,
            "uses_day_d_true_load": bool(uses_day_d_true_load),
            "basis": basis,
        })

    if not covered.all():
        missing = schema.loc[~covered, ["feature_index", "feature_name"]]
        raise AssertionError(f"unclassified 539-D features:\n{missing.to_string(index=False)}")
    if sum(r["dimension"] for r in out_rows) != 539:
        raise AssertionError("feature decomposition does not sum to 539")

    # Explicitly audit feature classes that must not appear in the final 539-D cache.
    forbidden_tokens = ("target_s_price_", "target_e_price_", "target_total_price_", "weather", "airport_")
    forbidden = [n for n in names if any(token in n.lower() for token in forbidden_tokens)]
    if forbidden:
        raise AssertionError(f"unexpected target-day price/weather features in frozen 539-D schema: {forbidden[:10]}")

    model_boundary_rows = [
        {
            "item": "shared_model_scaler_and_ridge",
            "availability": "fit only from training-station samples in the training period",
            "uses_test_labels": False,
            "basis": "training sufficient statistics only; held-out validation/test stations are excluded from shared-model fitting",
        },
        {
            "item": "fold_hyperparameters",
            "availability": "selected by the held-out validation stations of each fold",
            "uses_test_labels": False,
            "basis": "test stations do not participate in ridge/type-weight/history-window/regularization selection",
        },
        {
            "item": "target_station_local_history",
            "availability": "only observations completed before each forecast day d",
            "uses_test_labels": False,
            "basis": "pre-d target-station history may be used for rolling adaptation, but day-d and future labels are excluded",
        },
    ]

    state_rows = [
        {
            "item": "alpha_jd",
            "dimension": "1 scalar / station-day",
            "availability": "estimated only from dates before d",
            "uses_day_d_true_load": False,
            "basis": "rolling scalar adaptation uses [start:index), excluding current day",
        },
        {
            "item": "gamma_d",
            "dimension": "1 scalar / deployment-group day",
            "availability": "estimated only from previous group-history days",
            "uses_day_d_true_load": False,
            "basis": "group total correction uses [start:day_index), excluding current day",
        },
        {
            "item": "risk_eligibility",
            "dimension": "1 binary state / station-day",
            "availability": "recent/cumulative losses accumulated before current prediction",
            "uses_day_d_true_load": False,
            "basis": "recent screen excludes current day; cumulative guard updates losses after decision",
        },
    ]

    feature_table = pd.DataFrame(out_rows)
    state_table = pd.DataFrame(state_rows)
    model_boundary_table = pd.DataFrame(model_boundary_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(OUT / "r1_information_boundary_features_539.csv", index=False, encoding="utf-8-sig")
    state_table.to_csv(OUT / "r1_information_boundary_states.csv", index=False, encoding="utf-8-sig")
    model_boundary_table.to_csv(OUT / "r1_information_boundary_model_protocol.csv", index=False, encoding="utf-8-sig")

    print("FEATURE INPUT AUDIT (exact 539-D decomposition)")
    print(feature_table.to_string(index=False))
    print(f"\nFEATURE DIMENSION SUM = {int(feature_table['dimension'].sum())}")
    print("TARGET-DAY MEASURED PRICE/WEATHER FEATURES = 0")
    print("\nROLLING STATE AUDIT")
    print(state_table.to_string(index=False))
    print("\nMODEL/TRAIN-VALIDATION-TEST BOUNDARY AUDIT")
    print(model_boundary_table.to_string(index=False))

    # Execute the existing perturbation test as an independent causal guard.
    subprocess.run([sys.executable, "tests/test_information_boundary.py"], cwd=ROOT, check=True)
    print("CAUSAL TEST: PASS")


if __name__ == "__main__":
    main()
