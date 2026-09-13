"""Versioned reconstruction of the local 60-day ridge baseline.

This does not claim exact recovery of the historical summary-only experiment.
It writes new evidence only and never patches V1.0 reference numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Small dense ridge systems do not benefit from unrestricted BLAS threading.
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import cho_factor, cho_solve

from baselines.reference import load_locked_comparison_predictions
from src.config import build_folds, load_config
from src.data_preparation import load_model_input_cache
from src.evaluation import (aggregate_system_metrics, deployment_group_daily_wape,
                            paired_daily_inference, station_wape_table, type_aggregate_wape_table)
from src.group_total_correction import GroupCorrectionConfig
from src.negative_transfer_control import NegativeTransferConfig
from experiments.r1_postprocessing_fairness import _attach_stage_columns, _stage_summary

KEYS = ['fold', 'station_id', 'type', 'prediction_date', 'slot']
NUMERIC = ['y_true', 'previous_day_baseline', 'prediction']
PROTOCOL_PATH = Path(__file__).with_name('r1_local_ridge_rebuild_protocol.json')


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha(path: Path) -> str:
    text = path.read_text(encoding='utf-8-sig').replace('\r\n', '\n').replace('\r', '\n')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(['git', *args], cwd=ROOT, text=True, capture_output=True,
                          encoding='utf-8', errors='replace', check=check, timeout=60)


def verify_environment(protocol: dict) -> dict:
    head = git('rev-parse', 'HEAD').stdout.strip()
    branch = git('branch', '--show-current').stdout.strip()
    if branch != protocol['branch']:
        raise RuntimeError(f'Wrong branch: {branch}')
    if git('merge-base', '--is-ancestor', protocol['parent_result_commit'], 'HEAD', check=False).returncode != 0:
        raise RuntimeError('Current HEAD is not descended from the result lock.')
    manifest = json.loads((ROOT / protocol['payload_manifest']).read_text(encoding='utf-8'))
    for rel, expected in manifest['canonical_text_sha256'].items():
        if canonical_sha(ROOT / rel) != expected:
            raise RuntimeError(f'Protocol payload changed: {rel}')
    watched = list(manifest['canonical_text_sha256']) + [protocol['payload_manifest']]
    for rel in watched:
        git('ls-files', '--error-unmatch', '--', rel)
    if git('status', '--porcelain', '--', *watched).stdout.strip():
        raise RuntimeError('Protocol payload must be committed and clean before real-data execution.')
    for rel, expected in protocol['dependency_canonical_sha256'].items():
        if canonical_sha(ROOT / rel) != expected:
            raise RuntimeError(f'Frozen dependency differs from the audited source: {rel}')
    files = {}
    for role, item in protocol['inputs'].items():
        path = ROOT / item['path']
        actual = sha_file(path)
        if actual != item['sha256']:
            raise RuntimeError(f'Frozen input SHA mismatch: {role}')
        files[role] = {'path': str(path), 'sha256': actual, 'size_bytes': path.stat().st_size}
    return {'git_commit': head, 'branch': branch, 'python': sys.version,
            'python_executable': sys.executable, 'platform': platform.platform(),
            'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
            'input_files': files, 'protocol_sha256': canonical_sha(PROTOCOL_PATH),
            'script_sha256': canonical_sha(Path(__file__)), 'dependency_check': 'PASS'}


def ridge_prediction(x: np.ndarray, residual: np.ndarray, query: np.ndarray,
                     ridge_alpha: float) -> tuple[np.ndarray, dict]:
    """Minimize ||R - ZW - b||_F^2 + alpha ||W||_F^2; b is unpenalized.

    Solve the equivalent n-by-n dual system; no target/test normalization.
    """
    x = np.asarray(x, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if x.ndim != 2 or residual.ndim != 2 or query.ndim != 2:
        raise ValueError('All ridge arrays must be two-dimensional.')
    if len(x) < 2 or len(x) != len(residual) or query.shape[1] != x.shape[1]:
        raise ValueError('Ridge shapes do not align.')
    if not np.isfinite(ridge_alpha) or ridge_alpha <= 0:
        raise ValueError('Ridge alpha must be finite and positive.')
    if not all(np.isfinite(a).all() for a in (x, residual, query)):
        raise ValueError('Non-finite ridge input.')
    mean = x.mean(axis=0)
    centered = x - mean
    # Exact constant columns are explicitly zeroed, including static POI/location.
    constant = np.ptp(x, axis=0) == 0.0
    centered[:, constant] = 0.0
    scale = np.sqrt(np.mean(centered * centered, axis=0))
    constant |= scale < 1e-12
    scale[constant] = 1.0
    z = centered / scale
    z[:, constant] = 0.0
    zq = (query - mean) / scale
    rmean = residual.mean(axis=0)
    rc = residual - rmean
    gram = z @ z.T
    gram.flat[::len(x) + 1] += float(ridge_alpha)
    dual = cho_solve(cho_factor(gram, lower=True, check_finite=True), rc, check_finite=True)
    prediction = (zq @ z.T) @ dual + rmean
    if not np.isfinite(prediction).all():
        raise FloatingPointError('Non-finite ridge prediction.')
    return prediction, {'constant_feature_count': int(constant.sum()),
                        'effective_feature_count': int((~constant).sum())}


def prepare_client(raw: dict) -> dict:
    dates = pd.DatetimeIndex(pd.to_datetime(raw['forecast_date'])).normalize()
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError(f'Unsorted or duplicated dates at station {raw["station_id"]}')
    return {'station_id': int(raw['station_id']), 'type': str(raw['station_type']),
            'dates': dates, 'x': np.asarray(raw['x'], dtype=np.float64),
            'residual': np.asarray(raw['y_residual'], dtype=np.float64),
            'baseline': np.asarray(raw['baseline'], dtype=np.float64),
            'y_true': np.asarray(raw['y_true'], dtype=np.float64)}


def predict_one_day(client: dict, date: pd.Timestamp, alpha: float,
                    history_days: int = 60) -> tuple[np.ndarray, dict]:
    date = pd.Timestamp(date).normalize()
    dates = client['dates']
    pos = dates.get_indexer([date])[0]
    if pos < 0:
        raise ValueError(f'Missing prediction day: {client["station_id"]} {date.date()}')
    wanted = pd.date_range(date - pd.Timedelta(days=history_days), periods=history_days, freq='D')
    idx = dates.get_indexer(wanted)
    if (idx < 0).any():
        raise ValueError(f'Incomplete {history_days}-day history: {client["station_id"]} {date.date()}')
    if not (dates[idx] < date).all():
        raise AssertionError('History contains current or future samples.')
    res, info = ridge_prediction(client['x'][idx], client['residual'][idx], client['x'][[pos]], alpha)
    prediction = np.maximum(0.0, client['baseline'][pos] + res[0])
    info.update({'station_id': client['station_id'], 'prediction_date': date.strftime('%Y-%m-%d'),
                 'history_first': wanted[0].strftime('%Y-%m-%d'),
                 'history_last': wanted[-1].strftime('%Y-%m-%d'), 'history_days': history_days,
                 'ridge_alpha': float(alpha), 'feature_dimension': int(client['x'].shape[1]),
                 'output_dimension': int(client['residual'].shape[1]),
                 'max_history_date_before_forecast': bool(wanted[-1] < date)})
    return prediction, info


def predict_pair(clients: dict, station_ids, fold: str, start: str, end: str,
                 alpha: float, split: str) -> tuple[pd.DataFrame, list[dict]]:
    pieces, audits = [], []
    for sid in sorted(station_ids):
        c = clients[int(sid)]
        for d in pd.date_range(start, end, freq='D'):
            p, audit = predict_one_day(c, d, alpha)
            audit.update({'fold': fold, 'split': split})
            audits.append(audit)
            i = c['dates'].get_indexer([d])[0]
            n = len(p)
            pieces.append(pd.DataFrame({'fold': fold, 'station_id': int(sid), 'type': c['type'],
                         'prediction_date': d, 'slot': np.arange(n), 'y_true': c['y_true'][i],
                         'previous_day_baseline': c['baseline'][i], 'prediction': p}))
    result = pd.concat(pieces, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    return result, audits


def validate_frame(frame: pd.DataFrame, expected_dates, fold_station_pairs: dict) -> None:
    missing = set(KEYS + NUMERIC) - set(frame.columns)
    if missing:
        raise ValueError(f'OOF missing columns: {sorted(missing)}')
    if frame.duplicated(KEYS).any():
        raise ValueError('OOF contains duplicate keys.')
    if not np.isfinite(frame[NUMERIC].to_numpy(dtype=float)).all():
        raise ValueError('OOF contains non-finite numerical entries.')
    if (frame['prediction'] < 0).any():
        raise ValueError('Negative load prediction after clipping.')
    ds = list(pd.DatetimeIndex(expected_dates).normalize())
    expected_index = pd.MultiIndex.from_tuples(
        [(fold, int(sid), d, slot) for fold, sids in fold_station_pairs.items()
         for sid in sids for d in ds for slot in range(96)],
        names=['fold', 'station_id', 'prediction_date', 'slot'])
    actual_index = pd.MultiIndex.from_frame(frame[expected_index.names])
    if len(actual_index) != len(expected_index) or not actual_index.is_unique:
        raise ValueError('OOF record count or station/date/slot uniqueness failed.')
    if len(expected_index.difference(actual_index)) or len(actual_index.difference(expected_index)):
        raise ValueError('OOF does not cover the exact locked deployment grid.')


def align_with_reference(frame: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if reference.duplicated(KEYS).any():
        raise ValueError('Locked reference contains duplicate keys.')
    merged = frame.merge(reference[KEYS + ['y_true', 'previous_day_baseline', 'proposed_prediction']],
                         how='outer', on=KEYS, suffixes=('_new', '_ref'),
                         indicator=True, validate='one_to_one')
    if len(merged) != len(frame) or not merged['_merge'].eq('both').all():
        raise ValueError('Rebuilt/reference OOF key mismatch.')
    diffs = {}
    for col in ['y_true', 'previous_day_baseline']:
        diffs[col] = float(np.abs(merged[col + '_new'] - merged[col + '_ref']).max())
        if not np.isfinite(diffs[col]) or diffs[col] > 1e-9:
            raise ValueError(f'Locked {col} values mismatch: {diffs[col]}')
    aligned = merged[KEYS + ['y_true_ref', 'previous_day_baseline_ref', 'prediction',
                             'proposed_prediction']].rename(columns={'y_true_ref': 'y_true',
                                                        'previous_day_baseline_ref': 'previous_day_baseline'})
    return aligned.sort_values(KEYS).reset_index(drop=True), diffs


def collect_station(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    a = station_wape_table(frame, 'previous_day_baseline').rename(columns={'WAPE': 'baseline_WAPE'})
    p = station_wape_table(frame, column).rename(columns={'WAPE': 'method_WAPE'})
    out = a.merge(p, on=['station_id', 'type'], validate='one_to_one')
    out['relative_change'] = out['method_WAPE'] / out['baseline_WAPE'] - 1.0
    return out


def daily_evidence(frame: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    items = {name: deployment_group_daily_wape(frame, col) for name, col in columns.items()}
    result = pd.DataFrame(items).sort_index()
    if len(result) != 44 or result.isna().any().any():
        raise ValueError('Daily evidence must contain all 44 prediction days.')
    for name in ('raw', 'group', 'group_risk'):
        result['proposed_minus_' + name] = result['proposed'] - result[name]
    result.index.name = 'prediction_date'
    return result.reset_index()


def run_experiment(protocol: dict, output_dir: Path, meta: dict) -> None:
    paper = load_config(ROOT / 'paper_config.json')
    if paper['baseline_protocols']['local_ridge_history_days'] != 60:
        raise AssertionError('Frozen local-ridge history is not 60 days.')
    folds = build_folds(paper)
    if [list(f.test) for f in folds] != protocol['test_pairs']:
        raise AssertionError('Fold test pairs differ from the protocol.')
    raw_clients, names = load_model_input_cache(ROOT / protocol['inputs']['clients_cache']['path'])
    if len(raw_clients) != 40 or len(names) != 539:
        raise AssertionError('Frozen input contract mismatch.')
    target_ids = {int(s) for f in folds for s in f.test}
    clients = {sid: prepare_client(raw_clients[sid]) for sid in target_ids}
    del raw_clients
    audits, grids, val_frames, selected = [], [], [], {}
    # Complete all validation selection before evaluating any test prediction.
    for f in folds:
        scores = []
        for alpha in protocol['ridge_alpha_grid']:
            frame, audit = predict_pair(clients, f.validation, f.fold, *protocol['validation_period'],
                                        alpha, 'validation')
            validate_frame(frame, pd.date_range(*protocol['validation_period']), {f.fold: f.validation})
            metric = aggregate_system_metrics(frame, 'prediction')
            row = {'fold': f.fold, 'ridge_alpha': alpha, 'validation_WAPE': metric['WAPE'],
                   'validation_nRMSE': metric['nRMSE'], 'record_count': len(frame)}
            grids.append(row); scores.append(row); audits.extend(audit)
            val_frames.append(frame.assign(ridge_alpha=alpha))
        choice = min(scores, key=lambda r: (r['validation_WAPE'], r['ridge_alpha']))
        selected[f.fold] = choice['ridge_alpha']
        print(f'[validation] {f.fold} selected lambda={choice["ridge_alpha"]:g}', flush=True)
    grid_df = pd.DataFrame(grids)
    grid_df.to_csv(output_dir / 'local_ridge_validation_grid.csv', index=False, encoding='utf-8-sig')
    dump_json(output_dir / 'local_ridge_selected_alpha.json', selected)
    pd.concat(val_frames, ignore_index=True).to_csv(output_dir / 'local_ridge_validation_oof.csv.gz',
                                                   index=False, compression='gzip')
    test_frames = []
    for f in folds:
        frame, audit = predict_pair(clients, f.test, f.fold, *protocol['test_period'],
                                    selected[f.fold], 'test')
        test_frames.append(frame); audits.extend(audit)
        print(f'[test] {f.fold} records={len(frame)}', flush=True)
    oof = pd.concat(test_frames, ignore_index=True)
    validate_frame(oof, pd.date_range(*protocol['test_period']), {f.fold: f.test for f in folds})
    reference = load_locked_comparison_predictions(ROOT / protocol['inputs']['locked_oof']['path'])
    oof, alignment = align_with_reference(oof, reference)
    # These are integrity checks on existing evidence, not tuning targets for ridge.
    ref_metrics = aggregate_system_metrics(reference, 'proposed_prediction')
    ref_inf = paired_daily_inference(reference)
    if not np.isclose(ref_metrics['WAPE'], protocol['reference_integrity']['proposed_WAPE'], rtol=0, atol=1e-12):
        raise AssertionError('Existing proposed reference WAPE does not reproduce.')
    if not np.isclose(ref_inf['mean_daily_WAPE_difference'], protocol['reference_integrity']['proposed_minus_persistence_mean_daily_WAPE'], rtol=0, atol=1e-12):
        raise AssertionError('Existing deployment-group daily metric does not reproduce.')
    oof.to_csv(output_dir / 'local_ridge_oof.csv.gz', index=False, compression='gzip')
    pd.DataFrame(audits).to_csv(output_dir / 'local_ridge_window_audit.csv', index=False, encoding='utf-8-sig')
    metrics = aggregate_system_metrics(oof, 'prediction')
    inference = paired_daily_inference(oof, baseline_column='prediction')
    station = collect_station(oof, 'prediction')
    summary = {'method': 'local_ridge_rebuild_v1', 'provenance': 'versioned_reconstruction_not_exact_historical_recovery',
               'record_count': len(oof), **metrics, 'improved_stations': int((station['relative_change'] < 0).sum()),
               'max_station_relative_degradation': float(max(0., station['relative_change'].max())),
               'paired_daily_proposed_minus_method': inference}
    dump_json(output_dir / 'local_ridge_summary.json', summary)
    dump_json(output_dir / 'local_ridge_daily_inference.json', inference)
    station.to_csv(output_dir / 'local_ridge_station.csv', index=False, encoding='utf-8-sig')
    type_aggregate_wape_table(oof, 'prediction', 'previous_day_baseline').to_csv(
        output_dir / 'local_ridge_type.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([{'method': '本地岭回归', 'WAPE': metrics['WAPE'], 'nRMSE': metrics['nRMSE'],
                   'p_NW': inference['newey_west_p_two_sided'], 'source': 'local_ridge_rebuild_v1'}]).to_csv(
        output_dir / 'table2_local_ridge_row.csv', index=False, encoding='utf-8-sig')
    comparison = {'historical_summary_only': {'WAPE': 0.1025, 'nRMSE': 0.0727},
                  'rebuilt': {'WAPE': metrics['WAPE'], 'nRMSE': metrics['nRMSE']},
                  'difference_rebuilt_minus_historical': {k: metrics[k] - v for k, v in [('WAPE', .1025), ('nRMSE', .0727)]},
                  'printed_four_decimals_match': {k: f'{metrics[k]:.4f}' == f'{v:.4f}' for k, v in [('WAPE', .1025), ('nRMSE', .0727)]},
                  'note': 'Comparison for provenance only. Matching rounded summaries does not prove identical historical predictions. Never mix old errors and a new p value.'}
    dump_json(output_dir / 'local_ridge_historical_comparison.json', comparison)
    enriched = _attach_stage_columns(oof, 'ridge_rebuild', 'prediction',
                   GroupCorrectionConfig(**paper['group_total_correction']),
                   NegativeTransferConfig(**paper['negative_transfer']))
    stages = {'raw': 'prediction', '+group': 'r1_ridge_rebuild_group',
              '+group+risk': 'r1_ridge_rebuild_group_risk'}
    fairness_rows = [_stage_summary(enriched, '本地岭回归', col, stage) for stage, col in stages.items()]
    pd.DataFrame(fairness_rows).to_csv(output_dir / 'local_ridge_fairness_summary.csv', index=False, encoding='utf-8-sig')
    fair_inf = {stage: paired_daily_inference(enriched, baseline_column=col) for stage, col in stages.items()}
    dump_json(output_dir / 'local_ridge_fairness_daily_inference.json', fair_inf)
    enriched.to_csv(output_dir / 'local_ridge_fairness_oof.csv.gz', index=False, compression='gzip')
    daily_evidence(enriched, {'proposed': 'proposed_prediction', 'baseline': 'previous_day_baseline',
                   'raw': 'prediction', 'group': stages['+group'], 'group_risk': stages['+group+risk']}).to_csv(
        output_dir / 'local_ridge_daily_wape.csv', index=False, encoding='utf-8-sig')
    numeric_evidence = [metrics['WAPE'], metrics['nRMSE'], inference['newey_west_p_two_sided']]
    numeric_evidence += [v['newey_west_p_two_sided'] for v in fair_inf.values()]
    if not np.isfinite(numeric_evidence).all():
        raise FloatingPointError('A final metric or inference value is non-finite.')
    meta.update({'status': 'COMPLETED', 'completed_at_utc': utc_now(), 'record_count': len(oof),
                 'alignment_max_absolute_difference': alignment, 'selected_alpha': selected,
                 'validation_only_selection': True, 'history_window_check': 'PASS',
                 'history_model_fit_count': len(audits), 'reference_integrity_check': 'PASS',
                 'fairness_implementation': 'reused existing experiments/r1_postprocessing_fairness.py functions',
                 'fairness_parameters': {k: paper[k] for k in ['group_total_correction', 'negative_transfer']},
                 'scientific_acceptance_does_not_require_significance_or_matching_old_numbers': True})
    meta['outputs'] = {p.name: {'size_bytes': p.stat().st_size, 'sha256': sha_file(p)}
                        for p in sorted(output_dir.iterdir()) if p.is_file() and p.name != 'run_metadata.json'}
    dump_json(output_dir / 'run_metadata.json', meta)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorize-reconstruction', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.authorize_reconstruction:
        parser.error('Use --authorize-reconstruction only after the protocol is committed.')
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))
    parent = (ROOT / 'results_revision_r1' / 'local_ridge_rebuild_v1').resolve()
    output = args.output_dir.resolve()
    if output.parent != parent:
        raise RuntimeError(f'Output must be a fresh run directory immediately below {parent}')
    output.mkdir(parents=True, exist_ok=False)
    meta = {'protocol_id': protocol['protocol_id'], 'status': 'PREFLIGHT',
            'started_at_utc': utc_now(), 'output_dir': str(output)}
    try:
        meta.update(verify_environment(protocol))
        meta['status'] = 'RUNNING'
        dump_json(output / 'run_metadata.json', meta)
        run_experiment(protocol, output, meta)
        end_environment = verify_environment(protocol)
        if end_environment['git_commit'] != meta['git_commit']:
            raise RuntimeError('Git commit changed during reconstruction.')
        meta['postrun_integrity_check'] = 'PASS'
        dump_json(output / 'run_metadata.json', meta)
        print(f'RECONSTRUCTION_COMPLETED={output}', flush=True)
    except BaseException as exc:
        import traceback
        meta.update({'status': 'FAILED', 'completed_at_utc': utc_now(),
                     'error': repr(exc), 'traceback': traceback.format_exc()})
        dump_json(output / 'run_metadata.json', meta)
        raise


if __name__ == '__main__':
    main()
