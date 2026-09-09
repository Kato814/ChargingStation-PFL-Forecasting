from __future__ import annotations

import argparse
import copy
import json
import pickle
import platform
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn

from baselines.protocols import NEURAL_BASELINE_SEEDS, RESIDUAL_MLP_PROTOCOL
from baselines.reference import load_locked_comparison_predictions
from src.config import build_folds, load_config
from src.evaluation import aggregate_system_metrics, paired_daily_inference, station_wape_table, type_aggregate_wape_table

TRAIN_START, TRAIN_END = pd.Timestamp("2022-09-02"), pd.Timestamp("2022-12-15")
VAL_START, VAL_END = pd.Timestamp("2022-12-16"), pd.Timestamp("2023-01-15")
TEST_START, TEST_END = pd.Timestamp("2023-01-16"), pd.Timestamp("2023-02-28")
FEDPROX_MU_GRID = (1e-3, 1e-2, 1e-1, 1.0)
SHARED_PREFIXES = ("fc1.", "ln1.", "fc2.")
PERSONAL_PREFIXES = ("out.",)


class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        p = RESIDUAL_MLP_PROTOCOL
        self.fc1 = nn.Linear(p.input_dim, p.hidden_dim_1)
        self.ln1 = nn.LayerNorm(p.hidden_dim_1)
        self.fc2 = nn.Linear(p.hidden_dim_1, p.hidden_dim_2)
        self.out = nn.Linear(p.hidden_dim_2, p.output_dim)
        self.drop = nn.Dropout(p.dropout)

    def forward(self, x):
        x = self.drop(torch.relu(self.ln1(self.fc1(x))))
        x = self.drop(torch.relu(self.fc2(x)))
        return self.out(x)


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def mask(client, start, end):
    dates = pd.to_datetime(client["forecast_date"]).normalize()
    return np.asarray((dates >= start) & (dates <= end), dtype=bool)


def load_clients():
    with (ROOT / "data/clients_with_spatial.pkl").open("rb") as f:
        return {int(k): v for k, v in pickle.load(f)["clients"].items()}


def make_scaler(clients, train_ids):
    x = np.vstack([np.asarray(clients[s]["x"], float)[mask(clients[s], TRAIN_START, TRAIN_END)] for s in train_ids])
    mean, scale = x.mean(0), x.std(0); scale[scale < 1e-8] = 1.0
    return mean, scale


def transform(x, mean, scale):
    return ((np.asarray(x, float) - mean) / scale).astype(np.float32)


def client_train(client, mean, scale):
    m = mask(client, TRAIN_START, TRAIN_END)
    return transform(np.asarray(client["x"])[m], mean, scale), np.asarray(client["y_residual"], np.float32)[m]


def weighted_average(states, counts, keys=None):
    keys = list(keys or states[0].keys()); total = float(sum(counts))
    return {k: sum(s[k].detach().float() * (n / total) for s, n in zip(states, counts)) for k in keys}


def local_epoch(model, x, y, seed, round_i, station_id, device, mu=0.0, global_params=None):
    p = RESIDUAL_MLP_PROTOCOL
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=p.shared_learning_rate, weight_decay=p.shared_weight_decay)
    xt, yt = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed * 100000 + round_i * 1000 + station_id)
    order = torch.randperm(len(x), generator=gen).cpu().numpy()
    for st in range(0, len(order), p.shared_batch_size):
        idx = torch.as_tensor(order[st:st+p.shared_batch_size], dtype=torch.long, device=device)
        opt.zero_grad(set_to_none=True); loss = torch.mean((model(xt[idx]) - yt[idx]) ** 2)
        if mu > 0:
            prox = sum(torch.sum((param - global_params[name]) ** 2) for name, param in model.named_parameters())
            loss = loss + 0.5 * mu * prox
        loss.backward(); opt.step()


def train_fedavg_prox(clients, train_ids, seed, device, mu=0.0, rounds=None):
    set_seed(seed); p = RESIDUAL_MLP_PROTOCOL; mean, scale = make_scaler(clients, train_ids)
    cached = {s: client_train(clients[s], mean, scale) for s in train_ids}
    global_model = ResidualMLP().to(device)
    for r in range(int(rounds or p.shared_rounds)):
        gp = {n: v.detach().clone() for n, v in global_model.named_parameters()}
        states, counts = [], []
        for sid in train_ids:
            local = copy.deepcopy(global_model).to(device); x, y = cached[sid]
            local_epoch(local, x, y, seed, r, sid, device, mu, gp)
            states.append({k: v.detach().clone() for k, v in local.state_dict().items()}); counts.append(len(x))
        global_model.load_state_dict(weighted_average(states, counts))
    return global_model, mean, scale


def train_fedper(clients, train_ids, seed, device, rounds=None):
    set_seed(seed); p = RESIDUAL_MLP_PROTOCOL; mean, scale = make_scaler(clients, train_ids)
    cached = {s: client_train(clients[s], mean, scale) for s in train_ids}
    initial = ResidualMLP().to(device).state_dict()
    shared = {k: v.detach().clone() for k, v in initial.items() if k.startswith(SHARED_PREFIXES)}
    heads = {sid: {k: v.detach().clone() for k, v in initial.items() if k.startswith(PERSONAL_PREFIXES)} for sid in train_ids}
    target_head = {k: v.detach().cpu().clone() for k, v in initial.items() if k.startswith(PERSONAL_PREFIXES)}
    for r in range(int(rounds or p.shared_rounds)):
        updates, counts = [], []
        for sid in train_ids:
            local = ResidualMLP().to(device); state = local.state_dict(); state.update(shared); state.update(heads[sid]); local.load_state_dict(state)
            x, y = cached[sid]; local_epoch(local, x, y, seed, r, sid, device)
            state = local.state_dict(); updates.append({k: v.detach().clone() for k, v in state.items() if k.startswith(SHARED_PREFIXES)})
            heads[sid] = {k: v.detach().clone() for k, v in state.items() if k.startswith(PERSONAL_PREFIXES)}; counts.append(len(x))
        shared = weighted_average(updates, counts, shared.keys())
    return shared, target_head, mean, scale


def to_long(client, dates, y, anchor, pred, fold, seed):
    rows = []
    for d, date in enumerate(dates):
        for slot in range(96):
            rows.append({"fold": fold, "seed": seed, "station_id": int(client["station_id"]), "type": str(client["station_type"]),
                         "prediction_date": pd.Timestamp(date), "slot": slot, "y_true": float(y[d, slot]),
                         "previous_day_baseline": float(anchor[d, slot]), "prediction": float(pred[d, slot])})
    return pd.DataFrame(rows)


def predict_direct(model, mean, scale, client, start, end, fold, seed, device):
    m = mask(client, start, end); dates = pd.to_datetime(np.asarray(client["forecast_date"])[m]).normalize()
    x = torch.from_numpy(transform(np.asarray(client["x"])[m], mean, scale)).to(device)
    model.eval()
    with torch.no_grad(): residual = model(x).cpu().numpy()
    anchor = np.asarray(client["baseline"], float)[m]; y = np.asarray(client["y_true"], float)[m]
    return to_long(client, dates, y, anchor, np.maximum(0, anchor + residual), fold, seed)


def predict_fedper(shared, target_head, mean, scale, client, start, end, fold, seed, device):
    p = RESIDUAL_MLP_PROTOCOL; dates = pd.to_datetime(client["forecast_date"]).normalize()
    x = transform(client["x"], mean, scale); residual = np.asarray(client["y_residual"], np.float32)
    anchor = np.asarray(client["baseline"], float); y = np.asarray(client["y_true"], float); rows = []
    for i, date in enumerate(dates):
        if date < start or date > end: continue
        hist = np.where(np.asarray((dates < date) & (dates >= date - pd.Timedelta(days=p.target_history_days))))[0]
        if len(hist) == 0: raise AssertionError("FedPer target history missing")
        model = ResidualMLP().to(device); state = model.state_dict(); state.update(shared); state.update({k: v.to(device) for k, v in target_head.items()}); model.load_state_dict(state)
        for n, param in model.named_parameters(): param.requires_grad = n.startswith(PERSONAL_PREFIXES)
        trainable = [v for v in model.parameters() if v.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=p.target_learning_rate, weight_decay=p.target_weight_decay)
        xt, yt = torch.from_numpy(x[hist]).to(device), torch.from_numpy(residual[hist]).to(device); model.train()
        for _ in range(p.target_inner_steps):
            opt.zero_grad(set_to_none=True); loss = torch.mean((model(xt) - yt) ** 2); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, p.target_grad_clip); opt.step()
        model.eval()
        with torch.no_grad(): r = model(torch.from_numpy(x[i:i+1]).to(device)).cpu().numpy()[0]
        rows.append(to_long(client, [date], y[i:i+1], anchor[i:i+1], np.maximum(0, anchor[i:i+1] + r[None, :]), fold, seed))
    return pd.concat(rows, ignore_index=True)


def pointwise_mean(frames):
    keys = ["fold", "station_id", "type", "prediction_date", "slot", "y_true", "previous_day_baseline"]
    out = frames[0].drop(columns="seed").rename(columns={"prediction": "p0"})
    for i, frame in enumerate(frames[1:], 1): out = out.merge(frame.drop(columns="seed").rename(columns={"prediction": f"p{i}"}), on=keys, validate="one_to_one")
    cols = [f"p{i}" for i in range(len(frames))]; out["prediction"] = out[cols].mean(axis=1)
    return out[keys + ["prediction"]]


def evaluate(name, frame, outdir):
    metrics = aggregate_system_metrics(frame, "prediction")
    b = station_wape_table(frame, "previous_day_baseline").rename(columns={"WAPE":"baseline_WAPE"}); m = station_wape_table(frame, "prediction").rename(columns={"WAPE":"method_WAPE"})
    st = b.merge(m, on=["station_id","type"]); st["relative_change"] = st.method_WAPE / st.baseline_WAPE - 1
    types = type_aggregate_wape_table(frame, "prediction", "previous_day_baseline")
    locked = load_locked_comparison_predictions(ROOT / "results_reference/oof_predictions_source.csv.gz")
    keys = ["fold","station_id","type","prediction_date","slot","y_true","previous_day_baseline"]
    comp = locked[keys+["proposed_prediction"]].merge(frame[keys+["prediction"]], on=keys, validate="one_to_one")
    daily = paired_daily_inference(comp, proposed_column="proposed_prediction", baseline_column="prediction")
    summary = {"method":name,"record_count":len(frame),"WAPE":float(metrics["WAPE"]),"nRMSE":float(metrics["nRMSE"]),"pooled_WAPE":float(metrics["pooled_WAPE"]),
               "improved_stations":int((st.relative_change<0).sum()),"max_station_relative_degradation":float(max(0,st.relative_change.max())),"paired_daily_proposed_minus_method":daily}
    frame.to_csv(outdir/f"{name}_oof.csv.gz", index=False, compression="gzip"); st.to_csv(outdir/f"{name}_station.csv", index=False); types.to_csv(outdir/f"{name}_type.csv", index=False)
    (outdir/f"{name}_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary


def fold_split(clients, fold):
    test, val = sorted(fold.test), sorted(fold.validation); train = sorted(set(clients)-set(test)-set(val))
    assert len(train)==36 and len(val)==2 and len(test)==2
    return train,val,test


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--methods",nargs="+",choices=["fedavg","fedprox","fedper"],default=["fedavg","fedprox","fedper"]); ap.add_argument("--device",default="auto")
    ap.add_argument("--folds",nargs="*"); ap.add_argument("--seeds",nargs="*",type=int,default=list(NEURAL_BASELINE_SEEDS)); ap.add_argument("--rounds",type=int); ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--output-dir",type=Path,default=ROOT/"results_revision_r1/neural_baselines"); args=ap.parse_args()
    cfg=load_config(ROOT/"paper_config.json"); folds=build_folds(cfg); clients=load_clients(); seeds=list(args.seeds); rounds=args.rounds
    if args.folds: folds=[f for f in folds if f.fold in set(args.folds)]
    if args.smoke: folds=folds[:1]; seeds=seeds[:1]; rounds=2; print("SMOKE MODE: not paper evidence",flush=True)
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device)); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    meta={"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"device":str(device),"folds":[f.fold for f in folds],"seeds":seeds,"rounds":int(rounds or RESIDUAL_MLP_PROTOCOL.shared_rounds),"methods":args.methods,"fedprox_mu_grid":FEDPROX_MU_GRID,"smoke":args.smoke}
    (out/"run_metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)
    summaries=[]

    if "fedavg" in args.methods:
        per_fold=[]
        for fold in folds:
            train,_,test=fold_split(clients,fold); seed_frames=[]
            for seed in seeds:
                model,mean,scale=train_fedavg_prox(clients,train,seed,device,0.0,rounds); seed_frames.append(pd.concat([predict_direct(model,mean,scale,clients[s],TEST_START,TEST_END,fold.fold,seed,device) for s in test],ignore_index=True)); print(f"[fedavg] {fold.fold} seed={seed}",flush=True)
            per_fold.append(pointwise_mean(seed_frames))
        avg=pd.concat(per_fold,ignore_index=True); locked=load_locked_comparison_predictions(ROOT/"results_reference/oof_predictions_source.csv.gz"); locked=locked[locked.fold.isin([f.fold for f in folds])]
        nm=aggregate_system_metrics(avg,"prediction")["WAPE"]; lm=aggregate_system_metrics(locked,"fedavg_direct_kw")["WAPE"]; formal=(not args.smoke and len(folds)==5 and len(seeds)==3 and int(rounds or 30)==30)
        cal={"new_WAPE":float(nm),"locked_WAPE":float(lm),"absolute_difference":float(abs(nm-lm)),"status":(("PASS" if abs(nm-lm)<=0.003 else "REVIEW") if formal else "DEVELOPMENT_ONLY")}; (out/"fedavg_calibration.json").write_text(json.dumps(cal,indent=2),encoding="utf-8"); print(json.dumps(cal,indent=2),flush=True)
        if formal: summaries.append(evaluate("fedavg_calibration",avg,out))

    if "fedprox" in args.methods:
        selected={}; final_folds=[]; grid_rows=[]
        for fold in folds:
            train,val,test=fold_split(clients,fold); cache={}; candidates=[]
            for mu in FEDPROX_MU_GRID:
                vals=[]
                for seed in seeds:
                    model,mean,scale=train_fedavg_prox(clients,train,seed,device,mu,rounds); cache[(mu,seed)]=(model,mean,scale); vals.append(pd.concat([predict_direct(model,mean,scale,clients[s],VAL_START,VAL_END,fold.fold,seed,device) for s in val],ignore_index=True)); print(f"[fedprox-val] {fold.fold} mu={mu} seed={seed}",flush=True)
                vm=pointwise_mean(vals); metric=aggregate_system_metrics(vm,"prediction"); row={"fold":fold.fold,"mu":mu,"validation_WAPE":metric["WAPE"],"validation_nRMSE":metric["nRMSE"]}; grid_rows.append(row); candidates.append(row)
            best=min(candidates,key=lambda r:(r["validation_WAPE"],r["mu"])); selected[fold.fold]=best["mu"]; tests=[]
            for seed in seeds:
                model,mean,scale=cache[(best["mu"],seed)]; tests.append(pd.concat([predict_direct(model,mean,scale,clients[s],TEST_START,TEST_END,fold.fold,seed,device) for s in test],ignore_index=True))
            final_folds.append(pointwise_mean(tests)); print(f"[fedprox] {fold.fold} selected mu={best['mu']}",flush=True)
        pd.DataFrame(grid_rows).to_csv(out/"fedprox_validation_grid.csv",index=False); (out/"fedprox_selected_mu.json").write_text(json.dumps(selected,indent=2),encoding="utf-8"); avg=pd.concat(final_folds,ignore_index=True)
        if not args.smoke and len(folds)==5 and len(seeds)==3: summaries.append(evaluate("fedprox",avg,out))

    if "fedper" in args.methods:
        final_folds=[]; val_rows=[]
        for fold in folds:
            train,val,test=fold_split(clients,fold); seed_tests=[]
            for seed in seeds:
                shared,head,mean,scale=train_fedper(clients,train,seed,device,rounds); vf=pd.concat([predict_fedper(shared,head,mean,scale,clients[s],VAL_START,VAL_END,fold.fold,seed,device) for s in val],ignore_index=True); vm=aggregate_system_metrics(vf,"prediction"); val_rows.append({"fold":fold.fold,"seed":seed,"validation_WAPE":vm["WAPE"],"validation_nRMSE":vm["nRMSE"]}); seed_tests.append(pd.concat([predict_fedper(shared,head,mean,scale,clients[s],TEST_START,TEST_END,fold.fold,seed,device) for s in test],ignore_index=True)); print(f"[fedper] {fold.fold} seed={seed}",flush=True)
            final_folds.append(pointwise_mean(seed_tests))
        pd.DataFrame(val_rows).to_csv(out/"fedper_validation.csv",index=False); avg=pd.concat(final_folds,ignore_index=True)
        if not args.smoke and len(folds)==5 and len(seeds)==3: summaries.append(evaluate("fedper",avg,out))

    if summaries:
        pd.DataFrame(summaries).to_csv(out/"new_baseline_summary.csv",index=False); print(pd.DataFrame(summaries).to_string(index=False),flush=True)


if __name__ == "__main__": main()
