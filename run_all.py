"""Driver for all revision experiments (parallel). Writes CSV + traces + LSTM weights."""
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rev_experiments as rx  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out")
SEEDS = list(range(1, 11))
TOPOS = ["T1", "T2", "T3"]
SCENS = ["normal", "high_traffic", "load_surge", "link_failure", "node_failure", "ddos"]
DTS = [0.02, 0.05, 0.1, 0.2, 0.5]


def _pre(args):
    topo, dt = args
    w, losses, n, secs = rx.pretrain(topo, dt_s=dt)
    return (topo, dt), dict(w={k: v.tolist() for k, v in w.items()}, losses=losses, n=n, secs=secs)


def _job(job):
    tag, kw, weights = job
    if weights is not None:
        kw = dict(kw, lstm_weights={k: np.asarray(v) for k, v in weights.items()})
    t0 = time.perf_counter()
    try:
        r = rx.run_one(**kw)
    except Exception as e:  # keep going, record failure
        return tag, dict(kw_err=str(e), **{k: v for k, v in kw.items() if k != "lstm_weights"}), None
    r["wall_s"] = time.perf_counter() - t0
    tr = r.pop("_trace", None)
    if tr is not None:
        tr.pop("all_q", None)
    return tag, r, tr


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    pre_keys = [(t, 0.1) for t in TOPOS] + [(t, d) for t in ("T1", "T2") for d in DTS if d != 0.1]
    with Pool(11) as pool:
        pre = dict(pool.map(_pre, pre_keys))
    json.dump({f"{k[0]}|{k[1]}": v for k, v in pre.items()}, open(os.path.join(OUT, "lstm_pretrain.json"), "w"))
    print("pretrain done", round(time.time() - t0), "s", flush=True)

    def W(topo, dt=0.1):
        return pre[(topo, dt)]["w"]

    jobs = []
    for topo in TOPOS:
        for scen in SCENS:
            for ctrl in rx.CONTROLLERS:
                for s in SEEDS:
                    kw = dict(topo=topo, scen=scen, ctrl=ctrl, seed=s, trace=(s == 1))
                    jobs.append((("A",), kw, W(topo) if ctrl == "PID+LSTM" else None))
        for ctrl in rx.CONTROLLERS:
            for s in SEEDS:
                kw = dict(topo=topo, scen="ddos", ctrl=ctrl, seed=s, defense=True, trace=(s == 1))
                jobs.append((("B",), kw, W(topo) if ctrl == "PID+LSTM" else None))
        for scen in ("normal", "load_surge", "high_traffic"):
            for ff in (0.1, 0.2):
                for ctrl in ("PID", "PID+LSTM"):
                    for s in SEEDS:
                        kw = dict(topo=topo, scen=scen, ctrl=ctrl, seed=s, floor_frac=ff)
                        jobs.append((("C",), kw, W(topo) if ctrl == "PID+LSTM" else None))
        for ctrl in rx.CONTROLLERS:
            for s in SEEDS[:5]:
                kw = dict(topo="T3", scen="normal", ctrl=ctrl, seed=s, duration=60.0) if topo == "T3" else None
                if kw:
                    jobs.append((("E",), kw, W("T3") if ctrl == "PID+LSTM" else None))
    for topo in TOPOS:   # F: open-loop (shadow) forecasting accuracy on every control law
        for scen in ("normal", "load_surge", "link_failure"):
            for ctrl in ("Static-16", "AIMD", "PCON-style", "PID"):
                for s in SEEDS:
                    kw = dict(topo=topo, scen=scen, ctrl=ctrl, seed=s, shadow=True)
                    jobs.append((("F",), kw, W(topo)))
    for topo in ("T1", "T2"):
        for dt in DTS:
            for scen in ("normal", "load_surge"):
                for s in SEEDS:
                    kw = dict(topo=topo, scen=scen, ctrl="PID+LSTM", seed=s, dt_s=dt)
                    jobs.append((("D",), kw, W(topo, dt)))
    # slowest first for better packing
    jobs.sort(key=lambda j: (j[1]["topo"] != "T3", j[1].get("scen") != "ddos"))
    print("jobs", len(jobs), flush=True)
    rows, traces = [], {}
    with Pool(11) as pool:
        for i, (tag, r, tr) in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            r["exp"] = tag[0]
            rows.append(r)
            if tr is not None:
                key = f"{tag[0]}|{r['topo']}|{r['scen']}|{r['ctrl']}|def{r.get('defense', 0)}"
                traces[key] = tr
            if i % 50 == 0:
                print(i, round(time.time() - t0), "s", flush=True)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(OUT, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json.dump(traces, open(os.path.join(OUT, "traces.json"), "w"))
    print("done", len(rows), round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
