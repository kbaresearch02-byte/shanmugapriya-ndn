"""Re-run failure-scenario jobs with the live-face strategy and merge into runs.csv/traces.json."""
import csv, json, os, sys, time
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rev_experiments as rx
from run_all import _job, SEEDS, TOPOS
OUT = sys.argv[1]
pre = json.load(open(os.path.join(OUT, "lstm_pretrain.json")))
W = lambda t: pre[f"{t}|0.1"]["w"]
if __name__ == "__main__":
    jobs = []
    for topo in TOPOS:
        for scen in ("link_failure", "node_failure"):
            for ctrl in rx.CONTROLLERS:
                for s in SEEDS:
                    jobs.append((("A",), dict(topo=topo, scen=scen, ctrl=ctrl, seed=s, trace=(s == 1)),
                                 W(topo) if ctrl == "PID+LSTM" else None))
        for ctrl in ("Static-16", "AIMD", "PCON-style", "PID"):
            for s in SEEDS:
                jobs.append((("F",), dict(topo=topo, scen="link_failure", ctrl=ctrl, seed=s, shadow=True), W(topo)))
    jobs.sort(key=lambda j: j[1]["topo"] != "T3")
    t0 = time.time(); rows = []; traces = json.load(open(os.path.join(OUT, "traces.json")))
    with Pool(11) as pool:
        for i, (tag, r, tr) in enumerate(pool.imap_unordered(_job, jobs)):
            r["exp"] = tag[0]; rows.append(r)
            if tr is not None:
                traces[f"{tag[0]}|{r['topo']}|{r['scen']}|{r['ctrl']}|def{r.get('defense', 0)}"] = tr
            if i % 50 == 0: print(i, round(time.time() - t0), flush=True)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(OUT, "rerun_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    json.dump(traces, open(os.path.join(OUT, "traces.json"), "w"))
    print("done", len(rows), round(time.time() - t0))
