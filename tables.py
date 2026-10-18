"""Table builders from summary.json (mean ± SD; CI where stated)."""
CTRL = ["Static-16", "Static-BDP", "AIMD", "PCON-style", "PID", "PID+LSTM"]
SCN = {"normal": "Normal operation", "high_traffic": "High traffic (2×)", "load_surge": "Load surge",
       "link_failure": "Link failure", "node_failure": "Node failure", "ddos": "Interest-flood DDoS"}
SCEN = list(SCN)


def ms(c, nd=0, sd=True):
    m, s, h, n = c
    if m != m:  # nan
        return "–"
    f = f"{{:.{nd}f}}"
    return f"{f.format(m)} ± {f.format(s)}" if sd else f.format(m)


def mci(c, nd=0):
    m, s, h, n = c
    f = f"{{:.{nd}f}}"
    return f"{f.format(m)} [{f.format(m - h)}, {f.format(m + h)}]"


def topo_table(S, topo):
    rows = []
    for r in S[f"table_{topo}"]:
        rows.append([r["ctrl"], mci(r["goodput"], 0), ms(r["util"], 2), ms(r["q_mean"], 1), ms(r["q_p95"], 1),
                     ms(r["rtt"], 1), ms(r["p95"], 1), ms(r["jitter"], 2), ms(r["timeouts"], 0),
                     ms(r["fairness"], 3)])
    head = ["Control law", "Goodput (kb/s) mean [95% CI]", "Util. U", "q̄ (pkts)", "q95 (pkts)", "Mean RTT (ms)",
            "p95 RTT (ms)", "Jitter (ms)", "Timeouts", "Jain J"]
    return head, rows, [1.25, 1.9, 0.75, 0.85, 0.85, 0.95, 0.95, 0.85, 0.9, 0.85]


def t3_60(S):
    rows = []
    for r in S["table_T3_60"]:
        rows.append([r["ctrl"], ms(r["goodput"], 0), ms(r["q_mean"], 1), ms(r["rtt"], 1), ms(r["p95"], 1),
                     ms(r["timeouts"], 0), ms(r["fairness"], 3)])
    return (["Control law (60 s, 5 seeds)", "Goodput (kb/s)", "q̄ (pkts)", "Mean RTT (ms)", "p95 RTT (ms)",
             "Timeouts", "Jain J"], rows, [1.6, 1.4, 1, 1, 1, 1, 0.9])


def scen_table(S, metric="goodput", nd=0):
    rows = []
    M = S["matrix"]
    for topo in ["T1", "T2", "T3"]:
        for s in SCEN:
            row = [topo, SCN[s]]
            for c in CTRL:
                cell = next(x for x in M if x["topo"] == topo and x["scen"] == s and x["ctrl"] == c)
                row.append(ms(cell[metric], nd))
            rows.append(row)
    return (["Topology", "Scenario"] + CTRL, rows, [0.55, 1.25] + [1.15] * 6)


def phase_table(S, scen):
    rows = []
    M = S["matrix"]
    for topo in ["T1", "T2", "T3"]:
        for c in CTRL:
            x = next(x for x in M if x["topo"] == topo and x["scen"] == scen and x["ctrl"] == c)
            rows.append([topo, c, ms(x["gp_pre"], 0), ms(x["gp_fail"], 0), ms(x["gp_post"], 0),
                         ms(x["recovery"], 1), ms(x["timeouts"], 0)])
    return (["Topology", "Control law", "Goodput before (kb/s)", "Goodput during fault (kb/s)",
             "Goodput after (kb/s)", "Recovery time (s)", "Timeouts"], rows, [0.6, 1.0, 1.2, 1.2, 1.2, 1.0, 0.9])


def ddos_table(S):
    rows = []
    for x in S["ddos_def"]:
        rows.append([x["topo"], x["ctrl"], ms(x["pre"], 0), ms(x["und"], 0), ms(x["dfd"], 0),
                     ms(x["nack_und"], 0), ms(x["nack_def"], 0), ms(x["to_def"], 0)])
    return (["Topology", "Control law", "Goodput pre-attack (kb/s)", "During flood, no defence (kb/s)",
             "During flood, PIT admission (kb/s)", "NACKs, no defence", "NACKs, PIT admission",
             "Timeouts, PIT admission"], rows, [0.6, 1.0, 1.1, 1.2, 1.2, 1.0, 1.0, 0.9])


def ablation_table(S):
    rows = []
    A = S["ablation"]
    for topo in ["T1", "T2", "T3"]:
        for s in ["normal", "load_surge", "high_traffic"]:
            for ff in [0.1, 0.2, 0.35]:
                a = next(x for x in A if x["topo"] == topo and x["scen"] == s and x["floor"] == ff and x["ctrl"] == "PID")
                b = next(x for x in A if x["topo"] == topo and x["scen"] == s and x["floor"] == ff and x["ctrl"] == "PID+LSTM")
                rows.append([topo, SCN[s], f"{ff:.2f}", ms(a["goodput"], 0), ms(b["goodput"], 0), ms(a["q_p95"], 1),
                             ms(b["q_p95"], 1), ms(a["marks"], 0), ms(b["marks"], 0)])
    return (["Topology", "Scenario", "θmin/B", "Goodput PID", "Goodput PID+LSTM", "q95 PID", "q95 PID+LSTM",
             "Marks PID", "Marks PID+LSTM"], rows, [0.6, 1.1, 0.6, 1.1, 1.1, 0.8, 0.9, 0.8, 0.9])


def forecast_table(S):
    rows = []
    for x in S["forecast"]:
        rows.append([x["topo"], x["ctrl"], ms(x["q_mean"], 1), ms(x["lstm"], 2), ms(x["persist"], 2),
                     f"{100 * (1 - x['lstm'][0] / x['persist'][0]):+.1f}%" if x["persist"][0] > 0 else "–"])
    return (["Topology", "Control law (queue being forecast)", "Mean queue (pkts)", "LSTM MAE (pkts)",
             "Persistence MAE (pkts)", "Error reduction vs persistence"], rows, [0.7, 1.7, 1.0, 1.0, 1.1, 1.1])


def sampling_table(S):
    rows = []
    for x in S["sampling"]:
        rows.append([x["topo"], SCN[x["scen"]], f"{x['dt']:.2f}", f"{x['samples_per_s']:.0f}", ms(x["lstm"], 2),
                     ms(x["persist"], 2), ms(x["goodput"], 0), ms(x["q_p95"], 1), f"{x['det_us'][0]:.1f}",
                     f"{x['inf_us'][0]:.0f}", f"{x['inf_us'][0] / x['dt'] / 1e3:.1f}"])
    return (["Topology", "Scenario", "Δts (s)", "Samples/s per face", "LSTM MAE (pkts)", "Persistence MAE (pkts)",
             "Goodput (kb/s)", "q95 (pkts)", "Detector CPU (µs/sample)", "LSTM inference (µs)",
             "LSTM CPU (ms per s per face)"], rows, [0.6, 1.0, 0.55, 0.7, 0.9, 0.9, 1.0, 0.7, 0.8, 0.8, 0.9])
