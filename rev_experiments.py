"""Revision experiments for the Objective-1 manuscript (reviewer response).

Corrects the original custom-topology runner, which ignored scenario events
(link/node failure, Interest flood) and never attached attacker nodes, and adds
what the reviewers asked for:

* independent replications (seeds) with stochastic forwarding + start offsets;
* six scenarios whose events are really injected (failures, surge, flood);
* baselines: Static-16 (the former "Fixed"), Static-BDP (oracle, delay-aware),
  loss-driven AIMD, PCON-style mark-driven AIMD, PID, and PID+LSTM (closed loop);
* router-side EWMA / adaptive-threshold detector (Eqs. 1-3) sampled every dt_s;
* closed-loop LSTM forecaster that can raise marks proactively (ablation),
  with offline pre-training on held-out seeds and periodic online fine-tuning;
* per-run overhead accounting (detector / LSTM CPU time, samples).
"""
import copy
import json
import math
import os
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from ndnsuite.core.spec import TopologySpec                      # noqa: E402
from ndnsuite.core.build import build_topology, fail_link, restore_link  # noqa: E402
from ndnsuite.core.traffic import (Consumer, CBRConsumer, FixedWindow, AIMD,  # noqa: E402
                                   PIDFeedback, make_producer)
from ndnsuite.core.strategy import make_strategy                 # noqa: E402
from ndnsuite.core.metrics import jain_fairness                  # noqa: E402
from ndnsuite.core.packets import DATA_SIZE                       # noqa: E402
from ndnsuite.ai.lstm import LSTM                                  # noqa: E402

LIB = os.path.join(os.path.dirname(HERE), "library", "topologies")


# ---------------------------------------------------------------------------
# Topologies
# ---------------------------------------------------------------------------
def topo_T1():
    """Linear single-bottleneck network: C1-C3 (+ attacker X1) -> R1 == R2 -> P1."""
    def node(n, k):
        return {"name": n, "kind": k}
    def link(a, b, bw, d, q=60):
        return {"a": a, "b": b, "bw_mbps": bw, "delay_ms": d, "queue_cap": q,
                "mark_threshold": 0, "loss": 0.0}
    d = {"name": "T1", "nodes": [node("C1", "consumer"), node("C2", "consumer"),
                                 node("C3", "consumer"), node("X1", "attacker"),
                                 node("R1", "router"), node("R2", "router"),
                                 node("P1", "producer")],
         "links": [link("C1", "R1", 50, 2), link("C2", "R1", 50, 2), link("C3", "R1", 50, 2),
                   link("X1", "R1", 50, 2), link("R1", "R2", 6, 20), link("R2", "P1", 50, 2)]}
    return d


def load_topo(name):
    if name == "T1":
        d = topo_T1()
    elif name == "T2":
        d = json.load(open(os.path.join(LIB, "Example-Diamond-Multipath.json")))
    else:
        d = json.load(open(os.path.join(LIB, "multi.json")))
    for l in d["links"]:
        l["mark_threshold"] = 0          # marking is done by the detector policy only
    d["name"] = name
    return TopologySpec.from_dict(d)


# failure targets per topology (documented in the manuscript, Table 2)
FAIL = {
    "T1": {"link": ("C1", "R1"), "node": "C3"},
    "T2": {"link": ("R1", "G"), "node": "G"},
    "T3": {"link": ("R1", "C3"), "node": "R1"},
}
T_FAIL, T_RESTORE = 6.0, 12.0
T_ATTACK = 5.0
ATTACK_PPS = 500.0          # per attacker node
T_SURGE = 8.0


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------
class AIMDECN(AIMD):
    """PCON-style consumer: AIMD that also backs off (once per RTT) on marks."""
    name = "PCON-style"

    def __init__(self, beta=0.5, **kw):
        super().__init__(**kw)
        self.beta = beta
        self._last_cut = -1e9
        self._srtt = 0.1
        self.now = lambda: 0.0

    def on_data(self, rtt, marked):
        self._srtt = 0.875 * self._srtt + 0.125 * rtt
        t = self.now()
        if marked and t - self._last_cut > self._srtt:
            self.ssthresh = max(2.0, self.cwnd * self.beta)
            self.cwnd = max(1.0, self.cwnd * self.beta)
            self._last_cut = t
            return
        super().on_data(rtt, marked)


class LiveFaceStrategy:
    """Adaptive weighted multipath that, like NFD, does not forward on a face
    whose link is down (face state DOWN) when an alternative next hop is up."""
    name = "AdaptiveWeighted(live)"

    def __init__(self):
        self.inner = make_strategy("AdaptiveWeighted")

    def choose(self, node, interest, candidates):
        live = [(fid, c) for fid, c in candidates if node.faces[fid].up]
        # no live next hop (e.g. a host whose only uplink is down): behave like the
        # original strategy; the Interest is lost and the consumer's RTO handles it
        return self.inner.choose(node, interest, live or candidates)


CONTROLLERS = ["Static-16", "Static-BDP", "AIMD", "PCON-style", "PID", "PID+LSTM"]


# ---------------------------------------------------------------------------
# Router-side detector (Eqs. 1-3) + optional LSTM forecaster
# ---------------------------------------------------------------------------
class Forecaster:
    """Shared LSTM (weights shared by all monitored faces of a topology)."""
    WIN = 16

    def __init__(self, horizon, weights=None, hidden=24, seed=1):
        self.H = horizon
        self.model = LSTM(4, hidden=hidden, seed=seed)
        if weights is not None:
            for k, v in weights.items():
                getattr(self.model, k)[...] = v
        self.model.opt.lr = 3e-4          # conservative online fine-tuning rate
        self.infer_time = 0.0
        self.infer_calls = 0
        self.train_time = 0.0
        self.train_calls = 0

    def weights(self):
        return {k: getattr(self.model, k).copy() for k in ("Wx", "Wh", "b", "Wo", "bo")}

    def predict(self, feats):
        t0 = time.perf_counter()
        x = np.asarray(feats, dtype=float)[None, :, :]
        y = float(self.model.predict(x)[0, 0])
        self.infer_time += time.perf_counter() - t0
        self.infer_calls += 1
        return y

    def finetune(self, X, Y, steps=2):
        t0 = time.perf_counter()
        for _ in range(steps):
            self.model.train_batch(X, Y)
        self.train_time += time.perf_counter() - t0
        self.train_calls += 1


class Detector:
    """EWMA queue smoothing + adaptive mean+kappa*sigma threshold (+ LSTM)."""

    def __init__(self, face, cap, alpha=0.15, kappa=2.0, win=50, floor_frac=0.35,
                 forecaster=None, pcon=False, dt=0.1, shadow=False):
        self.shadow = shadow
        self.face = face
        self.cap = cap
        self.alpha = alpha
        self.kappa = kappa
        self.hist = deque(maxlen=win)
        self.floor = floor_frac * cap
        self.ewma = 0.0
        self.theta = self.floor
        self.marking = False
        self.fc = forecaster
        self.pred = 0.0
        self.feat = deque(maxlen=Forecaster.WIN)
        self.qhist = []            # raw samples (for training / scoring)
        self.predhist = []         # (index, prediction)
        self.pcon = pcon
        self._above_since = None
        self.dt = dt
        self.cpu = 0.0
        self.samples = 0
        self._last_marks = 0
        face.mark_policy = self.policy

    def sample(self, now):
        t0 = time.perf_counter()
        q = self.face.qlen()
        self.ewma = (1 - self.alpha) * self.ewma + self.alpha * q          # Eq. (1)
        self.hist.append(self.ewma)
        if len(self.hist) >= 5:
            mu = sum(self.hist) / len(self.hist)
            var = sum((h - mu) ** 2 for h in self.hist) / len(self.hist)
            self.theta = min(0.8 * self.cap, max(self.floor, mu + self.kappa * math.sqrt(var)))  # Eq. (2)
        self.marking = self.ewma > self.theta                                # Eq. (3)
        self.samples += 1
        self.cpu += time.perf_counter() - t0
        self.qhist.append(q)
        dm = self.face.marks - self._last_marks
        self._last_marks = self.face.marks
        prev = self.qhist[-2] if len(self.qhist) > 1 else q
        self.feat.append([q / self.cap, self.ewma / self.cap, (q - prev) / self.cap,
                          min(1.0, dm / 10.0)])
        if self.fc is not None and len(self.feat) == Forecaster.WIN:
            self.pred = max(0.0, q + self.fc.predict(list(self.feat)) * self.cap)
            self.predhist.append((len(self.qhist) - 1, self.pred))
            if self.pred > self.theta and not self.shadow:                   # proactive mark
                self.marking = True

    def policy(self, face, pkt):
        if self.pcon:
            # CoDel-style: queuing-delay estimate above 5 ms for >= 100 ms
            qdelay = face.qlen() * DATA_SIZE * 8 / face.bandwidth_bps
            now = face.sim.now
            if qdelay > 0.005:
                if self._above_since is None:
                    self._above_since = now
                return now - self._above_since >= 0.1
            self._above_since = None
            return False
        return self.marking


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def consumer_paths(spec, prods):
    """Shortest path (node list) from each consumer to the nearest producer."""
    from collections import deque as dq
    paths = {}
    for c in spec.by_role("consumer"):
        prev = {c: None}
        q = dq([c])
        hit = None
        while q:
            cur = q.popleft()
            if cur in prods:
                hit = cur
                break
            for nb in spec.neighbors(cur):
                if nb not in prev:
                    prev[nb] = cur
                    q.append(nb)
        path = []
        while hit is not None:
            path.append(hit)
            hit = prev[hit]
        paths[c] = list(reversed(path))
    return paths


def link_of(spec, a, b):
    for l in spec.links:
        if (l.a, l.b) in ((a, b), (b, a)):
            return l
    return None


def _dist_from(spec, srcs):
    from collections import deque as dq
    dist = {x: 0 for x in srcs}
    q = dq(srcs)
    while q:
        cur = q.popleft()
        for nb in spec.neighbors(cur):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    return dist


def dag_layers(spec, prods, c):
    """Shortest-path DAG from consumer c to the producer, grouped per hop layer.
    Returns a list of layers; each layer is a frozenset of (upstream, downstream)
    edges (Data direction upstream -> downstream)."""
    dist = _dist_from(spec, prods)
    layers, frontier = [], {c}
    while frontier and not (frontier & set(prods)):
        edges, nxt = set(), set()
        for u in frontier:
            for nb in spec.neighbors(u):
                if dist.get(nb, 1e9) == dist[u] - 1:
                    edges.add((nb, u))
                    nxt.add(nb)
        if not edges:
            break
        layers.append(frozenset(edges))
        frontier = nxt
    return layers


def bdp_windows(spec, prods, n_flows_per_consumer=1, q_target=2):
    """Oracle delay-aware static window (Static-BDP).

    Per-consumer max-min fair rate obtained by water-filling over the
    shortest-path DAG (parallel equal-cost branches pooled), multiplied by the
    consumer's base RTT, plus a q_target-packet allowance."""
    cons = spec.by_role("consumer")
    lay = {c: dag_layers(spec, prods, c) for c in cons}
    cap = {}
    for c in cons:
        for L in lay[c]:
            cap[L] = sum(link_of(spec, u, v).bw_mbps * 1e6 for u, v in L)
    rate = {c: 0.0 for c in cons}
    active = set(cons)
    remaining = dict(cap)
    while active:
        users = {L: [c for c in active if L in lay[c]] for L in remaining}
        inc = min(remaining[L] / (len(u) * n_flows_per_consumer)
                  for L, u in users.items() if u)
        sat = set()
        for L, u in users.items():
            if not u:
                continue
            remaining[L] -= inc * len(u) * n_flows_per_consumer
            if remaining[L] <= 1e-6:
                sat |= set(u)
        for c in active:
            rate[c] += inc
        active -= sat
        if not sat:
            break
    w = {}
    for c in cons:
        rtt = 0.0
        for L in lay[c]:
            u, v = sorted(L)[0]
            l = link_of(spec, u, v)
            rtt += 2 * l.delay_ms / 1000.0 + DATA_SIZE * 8 / (l.bw_mbps * 1e6)
        w[c] = max(1, int(math.ceil(rate[c] * rtt / (DATA_SIZE * 8))) + q_target)
    return w


def monitored_faces(bt, spec, prods):
    """Faces carrying Data toward >=2 consumers over the shortest-path DAG."""
    cnt = {}
    for c in spec.by_role("consumer"):
        for L in dag_layers(spec, prods, c):
            for e in L:
                cnt[e] = cnt.get(e, 0) + 1
    out = []
    for (src, dst), n in sorted(cnt.items()):
        if n >= 2:
            f = bt.face_between.get((src, dst))
            if f is not None:
                out.append(((src, dst), f))
    return out


class PitQuota:
    """PIT fair-share admission: each incoming face may hold at most
    capacity / (number of active incoming faces) PIT entries."""

    def __init__(self, node):
        self.node = node
        self.counts = {}
        node.interest_admission = self

    def refresh(self):
        c = {}
        for e in self.node.pit.values():
            for f in e.downstreams:
                c[f] = c.get(f, 0) + 1
        self.counts = c

    def __call__(self, node, interest, in_fid):
        if node.producers and interest.name.startswith(tuple(node.producers)):
            return True
        if interest.name in node.pit:
            return True
        active = max(1, len(self.counts) or 1)
        quota = node.pit_capacity / active
        n = self.counts.get(in_fid, 0)
        if n >= quota and len(node.pit) >= 0.5 * node.pit_capacity:
            return "nack"
        self.counts[in_fid] = n + 1
        return True


def bottleneck_faces(bt, spec, prods):
    """All min-capacity links, Data direction (producer side -> consumer side)."""
    caps = [min(l.bw_mbps, l.rev()) for l in spec.links]
    m = min(caps)
    from collections import deque as dq
    dist = {p: 0 for p in prods}
    q = dq(prods)
    while q:
        cur = q.popleft()
        for nb in spec.neighbors(cur):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    out = []
    for l in spec.links:
        if min(l.bw_mbps, l.rev()) == m:
            ps, cs = (l.a, l.b) if dist.get(l.a, 1e9) <= dist.get(l.b, 1e9) else (l.b, l.a)
            # only links that consumer traffic actually traverses
            if "X" in cs[:1] or "X" in ps[:1]:
                continue
            f = bt.face_between.get((ps, cs))
            if f is not None:
                out.append(((ps, cs), f))
    return out


def node_down(bt, name, up):
    node = bt.topo.nodes[name]
    for f in node.faces.values():
        f.up = up
    # faces of neighbours pointing to this node
    for (a, b), f in bt.face_between.items():
        if b == name:
            f.up = up


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def run_one(topo, scen, ctrl, seed, duration=20.0, dt_s=0.1, horizon_s=0.8,
            lstm_weights=None, online=True, trace=False, kappa=2.0, alpha=0.15,
            floor_frac=0.35, defense=False, shadow=False):
    spec = load_topo(topo)
    prods = spec.by_role("producer")
    cons_nodes = spec.by_role("consumer")
    atts = spec.by_role("attacker")
    rng = np.random.default_rng(seed)
    bt = build_topology(spec, seed=seed)
    sim = bt.sim
    for n in bt.topo.nodes.values():
        n.strategy = LiveFaceStrategy()
    for pn in prods:
        make_producer(bt.topo.nodes[pn], "/data", phantom_sub="fake")
        bt.add_routes_for_prefix("/data", pn, multipath=True)
    paths = consumer_paths(spec, prods)
    nflows = 2 if scen == "high_traffic" else 1
    bdpw = bdp_windows(spec, prods, nflows)

    # router-side detectors on aggregation faces
    H = max(1, int(round(horizon_s / dt_s)))
    fc = None
    if ctrl == "PID+LSTM" or shadow:
        fc = Forecaster(H, weights=lstm_weights)
    dets = []
    for (pair, f) in monitored_faces(bt, spec, prods):
        cap = f.queue.capacity if hasattr(f.queue, "capacity") else 60
        dets.append(Detector(f, cap, alpha=alpha, kappa=kappa, forecaster=fc, floor_frac=floor_frac,
                             pcon=(ctrl == "PCON-style"), dt=dt_s, shadow=shadow))

    def make_ctrl(cnode):
        if ctrl == "Static-16":
            return FixedWindow(16)
        if ctrl == "Static-BDP":
            return FixedWindow(bdpw[cnode])
        if ctrl == "AIMD":
            return AIMD()
        if ctrl == "PCON-style":
            c = AIMDECN()
            c.now = lambda: sim.now
            return c
        return PIDFeedback()

    consumers = []
    owner = []
    surge = scen == "load_surge"
    for i, cn in enumerate(cons_nodes):
        for k in range(nflows):
            st = float(rng.uniform(0.0, 0.5))
            if surge and i % 2 == 1:
                st = T_SURGE + float(rng.uniform(0.0, 0.2))
            consumers.append(Consumer(sim, bt.topo.nodes[cn], f"/data/c{i}_{k}",
                                      make_ctrl(cn), start=st))
            owner.append(cn)

    attackers = []
    if scen == "ddos":
        for an in atts:
            attackers.append(CBRConsumer(sim, bt.topo.nodes[an], "/data/fake",
                                         rate_pps=ATTACK_PPS, start=T_ATTACK, unique=True,
                                         malicious=True, lifetime=2.0))
    quotas = []
    if defense:
        for n in bt.topo.nodes.values():
            if n.name not in prods and n.name not in atts:
                quotas.append(PitQuota(n))

        def qref():
            for qq in quotas:
                qq.refresh()
            sim.schedule(0.1, qref)
        sim.schedule(0.0, qref)
    events = []
    if scen == "link_failure":
        a, b = FAIL[topo]["link"]
        sim.schedule(T_FAIL, lambda: (fail_link(bt, a, b), events.append((sim.now, "link down"))))
        sim.schedule(T_RESTORE, lambda: (restore_link(bt, a, b), events.append((sim.now, "link up"))))
    if scen == "node_failure":
        nm = FAIL[topo]["node"]
        sim.schedule(T_FAIL, lambda: (node_down(bt, nm, False), events.append((sim.now, "node down"))))
        sim.schedule(T_RESTORE, lambda: (node_down(bt, nm, True), events.append((sim.now, "node up"))))

    # periodic detector sampling (dt_s) + online LSTM fine-tuning (every 2 s)
    def sample_tick():
        for d in dets:
            d.sample(sim.now)
        if fc is not None and online and len(dets[0].qhist) % max(1, int(2.0 / dt_s)) == 0:
            X, Y = [], []
            for d in dets:
                qh = d.qhist
                if len(qh) < Forecaster.WIN + H + 5:
                    continue
                fe = _features(qh, d.cap, d.alpha)
                for s in range(max(0, len(qh) - 60 - Forecaster.WIN - H), len(qh) - Forecaster.WIN - H + 1, 3):
                    X.append(fe[s:s + Forecaster.WIN])
                    Y.append([(qh[s + Forecaster.WIN + H - 1] - qh[s + Forecaster.WIN - 1]) / d.cap])
            if len(X) >= 8:
                fc.finetune(np.asarray(X), np.asarray(Y), steps=2)
        sim.schedule(dt_s, sample_tick)
    if dets:
        sim.schedule(dt_s, sample_tick)

    # 0.1 s metric sampler
    bnecks = bottleneck_faces(bt, spec, prods)
    ts, qs, busy, gp_bins, pit_r = [], [], [], [], []
    last_bytes = [0]
    entry_routers = sorted({p[1] for p in paths.values() if len(p) > 1})

    def tick(now):
        ts.append(now)
        qs.append(max(f.qlen() for _, f in bnecks) if bnecks else 0)
        busy.append(np.mean([1.0 if f.busy else 0.0 for _, f in bnecks]) if bnecks else 0)
        tot = sum(c.delivered_bytes for c in consumers)
        gp_bins.append((tot - last_bytes[0]) * 8 / 1000.0 / 0.1)
        last_bytes[0] = tot
        pit_r.append(max(len(bt.topo.nodes[r].pit) for r in entry_routers))

    sim.run(until=duration, tick_cb=tick, tick_dt=0.1)

    # ---- metrics ----
    qs = np.asarray(qs, float)
    rtts = [np.array([r for _, r in c.rtts]) for c in consumers]
    allr = np.concatenate([r for r in rtts if len(r)] or [np.array([0.0])])
    jit = [np.mean(np.abs(np.diff(r))) for r in rtts if len(r) > 2]
    per_cons = {}
    for c, o in zip(consumers, owner):
        per_cons[o] = per_cons.get(o, 0) + c.delivered_bytes
    gp = sum(c.delivered_bytes for c in consumers) * 8 / 1000.0 / duration
    sent_leg = sum(c.next_seq + c.retx for c in consumers)
    res = dict(topo=topo, scen=scen, ctrl=ctrl, seed=seed, dt_s=dt_s, duration=duration,
               floor_frac=floor_frac, kappa=kappa, defense=int(defense), shadow=int(shadow),
               goodput=gp, util=float(np.mean(busy)) if busy else 0.0,
               q_mean=float(qs.mean()), q_p95=float(np.percentile(qs, 95)),
               rtt=float(allr.mean() * 1000), p95=float(np.percentile(allr, 95) * 1000),
               jitter=float(np.mean(jit) * 1000) if jit else 0.0,
               timeouts=int(sum(c.timeouts for c in consumers)),
               nacks=int(sum(c.nacks for c in consumers)),
               drops=int(sum(f.drops for _, f in bnecks)),
               marks=int(sum(d.face.marks for d in dets)),
               fairness=jain_fairness(list(per_cons.values())),
               pit_max=int(max(pit_r) if pit_r else 0),
               isr=float(sum(c.delivered for c in consumers) / max(1, sent_leg)),
               att_sent=int(sum(a.sent for a in attackers)),
               det_cpu_us=float(1e6 * sum(d.cpu for d in dets) / max(1, sum(d.samples for d in dets))),
               det_samples=int(sum(d.samples for d in dets)), n_dets=len(dets))
    # goodput time series phases
    g = np.asarray(gp_bins, float)
    tt = np.asarray(ts, float)

    def win(a, b):
        m = (tt >= a) & (tt < b)
        return float(g[m].mean()) if m.any() else 0.0
    if scen in ("link_failure", "node_failure"):
        pre, dur_f, post = win(3, T_FAIL), win(T_FAIL, T_RESTORE), win(T_RESTORE + 2, duration)
        res.update(gp_pre=pre, gp_fail=dur_f, gp_post=post)
        sm = np.convolve(g, np.ones(10) / 10, mode="same")
        rec = np.where((tt >= T_RESTORE) & (sm >= 0.9 * pre))[0]
        res["recovery_s"] = float(tt[rec[0]] - T_RESTORE) if len(rec) else float("nan")
    if scen == "ddos":
        res.update(gp_pre=win(2, T_ATTACK), gp_fail=win(T_ATTACK + 1, duration), gp_post=float("nan"))
    if scen == "load_surge":
        res.update(gp_pre=win(3, T_SURGE), gp_fail=win(T_SURGE + 1, duration), gp_post=float("nan"),
                   q_surge_p95=float(np.percentile(qs[tt >= T_SURGE], 95)),
                   q_surge_max=float(qs[tt >= T_SURGE].max()))
    # LSTM accuracy on the monitored faces (closed loop, realised future)
    if fc is not None:
        errs, perr = [], []
        for d in dets:
            for idx, p in d.predhist:
                j = idx + H
                if j < len(d.qhist):
                    errs.append(abs(p - d.qhist[j]))
                    perr.append(abs(d.qhist[idx] - d.qhist[j]))
        res.update(lstm_mae=float(np.mean(errs)) if errs else float("nan"),
                   persist_mae=float(np.mean(perr)) if perr else float("nan"),
                   lstm_infer_us=1e6 * fc.infer_time / max(1, fc.infer_calls),
                   lstm_calls=fc.infer_calls,
                   lstm_train_ms=1e3 * fc.train_time / max(1, fc.train_calls),
                   lstm_train_calls=fc.train_calls)
    if trace:
        res["_trace"] = dict(t=tt.tolist(), q=qs.tolist(), gp=g.tolist(),
                             cwnd=[w for _, w in consumers[0].cwnd_trace][:0],
                             cw_t=[t for t, _ in consumers[0].cwnd_trace],
                             cw_w=[w for _, w in consumers[0].cwnd_trace],
                             pit=pit_r,
                             det=dict(q=dets[0].qhist, theta=None) if dets else None,
                             events=events)
        res["_trace"]["all_q"] = [(d.qhist, d.cap) for d in dets]
        res["_trace"]["faces"] = {"%s->%s" % (d.face.owner.name, d.face.peer[0].name):
                                  dict(q=d.qhist, pred=d.predhist, cap=d.cap) for d in dets}
        if dets:
            res["_trace"]["det_ewma"] = _ewma_series(dets[0].qhist, alpha)
            res["_trace"]["det_face"] = "%s->%s" % (dets[0].face.owner.name,
                                                     dets[0].face.peer[0].name)
            th = []
            hist = deque(maxlen=50)
            for e in res["_trace"]["det_ewma"]:
                hist.append(e)
                mu = np.mean(hist)
                th.append(min(0.8 * dets[0].cap, max(floor_frac * dets[0].cap, mu + kappa * np.std(hist))))
            res["_trace"]["det_theta"] = th
            if fc is not None:
                res["_trace"]["pred"] = dets[0].predhist
                res["_trace"]["H"] = H
    return res


def _ewma_series(q, alpha):
    out, e = [], 0.0
    for v in q:
        e = (1 - alpha) * e + alpha * v
        out.append(e)
    return out


def _features(qh, cap, alpha):
    e = _ewma_series(qh, alpha)
    f = []
    for i, q in enumerate(qh):
        prev = qh[i - 1] if i else q
        f.append([q / cap, e[i] / cap, (q - prev) / cap, 0.0])
    return np.asarray(f)


# ---------------------------------------------------------------------------
# Offline LSTM pre-training on held-out seeds (no overlap with test seeds)
# ---------------------------------------------------------------------------
def pretrain(topo, dt_s=0.1, horizon_s=0.8, seeds=(1001, 1002), epochs=25):
    H = max(1, int(round(horizon_s / dt_s)))
    X, Y = [], []
    for s in seeds:
        for ctrl in ("AIMD", "PID", "Static-16"):
            for scen in ("normal", "load_surge"):
                r = _collect(topo, scen, ctrl, s, dt_s)
                for qh, cap in r:
                    fe = _features(qh, cap, 0.15)
                    for i in range(0, len(qh) - Forecaster.WIN - H + 1):
                        X.append(fe[i:i + Forecaster.WIN])
                        Y.append([(qh[i + Forecaster.WIN + H - 1] - qh[i + Forecaster.WIN - 1]) / cap])
    X, Y = np.asarray(X), np.asarray(Y)
    fc = Forecaster(H)
    rng = np.random.default_rng(0)
    if len(X) > 4000:
        idx = rng.choice(len(X), 4000, replace=False)
        X, Y = X[idx], Y[idx]
    t0 = time.perf_counter()
    losses = fc.model.fit(X, Y, epochs=epochs, batch=32, rng=rng)
    return fc.weights(), [float(x) for x in losses], len(X), time.perf_counter() - t0


def _collect(topo, scen, ctrl, seed, dt_s):
    """Run with detectors only and return raw queue samples of every monitored face."""
    rr = run_one(topo, scen, ctrl, seed, dt_s=dt_s, trace=True)
    return rr["_trace"].get("all_q", [])
