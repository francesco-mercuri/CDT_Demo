"""E3: SPARQL/mirror equivalence on simulated cycles, query latency vs KG size, detection-to-replan latency."""
import time, numpy as np, pandas as pd, sim, kg
from exp_run import lib, wf as _wf, Cyc
wf = {s: e for s, e in _wf.values()}
rng = np.random.default_rng(404)
# (a) equivalence on 240 cycles (in/out regime, grade D, 0-2 faults)
g = kg.build_kg(lib, _wf); orch = kg.KGOrchestrator(g).refresh()
agree = n = 0; bad = []
for i in range(240):
    regime = 'out' if i % 2 else 'in'; grade = 'D' if i % 5 == 0 else None
    c = sim.gen_cycle(rng, grade=grade, regime=regime); arrs, st, _ = sim.apply_scenario(c, rng, i % 3)
    st = {k: v for k, v in st.items() if k != 'PYRO_2'}
    kg.add_run(g, 'R', 'Furnace1', c, st)
    a = orch.select('R')['choice']; kg.remove_run(g, 'R')
    for s in st: g.set((kg.DT[s], kg.DT.status, kg.Literal('ok')))
    b = kg.select_py(lib.contracts, wf, st, {'PCR': c['pcr'], 'TH': c['d']}, c['grade'])
    n += 1; agree += (a == b)
print('equivalence SPARQL vs mirror: %d/%d' % (agree, n))
# (b) latency vs KG size
rows = []
c = sim.gen_cycle(np.random.default_rng(5)); st = {'TC_A': 'failed', 'TC_B': 'ok', 'PYRO': 'ok'}
for nf in (0, 10, 50, 200, 500):
    g = kg.build_kg(lib, _wf, extra_furnaces=nf); o = kg.KGOrchestrator(g)
    t0 = time.perf_counter(); o.refresh(); t_ref = time.perf_counter() - t0
    kg.add_run(g, 'R', 'Furnace1', c, st)
    ts = []
    for _ in range(15):
        t0 = time.perf_counter(); o.select('R'); ts.append(time.perf_counter() - t0)
    rows.append(dict(furnaces=nf + 1, triples=len(g), refresh_ms=t_ref * 1e3, select_ms_med=np.median(ts) * 1e3,
                     select_ms_p95=np.percentile(ts, 95) * 1e3))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv('results/e3_scaling.csv', index=False)
# (c) detection-to-replan latency: samples (=minutes) between fault onset and a valid new plan
rows = []
for k in range(150):
    c = sim.gen_cycle(rng); onset = int(0.7 * c['t_dec'])
    kind = ['dropout', 'stuck'][k % 2]
    a = sim.inject_fault(c['TC_A'], kind, onset)
    t_det = None
    for t in range(onset, len(a)):
        if sim.monitor(a, t) == 'failed': t_det = t; break
    rows.append(dict(kind=kind, detect_min=None if t_det is None else t_det - onset, margin_min=c['t_dec'] - (t_det if t_det else onset)))
d = pd.DataFrame(rows); d.to_csv('results/e3_replan.csv', index=False)
print(d.groupby('kind')[['detect_min', 'margin_min']].describe().round(1).T.to_string())
print('model latencies [ms]:', {m: round(v['latency'] * 1e3, 2) for m, v in lib.contracts.items()})
