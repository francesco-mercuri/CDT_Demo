"""E2: validation gates for LLM-lifted contracts. Corruptions are injected parametrically (LLM errors are simulated,
not sampled from a real LLM); gates are real code; downstream harm is measured with the real orchestrator logic."""
import pickle, copy, json, numpy as np, pandas as pd
import sim, kg
from sim import TOL_HV
from exp_run import Cyc, lib, wf as _wf
wf = {s: e for s, e in _wf.values()}

QUANT = {'FTP','STP','TH','PCR','ST','HT','MG','CCR','HV'}
GRADESET = {'A','B','C','D'}
PHYS = {'PCR': (0, 50.0), 'TH': (0, 100.0)}
CLASS_OUT = {'M_core_lumped':'CCR','M_core_FD':'CCR','M_core_surr':'CCR','M_HV_phys':'HV','M_HV_hybrid':'HV','M_HV_ml':'HV','M_HV_FD':'HV'}
TYPES = ['schema','unit','swap_out','widen','drop_grade','hall_input']
SYSTEMATIC = {'schema':0.3,'unit':0.5,'swap_out':0.6,'widen':0.8,'drop_grade':0.8,'hall_input':0.6}  # P(same wrong answer in every sample)

def corrupt(c, t, rng):
    c = copy.deepcopy(c)
    if t == 'schema': c['inputs'] = list(c['inputs']) + ['TempCore']
    elif t == 'unit':
        q = rng.choice(list(c['validity'])) if c['validity'] else None
        if q: lo, hi = c['validity'][q]; c['validity'][q] = (lo, hi*60.0 if q=='PCR' else hi*10.0)
    elif t == 'swap_out': c['output'] = 'HV' if c['output']=='CCR' else 'CCR'
    elif t == 'widen':
        q = rng.choice(list(c['validity'])) if c['validity'] else None
        if q: lo, hi = c['validity'][q]; c['validity'][q] = (lo, hi*2.0)
    elif t == 'drop_grade': c['grades'] = None
    elif t == 'hall_input': c['inputs'] = list(c['inputs']) + ['HV'] if c['output']=='CCR' else list(c['inputs'])+['CCR']
    return c

def g1_schema(name, c):
    if not set(c['inputs']) <= QUANT or c['output'] not in QUANT: return False
    for q,(lo,hi) in c['validity'].items():
        if q not in QUANT or not lo < hi: return False
    return c['grades'] is None or set(c['grades']) <= GRADESET

def g2_rules(name, c):
    if c['output'] != CLASS_OUT[name]: return False          # quantity class vs. model class
    if c['output'] in c['inputs']: return False                # no cycles / self-dependence
    if c['output']=='CCR' and 'HV' in c['inputs']: return False
    for q,(lo,hi) in c['validity'].items():
        if q in PHYS and hi > PHYS[q][1]: return False         # physical bounds
    return True

def gates(contracts, level, rng, k=5, q_ind=0.0, audit_frac=0.3, truth=None, kinds=None):
    """Return the accepted contract set. level in G0..G4. kinds: dict name->error type or None."""
    acc = {}
    for name, c in contracts.items():
        ok = True
        flagged = False
        if level >= 1 and not g1_schema(name, c): ok = False
        if ok and level >= 2 and not g2_rules(name, c): ok = False
        if ok and level >= 3 and kinds.get(name):
            # self-consistency: with prob (1-systematic) the k samples disagree -> flagged for audit
            if rng.random() > SYSTEMATIC[kinds[name]]: flagged = True
        if ok and level >= 4:
            if flagged or rng.random() < audit_frac:
                if kinds.get(name): ok = False                 # reviewer catches
        elif ok and flagged and level == 3:
            ok = False                                          # G3 alone: discard disagreeing contracts
        if ok: acc[name] = c
    return acc

def make_cycles(n_each, seed):
    rng = np.random.default_rng(seed)
    cys = []
    for regime, grade in (('in',None),('out',None),('in','D')):
        for _ in range(n_each):
            c = sim.gen_cycle(rng, grade=grade, regime=regime)
            arrs, status, _ = sim.apply_scenario(c, rng, 1)
            cys.append(Cyc(c, arrs, status))
    return cys

def evaluate(cons, cys, regime_aware=True):
    dl = us = ha = 0
    for cy in cys:
        c = cy.c
        ch = kg.select_py(cons, wf, cy.status, {'PCR': c['pcr'], 'TH': c['d']}, c['grade'], sources=None)
        v = None if ch is None else cy.out(*ch)
        if v is not None:
            dl += 1; e = abs(v - c['HV']); us += e <= TOL_HV; ha += e > TOL_HV
    n = len(cys)
    return dl/n, us/n, ha/n

if __name__ == '__main__':
    rng = np.random.default_rng(55)
    cys = make_cycles(25, 77)
    clean = {m: dict(c) for m, c in lib.contracts.items()}
    base = evaluate(clean, cys); print('clean contracts: deliver %.2f useful %.2f harmful %.2f' % base)
    # detectability by error type (single injected error, 400 trials each)
    names = list(clean)
    det = []
    for t in TYPES:
        cnt = {1:0, 2:0, 3:0, 4:0}; n_eff = 0
        for _ in range(400):
            m = rng.choice(names); bad = corrupt(clean[m], t, rng)
            if bad == clean[m]: continue
            n_eff += 1
            con = dict(clean); con[m] = bad
            for lv in (1,2,3,4):
                acc = gates(con, lv, rng, kinds={m: t})
                if m not in acc: cnt[lv] += 1
        det.append(dict(error=t, **{'G%d' % lv: cnt[lv]/n_eff for lv in cnt}))
    det = pd.DataFrame(det); print(det.round(3).to_string()); det.to_csv('results/e2_detect.csv', index=False)
    #  downstream impact, contract-level error rate p
    rows = []
    for p in (0.1, 0.2, 0.3):
        for trial in range(120):
            con = dict(clean); kinds = {}
            for m in names:
                if rng.random() < p:
                    t = TYPES[rng.integers(len(TYPES))]; con[m] = corrupt(clean[m], t, rng); kinds[m] = t
            for lv in (0,1,2,3,4):
                acc = gates(con, lv, rng, kinds=kinds)
                d, u, h = evaluate(acc, cys)
                rows.append(dict(p=p, gate='G%d' % lv, trial=trial, deliver=d, useful=u, harmful=h,
                                 contaminated=sum(1 for m in acc if kinds.get(m))/max(len(acc),1) , n_acc=len(acc)))
        print('p', p, 'done', flush=True)
    r = pd.DataFrame(rows); r.to_csv('results/e2_downstream.csv', index=False)
    print(r.groupby(['p','gate'])[['deliver','useful','harmful']].mean().round(3).to_string())
