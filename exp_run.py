"""E1 (fault/regime scenarios), E4a (new grade D), E4b (new sensor PYRO_2 registered by triples only)."""
import sys, pickle, time, json
import numpy as np, pandas as pd
import sim, kg
from sim import TOL_HV

lib, wf = pickle.load(open('lib.pkl', 'rb'))
ALL = {k: {'inputs': v['inputs'], 'output': v['output']} for k, v in lib.contracts.items()}
table = kg.contract_table(lib)
LP, LH = 'M_core_lumped', 'M_HV_hybrid'
FD, HVFD, SURR, PHYS = 'M_core_FD', 'M_HV_FD', 'M_core_surr', 'M_HV_phys'


class Cyc:
    def __init__(self, c, arrs, status):
        self.c, self.arrs, self.status, self.memo = c, arrs, status, {}

    def out(self, steps, binding):
        key = (steps, kg.sensor_key(binding))
        if key not in self.memo:
            self.memo[key] = lib.run_chain(steps, binding, self.c, self.arrs)
        return self.memo[key]


def first_ok(cy, options):
    for src, steps, q in options:
        if cy.status[src] == 'ok':
            return steps, {q: src}
    return None


def strat_B0(cy):
    return ((LP, LH), {'FTP': 'TC_A'}) if cy.status['TC_A'] == 'ok' else None


def strat_B1(cy):
    return first_ok(cy, [('TC_A', (LP, LH), 'FTP'), ('TC_B', (LP, LH), 'FTP'), ('PYRO', (SURR, LH), 'STP')])


def strat_B2(cy):
    c = cy.c
    inreg = c['pcr'] <= 8.0 and 10.0 <= c['d'] <= 40.0  # regimes known at design time
    if inreg:
        return first_ok(cy, [('TC_A', (FD, LH), 'FTP'), ('TC_B', (FD, LH), 'FTP'), ('PYRO', (SURR, LH), 'STP')])
    return first_ok(cy, [('TC_A', (HVFD,), 'FTP'), ('TC_B', (HVFD,), 'FTP')])


def oracle(cy, extra_sources=()):
    avail = {'FTP': [s for s in ('TC_A', 'TC_B') if cy.status[s] == 'ok'],
             'STP': [s for s in ('PYRO',) + tuple(extra_sources) if cy.status[s] == 'ok'],
             'PCR': ['PLC'], 'ST': ['PLC'], 'HT': ['PLC'], 'TH': ['MES'], 'MG': ['MES']}
    best = None
    for steps, binding in kg.compose('HV', ALL, avail):
        v = cy.out(steps, binding)
        if v is not None:
            e = abs(v - cy.c['HV'])
            if best is None or e < best[0]:
                best = (e, steps, binding)
    return None if best is None else (best[1], best[2])


def run_cycles(specs, seed, tag, pyro2=False):
    rng = np.random.default_rng(seed)
    g = kg.build_kg(lib, wf)
    if pyro2:
        kg.add_source(g, 'PYRO_2', 'Sensor', ['STP'])
    orch = kg.KGOrchestrator(g).refresh()
    orch_noreg = orch
    rows = []
    t0 = time.time()
    for (regime, grade, nf, count, label) in specs:
        for i in range(count):
            c = sim.gen_cycle(rng, grade=grade, regime=regime)
            arrs, status, desc = sim.apply_scenario(c, rng, nf)
            cy = Cyc(c, arrs, status)
            st_kg = dict(status)
            if not pyro2:
                st_kg.pop('PYRO_2', None)
            kg.add_run(g, 'R', 'Furnace1', c, st_kg)
            s_full = orch.select('R', budget=1.0, use_regime=True)
            s_nor = orch.select('R', budget=1.0, use_regime=False)
            s_tight = orch.select('R', budget=0.002, use_regime=True)
            kg.remove_run(g, 'R')
            for s in st_kg:
                g.set((kg.DT[s], kg.DT.status, kg.Literal('ok')))
            extra = ('PYRO_2',) if pyro2 else ()
            choices = {'B0': strat_B0(cy), 'B1': strat_B1(cy), 'B2': strat_B2(cy),
                       'S': s_full['choice'], 'S_noregime': s_nor['choice'], 'S_tight': s_tight['choice'],
                       'Oracle': oracle(cy, extra)}
            for name, ch in choices.items():
                v = None if ch is None else cy.out(*ch)
                err = None if v is None else v - c['HV']
                rows.append(dict(tag=tag, label=label, regime=regime, grade=c['grade'], nf=nf, strategy=name,
                                 delivered=v is not None, err=err,
                                 useful=(v is not None and abs(err) <= TOL_HV),
                                 harmful=(v is not None and abs(err) > TOL_HV),
                                 steps='' if ch is None else '>'.join(x.replace('M_', '') for x in ch[0]),
                                 t_query=s_full['t'] if name == 'S' else None,
                                 faults=json.dumps([(a, b) for a, b, _ in desc])))
        print(tag, label, 'done', '%.0fs' % (time.time() - t0), flush=True)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'e1':
        specs = [(r, None, nf, 150, '%s-regime, %d faults' % (r, nf)) for r in ('in', 'out') for nf in (0, 1, 2)]
        run_cycles(specs, 101, 'E1').to_csv('results/e1.csv', index=False)
    else:
        a = run_cycles([('in', 'D', 0, 100, 'new grade D, 0 faults'), ('in', 'D', 1, 100, 'new grade D, 1 fault')], 202, 'E4a')
        b0 = run_cycles([('in', None, 3, 100, 'all three sensors faulty')], 303, 'E4b_noreg', pyro2=False)
        b1 = run_cycles([('in', None, 3, 100, 'all three sensors faulty')], 303, 'E4b_pyro2', pyro2=True)
        pd.concat([a, b0, b1]).to_csv('results/e4.csv', index=False)
