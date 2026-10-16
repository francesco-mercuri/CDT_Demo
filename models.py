"""Model library of the demonstrator: physics, hybrid and data-driven services.

Models are callable plus a declared contract (inputs/outputs as semantic quantity
ids, validity regime, required material properties, measured latency). 
KG stores the contract (the orchestrator never looks inside the callables)
"""
import time
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

import sim
from sim import GRADES, H_NOM, cooling_rate, conduction, gen_cycle, t_end

# lifted datasheet values
# The semantic-lifting step puts true values perturbed by 3 % (datasheet scatter) in the KG
_rs = np.random.default_rng(12345)
DATASHEET = {
    g: {p: v * (1.0 + 0.03 * _rs.uniform(-1, 1)) for p, v in props.items()} for g, props in GRADES.items()
}

ONEHOT = {"A": [1, 0, 0], "B": [0, 1, 0], "C": [0, 0, 1]}


def _oh(g):
    # like OneHotEncoder(handle_unknown="ignore"): an unseen grade silently becomes all zeros
    return ONEHOT.get(g, [0, 0, 0])


def trans_fd(C):
    return 23.0 * (1.0 - np.exp(-(C - 10.0) / 9.0)) if C > 10.0 else 0.0


# callables
def _lumped(inp):
    x = inp["FTP"]
    if np.isnan(x).any():
        return None
    tau = GRADES["A"]["rcp"] * (inp["TH"] / 2000.0) / H_NOM  # nominal-grade lumped capacitance [s]
    a = 1.0 - np.exp(-60.0 / tau)
    Tc = np.empty(len(x))
    Tc[0] = 25.0
    for i in range(1, len(x)):
        Tc[i] = Tc[i - 1] + a * (x[i] - Tc[i - 1])
    C, _ = cooling_rate(Tc)
    return None if not np.isfinite(C) else C


def _fd_core(inp):
    x = inp["FTP"]
    if np.isnan(x).any():
        return None
    p = DATASHEET[inp["MG"]]
    Tc, _ = conduction(x, inp["TH"] / 1000.0, p["k"], p["rcp"], H_NOM)
    C, _ = cooling_rate(Tc)
    return None if not np.isfinite(C) else C


class _Surr:
    def fit(self, X, y):
        self.m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0).fit(X, y)
        return self

    def feat(self, inp):
        y = inp["STP"]
        if np.isnan(y).any():
            return None
        Cs, _ = cooling_rate(y)
        if not np.isfinite(Cs):
            return None
        return [Cs, inp["TH"], inp["PCR"]] + _oh(inp["MG"]), Cs

    def __call__(self, inp):
        f = self.feat(inp)
        if f is None:
            return None
        return float(self.m.predict([f[0]])[0])


def hv_phys_formula(C, Ts):
    return 170.0 + 38.0 * np.log10(1.0 + C) + 0.07 * (Ts - 800.0)


def _hv_phys(inp):
    C = inp["CCR"]
    return None if C is None else float(hv_phys_formula(C, inp["ST"]))


class _Hybrid:
    def fit(self, X, r):
        self.m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=1).fit(X, r)
        return self

    def __call__(self, inp):
        C = inp["CCR"]
        if C is None:
            return None
        x = [C, inp["ST"], inp["HT"], inp["TH"]] + _oh(inp["MG"])
        return float(hv_phys_formula(C, inp["ST"]) + self.m.predict([x])[0])


class _HVml:
    def fit(self, X, y):
        self.m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=2).fit(X, y)
        return self

    def __call__(self, inp):
        y = inp["STP"]
        if np.isnan(y).any():
            return None
        Cs, _ = cooling_rate(y)
        if not np.isfinite(Cs):
            return None
        x = [Cs, inp["ST"], inp["HT"], inp["TH"]] + _oh(inp["MG"])
        return float(self.m.predict([x])[0])


def _hv_fd(inp):
    x = inp["FTP"]
    if np.isnan(x).any():
        return None
    p = DATASHEET[inp["MG"]]
    Tc, Tsf = conduction(x, inp["TH"] / 1000.0, p["k"], p["rcp"], H_NOM)
    Cc, _ = cooling_rate(Tc)
    Cs, _ = cooling_rate(Tsf)
    if not (np.isfinite(Cc) and np.isfinite(Cs)):
        return None
    HI = p["HI"]
    return float(
        175.0 + 150.0 * (HI - 0.5) + 41.0 * np.log10(1.0 + Cc) + 0.088 * (inp["ST"] - 800.0)
        - 0.03 * (inp["HT"] - 60.0) + 13.0 * (HI / 0.5) * np.tanh((Cc - 4.0) / 3.0)
        + 8.0 * np.log(max(Cs, 1e-3) / max(Cc, 1e-3)) + trans_fd(Cc)
    )


# contracts
# validity: {quantity: (min, max)}; grades: allowed grade labels or None; requires: material props
CONTRACTS = {
    "M_core_lumped": dict(inputs=["FTP", "TH"], output="CCR", validity={"PCR": (1.0, 8.0), "TH": (10.0, 25.0)}, grades=None, requires=[]),
    "M_core_FD": dict(inputs=["FTP", "TH", "MG"], output="CCR", validity={"PCR": (1.0, 25.0), "TH": (5.0, 60.0)}, grades=None, requires=["ThermalConductivity", "HeatCapacity"]),
    "M_core_surr": dict(inputs=["STP", "TH", "PCR", "MG"], output="CCR", validity={"PCR": (1.0, 8.0), "TH": (10.0, 40.0)}, grades=["A", "B", "C"], requires=[]),
    "M_HV_phys": dict(inputs=["CCR", "ST"], output="HV", validity={"PCR": (1.0, 8.0)}, grades=None, requires=[]),
    "M_HV_hybrid": dict(inputs=["CCR", "ST", "HT", "TH", "MG"], output="HV", validity={"PCR": (1.0, 8.0), "TH": (10.0, 40.0)}, grades=["A", "B", "C"], requires=[]),
    "M_HV_ml": dict(inputs=["STP", "ST", "HT", "TH", "MG"], output="HV", validity={"PCR": (1.0, 8.0), "TH": (10.0, 40.0)}, grades=["A", "B", "C"], requires=[]),
    "M_HV_FD": dict(inputs=["FTP", "ST", "HT", "TH", "MG"], output="HV", validity={"PCR": (1.0, 25.0), "TH": (5.0, 60.0)}, grades=None, requires=["ThermalConductivity", "HeatCapacity", "HardenabilityIndex"]),
}


def cycle_inputs(c, arrs):
    """Quantities that the *sources* provide for one cycle, with measured (possibly faulted) arrays."""
    te = t_end(c)
    return {
        "TC_A": ("FTP", arrs["TC_A"][: te + 1]),
        "TC_B": ("FTP", arrs["TC_B"][: te + 1]),
        "PYRO": ("STP", arrs["PYRO"][: te + 1]),
        "PYRO_2": ("STP", arrs["PYRO_2"][: te + 1]),
    }


def recipe_inputs(c):
    return {"PCR": c["pcr"], "ST": c["Ts"], "HT": c["th"], "TH": c["d"], "MG": c["grade"]}


class Library:
    def __init__(self, seed=7, n_train=2200, n_cal=500, verbose=True):
        rng = np.random.default_rng(seed)
        t0 = time.time()
        train = [gen_cycle(rng) for _ in range(n_train)]
        self.cal = [gen_cycle(rng) for _ in range(n_cal)]
        if verbose:
            print("generated %d training + %d calibration cycles in %.0fs" % (n_train, n_cal, time.time() - t0))
        # surrogate core model
        Xs, ys = [], []
        for c in train:
            inp = dict(recipe_inputs(c), STP=c["PYRO"][: t_end(c) + 1])
            s = _Surr()
            f = s.feat(inp)
            if f is not None:
                Xs.append(f[0]); ys.append(c["Cc"])
        self.surr = _Surr().fit(np.array(Xs), np.array(ys))
        # hybrid residual model and pure-ML hardness model
        Xh, rh, Xm, ym = [], [], [], []
        for c in train:
            C = c["Cc"] * (1.0 + rng.normal(0, 0.03))
            Xh.append([C, c["Ts"], c["th"], c["d"]] + _oh(c["grade"]))
            rh.append(c["HV"] - hv_phys_formula(C, c["Ts"]))
            Cs, _ = cooling_rate(c["PYRO"][: t_end(c) + 1])
            if np.isfinite(Cs):
                Xm.append([Cs, c["Ts"], c["th"], c["d"]] + _oh(c["grade"])); ym.append(c["HV"])
        self.hybrid = _Hybrid().fit(np.array(Xh), np.array(rh))
        self.hvml = _HVml().fit(np.array(Xm), np.array(ym))
        self.fn = {
            "M_core_lumped": _lumped, "M_core_FD": _fd_core, "M_core_surr": self.surr,
            "M_HV_phys": _hv_phys, "M_HV_hybrid": self.hybrid, "M_HV_ml": self.hvml, "M_HV_FD": _hv_fd,
        }
        self.contracts = {k: dict(v) for k, v in CONTRACTS.items()}
        self._measure_latency()

    # --- execution of a chain template on concrete sources
    def run_chain(self, steps, binding, c, arrs):
        """steps: tuple of model names in execution order; binding: dict quantity->source name."""
        data = recipe_inputs(c)
        raw = cycle_inputs(c, arrs)
        for q, src in binding.items():
            if src in raw:  # recipe / MES quantities are already in `data`
                data[raw[src][0]] = raw[src][1]
        for m in steps:
            inp = {q: data.get(q) for q in self.contracts[m]["inputs"]}
            if any(v is None for v in inp.values()):
                return None
            data_out = self.fn[m](inp)
            if data_out is None:
                return None
            data[self.contracts[m]["output"]] = data_out
        return data.get("HV")

    def _measure_latency(self):
        lat = {m: [] for m in self.fn}
        for c in self.cal[:40]:
            arrs = {k: c[k] for k in ("TC_A", "TC_B", "PYRO", "PYRO_2")}
            data = recipe_inputs(c)
            te = t_end(c)
            data.update(FTP=arrs["TC_A"][: te + 1], STP=arrs["PYRO"][: te + 1], CCR=c["Cc"])
            for m, f in self.fn.items():
                inp = {q: data[q] for q in self.contracts[m]["inputs"]}
                t0 = time.perf_counter(); f(inp); lat[m].append(time.perf_counter() - t0)
        for m in lat:
            self.contracts[m]["latency"] = float(np.median(lat[m]))
