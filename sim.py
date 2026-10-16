"""Virtual annealing furnace - synthetic ground truth for the cognitive-DT demonstrator.

1D transient heat-conduction solution through a slab (implicit finite differences) 
driven by a furnace-gas temperature profile, and an empirical hardness law. 
Hidden per-cycle variability (convective coefficient, cooling-rate
disturbance, sensor bias/noise) makes the planned recipe an imperfect predictor, so that
measurements and models actually matter.
"""
import numpy as np
from scipy.linalg import solve_banded

# thermal properties (k [W/mK], rho*cp [J/m3K]) and hardenability index (HI, dimensionless).
# 'D' is the grade that is *new* at deployment time (never used in model training).
GRADES = {
    "A": dict(k=30.0, rcp=3.80e6, HI=0.50),
    "B": dict(k=38.0, rcp=3.70e6, HI=0.62),
    "C": dict(k=24.0, rcp=3.90e6, HI=0.40),
    "D": dict(k=18.0, rcp=4.10e6, HI=0.78),
}
H_NOM = 350.0  # nominal convective coefficient assumed by models [W/m2K]
TRAIN_GRADES = ["A", "B", "C"]
PCR_IN = (1.0, 8.0)  # planned cooling rate regime used for training [K/min]
TH_IN = (10.0, 40.0)  # thickness regime [mm]
TOL_HV = 10.0  # tolerance defining a "useful" hardness prediction [HV]


def conduction(Tf, d_m, k, rcp, h, N=16, dt=60.0, T0=25.0, substeps=1):
    """Implicit FD for a symmetric slab of thickness d_m with convective BC to gas temperature Tf.

    Tf: gas temperature at each sample (1 sample per `dt*substeps` seconds). Returns (T_core, T_surf)
    arrays with the same length as Tf.
    """
    L = d_m / 2.0
    dx = L / N
    n = N + 1
    ab = np.zeros((3, n))
    a = k / (rcp * dx * dx)
    # interior rows
    diag = np.full(n, 1.0 + 2.0 * a * dt)
    upper = np.full(n - 1, -a * dt)
    lower = np.full(n - 1, -a * dt)
    # centre (half cell, symmetry)
    diag[0] = 1.0 + 2.0 * a * dt
    upper[0] = -2.0 * a * dt
    # surface (half cell, convection)
    diag[-1] = 1.0 + dt * (2.0 * a + 2.0 * h / (rcp * dx))
    lower[-1] = -2.0 * a * dt
    ab[0, 1:] = upper
    ab[1, :] = diag
    ab[2, :-1] = lower
    T = np.full(n, T0)
    Tc = np.empty(len(Tf))
    Ts = np.empty(len(Tf))
    Tc[0], Ts[0] = T[0], T[-1]
    for i in range(1, len(Tf)):
        rhs = T.copy()
        rhs[-1] += dt * 2.0 * h / (rcp * dx) * Tf[i]
        T = solve_banded((1, 1), ab, rhs)
        Tc[i], Ts[i] = T[0], T[-1]
    return Tc, Ts


def furnace_profile(Ts, th, c_set, f_cool, rng):
    """Gas temperature [C] at 1 min resolution: ramp 8 K/min, hold, controlled cooling, tail at 400 C."""
    t_heat = (Ts - 25.0) / 8.0
    t_cool = (Ts - 400.0) / c_set
    n = int(t_heat + th + t_cool / min(f_cool, 1.0) + 150) + 2
    t = np.arange(n, dtype=float)
    c_act = c_set * f_cool
    t_cool_act = (Ts - 400.0) / c_act
    sp = np.where(
        t < t_heat,
        25.0 + 8.0 * t,
        np.where(
            t < t_heat + th,
            Ts,
            np.where(t < t_heat + th + t_cool_act, Ts - c_act * (t - t_heat - th), 400.0),
        ),
    )
    Tf = np.empty(n)
    Tf[0] = 25.0
    a = 1.0 - np.exp(-1.0 / 3.0)  # furnace-gas lag tau = 3 min
    for i in range(1, n):
        Tf[i] = Tf[i - 1] + a * (sp[i] - Tf[i - 1])
    return Tf


def cooling_rate(T, hi=700.0, lo=500.0, i0=0):
    """Mean cooling rate between `hi` and `lo` [K/min] on a 1-min trajectory (after sample i0).

    Returns (rate, index_of_lo_crossing) or (nan, -1)."""
    T = np.asarray(T)
    imax = i0 + int(np.argmax(T[i0:]))
    idx_hi = None
    for i in range(imax, len(T) - 1):
        if T[i] >= hi > T[i + 1]:
            idx_hi = i + (T[i] - hi) / (T[i] - T[i + 1])
            break
    if idx_hi is None:
        return np.nan, -1
    for i in range(int(idx_hi), len(T) - 1):
        if T[i] >= lo > T[i + 1]:
            idx_lo = i + (T[i] - lo) / (T[i] - T[i + 1])
            return (hi - lo) / (idx_lo - idx_hi), i + 1
    return np.nan, -1


def hardness_truth(Cc, Cs, Ts, th, grade, rng=None):
    HI = GRADES[grade]["HI"]
    hv = (
        175.0
        + 150.0 * (HI - 0.5)
        + 42.0 * np.log10(1.0 + Cc)
        + 0.09 * (Ts - 800.0)
        - 0.03 * (th - 60.0)
        + 14.0 * (HI / 0.5) * np.tanh((Cc - 4.0) / 3.0)
        + 8.0 * np.log(max(Cs, 1e-3) / max(Cc, 1e-3))
        + (25.0 * (1.0 - np.exp(-(Cc - 10.0) / 8.0)) if Cc > 10.0 else 0.0)
    )
    if rng is not None:
        hv += rng.normal(0.0, 2.0)
    return hv


def gen_cycle(rng, grade=None, regime="in", d_mm=None):
    """Simulate one annealing cycle; return context, clean sensor arrays and truth."""
    if grade is None:
        grade = rng.choice(TRAIN_GRADES)
    d = d_mm if d_mm is not None else rng.uniform(*TH_IN)
    Ts = rng.uniform(760.0, 880.0)
    th = rng.uniform(30.0, 120.0)
    pcr = rng.uniform(*PCR_IN) if regime == "in" else rng.uniform(10.0, 25.0)
    f_cool = rng.uniform(0.65, 1.35)  # uncontrolled cooling disturbance
    h = rng.uniform(250.0, 500.0)  # hidden convective coefficient
    Tf = furnace_profile(Ts, th, pcr, f_cool, rng)
    g = GRADES[grade]
    Tc, Tsf = conduction(Tf, d / 1000.0, g["k"], g["rcp"], h)
    Cc, i500 = cooling_rate(Tc)
    Cs, _ = cooling_rate(Tsf)
    if not np.isfinite(Cc) or not np.isfinite(Cs):
        return gen_cycle(rng, grade, regime, d_mm)
    n = len(Tf)
    # sensors (1/min): two redundant thermocouples (gas), two pyrometers (surface)
    TC_A = Tf + rng.normal(0, 2.0, n)
    TC_B = Tf + rng.normal(0, 2.0, n)
    biasP = rng.normal(0, 6.0)
    PYRO = Tsf + biasP + rng.normal(0, 4.0, n)
    PYRO_2 = Tsf + rng.normal(0, 6.0) + rng.normal(0, 4.0, n)
    hv = hardness_truth(Cc, Cs, Ts, th, grade, rng)
    return dict(
        grade=grade, d=d, Ts=Ts, th=th, pcr=pcr, f_cool=f_cool, h=h,
        Tf=Tf, Tcore=Tc, Tsurf=Tsf, Cc=Cc, Cs=Cs, t_dec=i500, HV=hv,
        TC_A=TC_A, TC_B=TC_B, PYRO=PYRO, PYRO_2=PYRO_2,
    )


# faults & health monitor
FAULTS = ("dropout", "stuck", "drift")


def inject_fault(arr, kind, onset):
    a = arr.copy()
    if kind == "dropout":
        a[onset:] = np.nan
    elif kind == "stuck":
        a[onset:] = a[onset - 1]
    elif kind == "drift":
        a[onset:] += 1.0 * (np.arange(onset, len(a)) - onset + 1)  # +1 K/min
    return a


def monitor(sig, t_dec, partner=None):
    """Rule-based health monitor on data up to t_dec. Returns 'ok'|'failed'|'suspect'.

    dropout: NaN for >=3 samples; stuck: zero variance for >=10 samples;
    drift/disagreement (needs a redundant partner): |a-b|>15 K for >=5 samples."""
    x = sig[: t_dec + 1]
    if np.isnan(x[-3:]).any():
        return "failed"
    w = x[-10:]
    if len(w) == 10 and np.ptp(w) < 1e-6:
        return "failed"
    if partner is not None:
        diff = np.abs(x - partner[: t_dec + 1])
        run = 0
        for v in diff:
            run = run + 1 if v > 15.0 else 0
            if run >= 5:
                return "suspect"
    return "ok"


def t_end(c):
    """Last sample available to the models (core crosses 500 C at t_dec; +10 min margin)."""
    return min(c["t_dec"] + 10, len(c["Tf"]) - 1)


def apply_scenario(c, rng, n_faults):
    """Inject n_faults independent faults on distinct sensors among TC_A, TC_B, PYRO. Returns
    measured arrays, statuses (ground-truth monitor output) and a description."""
    arrs = {k: c[k].copy() for k in ("TC_A", "TC_B", "PYRO", "PYRO_2")}
    desc = []
    t_dec = c["t_dec"]
    te = t_end(c)
    sensors = list(rng.choice(["TC_A", "TC_B", "PYRO"], size=n_faults, replace=False)) if n_faults else []
    for s in sensors:
        kind = rng.choice(["dropout", "stuck"] if s == "PYRO" else list(FAULTS))
        onset = int(rng.uniform(0.55, 0.97) * t_dec)
        arrs[s] = inject_fault(arrs[s], kind, onset)
        desc.append((s, kind, onset))
    status = {
        "TC_A": monitor(arrs["TC_A"], te, arrs["TC_B"]),
        "TC_B": monitor(arrs["TC_B"], te, arrs["TC_A"]),
        "PYRO": monitor(arrs["PYRO"], te),
        "PYRO_2": monitor(arrs["PYRO_2"], te),
    }
    # disagreement flags both thermocouples; unflagged TC with 'suspect' partner is also unusable
    return arrs, status, desc


if __name__ == "__main__":
    import time

    rng = np.random.default_rng(0)
    t0 = time.time()
    cs = [gen_cycle(rng) for _ in range(40)]
    print("time/cycle %.1f ms" % ((time.time() - t0) / 40 * 1e3))
    Cc = np.array([c["Cc"] for c in cs]); pcr = np.array([c["pcr"] for c in cs]); hv = np.array([c["HV"] for c in cs])
    print("Cc/pcr ratio  mean %.2f  min %.2f max %.2f" % ((Cc / pcr).mean(), (Cc / pcr).min(), (Cc / pcr).max()))
    print("HV  mean %.1f std %.1f min %.1f max %.1f" % (hv.mean(), hv.std(), hv.min(), hv.max()))
    print("corr(pcr,Cc)=%.3f" % np.corrcoef(pcr, Cc)[0, 1])
    out = [gen_cycle(rng, regime="out") for _ in range(20)]
    print("OUT regime Cc range", min(c["Cc"] for c in out), max(c["Cc"] for c in out))
    d = [gen_cycle(rng, grade="D") for _ in range(20)]
    print("grade D HV mean %.1f" % np.mean([c["HV"] for c in d]))
