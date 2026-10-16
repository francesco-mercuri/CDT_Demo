"""Knowledge graph, semantic contracts, SPARQL-based discovery and chain composition."""
import itertools
import time
import numpy as np
from rdflib import Graph, Namespace, Literal, RDF, BNode, URIRef
from rdflib.namespace import XSD

import sim
from models import DATASHEET

DT = Namespace("http://example.org/dt#")
RUN = Namespace("http://example.org/run#")
QUANTITIES = ["FTP", "STP", "CCR", "PCR", "ST", "HT", "TH", "MG", "HV"]
PROPS = {"ThermalConductivity": "k", "HeatCapacity": "rcp", "HardenabilityIndex": "HI"}
SOURCES = {  # source -> (kind, provided quantities)
    "TC_A": ("Sensor", ["FTP"]), "TC_B": ("Sensor", ["FTP"]), "PYRO": ("Sensor", ["STP"]),
    "PLC": ("RecipeInterface", ["PCR", "ST", "HT"]), "MES": ("EnterpriseSystem", ["TH", "MG"]),
}
GRADE_LIST = ["A", "B", "C", "D"]


def n(x):
    return str(x).split("#")[-1]


# composition
def compose(target, models, avail):
    """All derivations of `target` as (steps, binding): steps = ordered model names, binding = quantity->source.

    models: name -> {'inputs': [...], 'output': q}; avail: quantity -> [sources]."""
    def resolve(q, stack):
        res = []
        for s in avail.get(q, []):
            res.append(((), {q: s}))
        for m, c in models.items():
            if c["output"] == q and m not in stack:
                opts = [resolve(i, stack + (m,)) for i in c["inputs"]]
                if all(opts):
                    for combo in itertools.product(*opts):
                        steps, binding, ok = (), {}, True
                        for st, b in combo:
                            for s_ in st:
                                if s_ not in steps:
                                    steps += (s_,)
                            for k, v in b.items():
                                if binding.get(k, v) != v:
                                    ok = False
                                binding[k] = v
                        if ok:
                            res.append((steps + (m,), binding))
        return res

    out = [(st, b) for st, b in resolve(target, ()) if st]
    # keep only sensor bindings in the key (recipe/MES quantities have a single source)
    return out


def sensor_key(binding):
    return (binding.get("FTP"), binding.get("STP"))


# KG construction
def add_contract(g, name, c):
    m = DT[name]
    g.add((m, RDF.type, DT.Model))
    g.add((m, DT.produces, DT[c["output"]]))
    for q in c["inputs"]:
        g.add((m, DT.hasInput, DT[q]))
    g.add((m, DT.latency, Literal(float(c["latency"]), datatype=XSD.double)))
    for q, (lo, hi) in c["validity"].items():
        v = BNode()
        g.add((m, DT.hasValidity, v))
        g.add((v, DT.onQuantity, DT[q]))
        g.add((v, DT.min, Literal(float(lo), datatype=XSD.double)))
        g.add((v, DT.max, Literal(float(hi), datatype=XSD.double)))
    for p in c["requires"]:
        g.add((m, DT.requiresProperty, DT[p]))
    for gr in (c["grades"] or []):
        g.add((m, DT.validForGrade, DT["Grade_" + gr]))


def add_workflow(g, wid, steps, err, cal="cal_v1"):
    wf = DT[wid]
    g.add((wf, RDF.type, DT.CompositeWorkflow))
    g.add((wf, DT.calibratedErrorHV, Literal(float(err), datatype=XSD.double)))
    g.add((wf, DT.calibrationSet, Literal(cal)))
    for i, m in enumerate(steps, 1):
        s = BNode()
        g.add((wf, DT.hasStep, s)); g.add((s, DT.position, Literal(i))); g.add((s, DT.model, DT[m]))


def add_source(g, src, kind, provides, furnace="Furnace1"):
    s = DT[src]
    g.add((s, RDF.type, DT[kind]))
    g.add((s, DT.installedOn, DT[furnace]))
    g.add((s, DT.status, Literal("ok")))
    for q in provides:
        g.add((s, DT.provides, DT[q]))


def build_kg(lib, workflows=None, extra_furnaces=0, grades=GRADE_LIST):
    g = Graph()
    g.bind("dt", DT)
    for q in QUANTITIES:
        g.add((DT[q], RDF.type, DT.Quantity))
    for p in PROPS:
        g.add((DT[p], RDF.type, DT.MaterialProperty))
    for gr in grades:
        gi = DT["Grade_" + gr]
        g.add((gi, RDF.type, DT.Grade))
        for p, key in PROPS.items():
            g.add((gi, DT.hasProperty, DT[p]))
            g.add((gi, DT[p[0].lower() + p[1:] + "Value"], Literal(float(DATASHEET[gr][key]), datatype=XSD.double)))
    for f in ["Furnace1"] + ["Furnace%d" % (i + 2) for i in range(extra_furnaces)]:
        g.add((DT[f], RDF.type, DT.Furnace))
        for src, (kind, prov) in SOURCES.items():
            add_source(g, src + ("" if f == "Furnace1" else "_" + f), kind, prov, f)
    for name, c in lib.contracts.items():
        add_contract(g, name, c)
    if workflows:
        for wid, (steps, err) in workflows.items():
            add_workflow(g, wid, steps, err)
    return g


def contract_table(lib):
    return {k: dict(v) for k, v in lib.contracts.items()}


def calibrate_workflows(lib, models=None):
    """Registration pipeline: enumerate every chain template for HV and calibrate end-to-end on the
    in-regime calibration set with healthy sensors. Returns {wf_id: (steps, RMSE_HV)}."""
    models = models or {k: {"inputs": v["inputs"], "output": v["output"]} for k, v in lib.contracts.items()}
    avail = {"FTP": ["TC_A"], "STP": ["PYRO"], "PCR": ["PLC"], "ST": ["PLC"], "HT": ["PLC"], "TH": ["MES"], "MG": ["MES"]}
    out, seen = {}, set()
    for steps, binding in compose("HV", models, avail):
        if steps in seen:
            continue
        seen.add(steps)
        errs = []
        for c in lib.cal:
            arrs = {k: c[k] for k in ("TC_A", "TC_B", "PYRO", "PYRO_2")}
            v = lib.run_chain(steps, binding, c, arrs)
            if v is not None:
                errs.append(v - c["HV"])
        out["WF_" + "_".join(s.replace("M_", "") for s in steps)] = (steps, float(np.sqrt(np.mean(np.square(errs)))))
    return out


# run context in the KG
def add_run(g, rid, furnace, c, status):
    r = RUN[rid]
    g.add((r, RDF.type, DT.Run)); g.add((r, DT.onFurnace, DT[furnace])); g.add((r, DT.grade, DT["Grade_" + c["grade"]]))
    for q, val in (("PCR", c["pcr"]), ("TH", c["d"])):
        b = BNode()
        g.add((r, DT.context, b)); g.add((b, DT.onQuantity, DT[q])); g.add((b, DT.value, Literal(float(val), datatype=XSD.double)))
    for src, st in status.items():
        s = DT[src + ("" if furnace == "Furnace1" else "_" + furnace)]
        if (s, DT.status, None) in g:
            g.set((s, DT.status, Literal(st)))
    return r


def remove_run(g, rid):
    r = RUN[rid]
    for b in list(g.objects(r, DT.context)):
        g.remove((b, None, None))
    g.remove((r, None, None))


# SPARQL orchestrator
Q_AVAIL = """PREFIX dt: <http://example.org/dt#>
SELECT ?q ?src WHERE {{ <{run}> dt:onFurnace ?f . ?src dt:installedOn ?f ; dt:provides ?q ; dt:status "ok" . }}"""

Q_MODELS = """PREFIX dt: <http://example.org/dt#>
SELECT ?m ?out ?lat (GROUP_CONCAT(DISTINCT STR(?in); separator=",") AS ?ins) WHERE {{
  ?m a dt:Model ; dt:produces ?out ; dt:latency ?lat ; dt:hasInput ?in .
  <{run}> dt:grade ?g .
  {regime}
  FILTER NOT EXISTS {{ ?m dt:requiresProperty ?p . FILTER NOT EXISTS {{ ?g dt:hasProperty ?p }} }}
  FILTER (?lat <= {budget})
}} GROUP BY ?m ?out ?lat"""

REGIME = """
  FILTER NOT EXISTS {{ ?m dt:hasValidity ?v . ?v dt:onQuantity ?cq ; dt:min ?lo ; dt:max ?hi .
     <{run}> dt:context ?c . ?c dt:onQuantity ?cq ; dt:value ?val . FILTER (?val < ?lo || ?val > ?hi) }}
  FILTER ( NOT EXISTS {{ ?m dt:validForGrade ?x }} || EXISTS {{ ?m dt:validForGrade ?g }} )"""

Q_WF = """PREFIX dt: <http://example.org/dt#>
SELECT ?wf ?err (GROUP_CONCAT(CONCAT(STR(?pos), ":", STR(?m)); separator="|") AS ?steps) WHERE {
  ?wf a dt:CompositeWorkflow ; dt:calibratedErrorHV ?err ; dt:hasStep ?s . ?s dt:position ?pos ; dt:model ?m .
} GROUP BY ?wf ?err"""


class KGOrchestrator:
    """Model discovery + composition driven only by graph content (contracts, statuses, context)."""

    def __init__(self, g):
        self.g = g
        self.wf = None

    def refresh(self):
        """(Re)load registered workflow contracts; call after the registry changes."""
        self.wf = {}
        for row in self.g.query(Q_WF):
            steps = tuple(n(p.split(":", 1)[1]) for p in sorted(str(row.steps).split("|"), key=lambda s: int(s.split(":")[0])))
            self.wf[steps] = float(row.err)
        return self

    def select(self, rid, budget=1.0, use_regime=True, furnace_run=None):
        if self.wf is None:
            self.refresh()
        run = str(RUN[rid])
        t0 = time.perf_counter()
        avail = {}
        for row in self.g.query(Q_AVAIL.format(run=run)):
            avail.setdefault(n(row.q), []).append(n(row.src))
        models = {}
        lat = {}
        q = Q_MODELS.format(run=run, budget=budget, regime=REGIME.format(run=run) if use_regime else "")
        for row in self.g.query(q):
            models[n(row.m)] = {"inputs": [n(i) for i in str(row.ins).split(",")], "output": n(row.out)}
            lat[n(row.m)] = float(row.lat)
        cands = []
        for steps, binding in compose("HV", models, avail):
            if steps in self.wf:
                cands.append((self.wf[steps], sum(lat[s] for s in steps), steps, binding))
        t_query = time.perf_counter() - t0
        if not cands:
            return dict(choice=None, t=t_query, ncand=0)
        cands.sort(key=lambda x: (round(x[0], 6), x[1], x[2], str(sorted(x[3].items()))))
        e, l, steps, binding = cands[0]
        return dict(choice=(steps, binding), t=t_query, ncand=len(cands), err=e, lat=l)


# python mirror (fast Monte Carlo)
def select_py(contracts, wf, status, ctx, grade, budget=1.0, use_regime=True, sources=None, props=None):
    """Same logic as KGOrchestrator.select, but over an in-memory contract table. Used for the large
    Monte-Carlo in E2; equality with the SPARQL version is checked in the experiments."""
    sources = sources or {s: v[1] for s, v in SOURCES.items()}
    props = props if props is not None else {gr: set(PROPS) for gr in GRADE_LIST}
    avail = {}
    for s, prov in sources.items():
        if status.get(s, "ok") == "ok":
            for q in prov:
                avail.setdefault(q, []).append(s)
    models, lat = {}, {}
    for m, c in contracts.items():
        ok = True
        if use_regime:
            for q, (lo, hi) in c["validity"].items():
                if q in ctx and not (lo <= ctx[q] <= hi):
                    ok = False
            if c["grades"] and grade not in c["grades"]:
                ok = False
        if any(p not in props[grade] for p in c["requires"]):
            ok = False
        if c["latency"] > budget:
            ok = False
        if ok:
            models[m] = {"inputs": c["inputs"], "output": c["output"]}
            lat[m] = c["latency"]
    cands = []
    for steps, binding in compose("HV", models, avail):
        if steps in wf:
            cands.append((wf[steps], sum(lat[s] for s in steps), steps, binding))
    if not cands:
        return None
    cands.sort(key=lambda x: (round(x[0], 6), x[1], x[2], str(sorted(x[3].items()))))
    return cands[0][2], cands[0][3]
