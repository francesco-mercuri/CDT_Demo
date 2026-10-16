# Demonstrator of a contract-based semantic model selection

Requires: Python 3 with numpy, scipy, scikit-learn, rdflib, pandas, matplotlib.

Usage:

`python3 build_lib.py` trains the model library, calibrates workflows and writes lib.pkl.

`python3 exp_run.py e1` and `python3 exp_run.py e4` run the fault, out-of-regime, new-grade and spare-sensor experiments.

`python3 exp_e2.py` (LLM-lifting gates, parametric error injection) and `python3 exp_e3.py` (equivalence, latency).

`python3 figs.py` makes the figures. 

All data are synthetic, results are in results/.
