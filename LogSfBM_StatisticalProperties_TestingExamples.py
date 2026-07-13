"""
run_scenarios_print_outputs.py
================================

Runs every scenario from the original usage examples against the real,
model-based LogSfBM_StatisticalAnalysis_Commented API and PRINTS THE
ACTUAL RETURNED OUTPUT of each function call (dict contents, array
values/shapes, or the exception raised) — no pass/fail assertions.

This lets you directly inspect what each function returns for the
exact (H, lambda2, T, tau, size, subsample, ...) combinations used in
the original examples, translated onto the model-based API as
established previously:

    Increments(1, size=2**14, ..., H=0.001, ...)
        -> model = LogS_fBM(H=0.001, ...); Increments(model, tau=1, ...)

Sizes are reduced from the originals (some up to 2**20/2**22) so this
script finishes in a reasonable time; every other parameter (H, lambda2,
T, sigma, tau, q_list, tau_list) is preserved exactly.
"""

import sys
import types
import os

import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt

import numpy as np

from LogSfBM_Class import LogS_fBM
import LogSfBM_StatisticalProperties as LogSfBM_stats


FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def save_current_figure(name):
    """Save whatever figure is currently open (i.e. the one the target
    function just built right before its internal plt.show() call) to
    FIGURES_DIR/<name>.png, then close it to free memory.

    Since matplotlib.use("Agg") is set, plt.show() is a no-op that does
    not block and does not clear the figure, so the figure object is
    still fully populated and grabbable via plt.gcf() immediately after
    the function returns.
    """
    fig = plt.gcf()
    if fig.get_axes():  # only save if something was actually plotted
        path = os.path.join(FIGURES_DIR, f"{name}.png")
        fig.savefig(path, dpi=110, bbox_inches="tight")
        print(f"  [figure saved -> figures/{name}.png]")
    else:
        print("  [no figure produced]")
    plt.close(fig)


def _model_with_increments(H, lambda2, T):
    model = LogS_fBM(H=H, lambda2=lambda2, T=T)
    model.Increments = types.MethodType(
        lambda self, tau, size, subsample, logincr_flag, type_inc: LogSfBM_stats.Increments(
            self, tau, size, subsample, logincr_flag, type_inc
        ),
        model,
    )
    return model


def header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show_array(name, arr, max_items=10):
    arr = np.asarray(arr)
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
    if arr.size <= max_items:
        print(f"  values: {arr}")
    else:
        print(f"  first {max_items}: {arr.flat[:max_items]}")
        print(f"  last 3:  {arr.flat[-3:]}")


def show_dict_result(result, max_items=10):
    for key, val in result.items():
        if isinstance(val, np.ndarray):
            show_array(f"  '{key}'", val, max_items=max_items)
        else:
            print(f"  '{key}': {val}")


def show_curve_list(curves, label="curve", max_items=10):
    for i, c in enumerate(curves):
        c = np.asarray(c)
        print(f"  {label}[{i}]: shape={c.shape}")
        if c.size <= max_items:
            print(f"    values: {c}")
        else:
            print(f"    first {max_items}: {c.flat[:max_items]}")
            print(f"    last 3:  {c.flat[-3:]}")


np.random.seed(0)

# ===========================================================================
header("SCENARIO 1 — Increments(tau=1, H=0.001, lambda2=0.02, T=200)")
# Original: Increments(1, size=2**14, subsample=4, T=200, lambda2=0.02,
#                       H=0.001, sigma=1, flagm=False, logincr_flag=False)
model = LogS_fBM(H=0.001, lambda2=0.02, T=200)
out = LogSfBM_stats.Increments(model, tau=1, size=2 ** 10, subsample=4,
                       logincr_flag=False, type_inc="increments MRW")
show_array("Increments output", out)

# ===========================================================================
header("SCENARIO 2 — LogIncrementsDistributionDensityAcrossScale([1,20,50], H=0.1, T=2**12)")
# Original: LogIncrementsDistributionDensityAcrossScale([1,20,50], size=2**14,
#               subsample=4, T=2**12, lambda2=0.02, H=0.1, ...)
model = LogS_fBM(H=0.1, lambda2=0.02, T=2 ** 12)
out = LogSfBM_stats.LogIncrementsDistributionDensityAcrossScale(
    model, taulist=[1, 20, 50], size=2 ** 9, subsample=4, logincr_flag=False
)
print(f"Return value: {out!r}  (function only produces a plot; no data returned)")
save_current_figure("scenario02_log_increments_distribution_density")
# ===========================================================================
header("SCENARIO 12a — MomentIncrementsRepresentation(q=[1,2,3,4], tau=1..50, H=0.01, type_inc='increments MRW')")
# Original (commented out): MomentIncrementsRepresentation([1,2,3,4],
#   range(1,50), size=2**14, subsample=4, T=2**12, lambda2=0.02, H=0.01, ...)
model = _model_with_increments(H=0.01, lambda2=0.02, T=2 ** 12)
out = LogSfBM_stats.MomentIncrementsRepresentation(
    model, q_list=[1, 2, 3, 4], tau_list=list(range(1, 50)),
    size=2 ** 9, subsample=4, type_inc="increments MRW",
)
show_curve_list(out, label="M_q(tau) curve")
save_current_figure("scenario12a_moment_increments_representation_MRW")

# ===========================================================================
header("SCENARIO 12b — MomentIncrementsRepresentation(q=[1,2,3,4], tau=1..50, H=0.01, type_inc='increments MRM')")
model = _model_with_increments(H=0.01, lambda2=0.02, T=2 ** 12)
out = LogSfBM_stats.MomentIncrementsRepresentation(
    model, q_list=[1, 2, 3, 4], tau_list=list(range(1, 50)),
    size=2 ** 9, subsample=4, type_inc="increments MRM",
)
show_curve_list(out, label="M_q(tau) curve")
save_current_figure("scenario12b_moment_increments_representation_MRM")

# ===========================================================================
header("SCENARIO 12c — MomentIncrementsRepresentation(q=[0.5,1,2,3], tau=1..20, T=5.778, lambda2=0.03, H=0.14, type_inc='log MRM')")
model = _model_with_increments(H=0.14, lambda2=0.03, T=5.778)
out = LogSfBM_stats.MomentIncrementsRepresentation(
    model, q_list=[0.5, 1, 2, 3], tau_list=list(range(1, 20)),
    size=2 ** 8, subsample=4, type_inc="log MRM",
)
show_curve_list(out, label="M_q(tau) curve")
save_current_figure("scenario12c_moment_increments_representation_logMRM_fractionalT")

# ===========================================================================
for type_inc in ["increments MRW", "increments MRM", "log MRM", "S-fbm"]:
    header(f"SCENARIO 13 — MomentIncrementsRepresentationSimulatedVSTheoretical(q=[0.5,1,1.5], tau=1..30, type_inc='{type_inc}')")
    # Original (commented out): MomentIncrementsRepresentation_simulatedvstheoretical(
    #   [0.5,1,1.5], range(1,150), size=2**14, subsample=4, T=2**12,
    #   lambda2=0.02, H=0.01, ..., type_inc=<each of the 4>)
    model = _model_with_increments(H=0.01, lambda2=0.02, T=2 ** 12)
    out = LogSfBM_stats.MomentIncrementsRepresentationSimulatedVSTheoretical(
        model, q_list=[0.5, 1, 1.5], tau_list=list(range(1, 30)),
        size=2 ** 8, subsample=4, type_inc=type_inc,
    )
    show_curve_list(out, label="")
    safe_type_inc = type_inc.replace(" ", "_").replace("-", "_")
    save_current_figure(f"scenario13_moment_simulated_vs_theoretical_{safe_type_inc}")

print("\n" + "=" * 78)
print("ALL SCENARIOS EXECUTED")
print("=" * 78)
