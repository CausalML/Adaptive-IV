"""
run_semi_synthetic.py – run a synthetic experiment with the AMRIV estimator.
"""
### General imports
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rcParams
matplotlib.use('Agg')
import re
from typing import Dict, List, Tuple

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed, cpu_count

### AMRIV imports
from utils import *
from data_TA import make_synthetic_iv_dgp, true_ate, OracleMuA, OracleMuY, OracleSigma
from models import AMRIVExperiment, A2IPWExperiment

###################
# Utils Functions #
###################
def parse_label(label: str):
        """Return (root, spec) where root = Oracle/AMRIV/DM/A2IPW, spec = '' | -NA | -MS."""
        m = re.match(r"^(Oracle|AMRIV|DM|A2IPW)(-NA|-MS)?$", label)
        if not m:
            raise ValueError(f"Label {label!r} not recognised.")
        root, spec = m.group(1), (m.group(2) or "")
        return root, spec

# calculate mse helper
def mse_curve(mat: np.ndarray, t_grid: List[int], true_tau: float) -> List[float]:
    """Return tau_MSE*T/true_Sigma for each horizon T."""
    means = mat.cumsum(axis=1) / np.arange(1, mat.shape[1] + 1)  # running means
    res = []
    sd  = []
    for T in t_grid:
        tau_hat = means[:, T-1]          # estimate from first T samples
        mse_T = np.mean((tau_hat - true_tau)**2)
        mse_sd = np.std((tau_hat - true_tau)**2, ddof=1)
        res.append(mse_T)
        sd.append(mse_sd)
    return np.array(res), np.array(sd)

    
def one_run(seed: int,
            n_rounds: int = 2000,
            burn_in: int = 200,
            adaptive: bool = False,
            deltaA_eps = 1e-2,
            batch_size: int = 200, 
            trunc_schedule = lambda t: 100,
            factories = None):
    """Single AMRIV replicate"""
    rng = np.random.default_rng(seed)
    data_gen = make_synthetic_iv_dgp(seed=seed)
    exp = AMRIVExperiment(
            generator      = data_gen,
            n_rounds       = n_rounds,
            burn_in        = burn_in,
            adaptive       = adaptive,
            batch_size     = batch_size,
            deltaA_eps     = deltaA_eps,
            trunc_schedule = trunc_schedule,
            factories      = factories,
         )
    exp.collect()
    return exp.phi[exp.burn_in+1:], exp.delta_dm[exp.burn_in+1:]

def one_run_a2ipw(seed: int,
            n_rounds: int = 2000,
            burn_in: int = 200,
            adaptive: bool = False,
            batch_size: int = 200, 
            trunc_schedule = lambda t: 100,
            factories = None):
    """Single A2IPW replicate"""
    rng = np.random.default_rng(seed)
    data_gen = make_synthetic_iv_dgp(seed=seed)
    exp = A2IPWExperiment(
            generator      = data_gen,
            n_rounds       = n_rounds,
            burn_in        = burn_in,
            adaptive       = adaptive,
            batch_size     = batch_size,
            trunc_schedule = trunc_schedule,
            factories      = factories,
         )
    exp.collect()
    return exp.phi[exp.burn_in+1:]

#######################
# Simulation Settings #
#######################
figs_dir = "./results/semi_synthetic/figures"
results_dir = "./results/semi_synthetic/logs"
d = 5
np.random.seed(1)
data_gen = make_synthetic_iv_dgp(seed=0)

factories = {
    "muY0": make_rf_factory(regression=True, n_estimators=100, max_depth=5, min_samples_leaf=5),
    "muY1": make_rf_factory(regression=True, n_estimators=100, max_depth=5, min_samples_leaf=5),
    "muA0": lambda **_: OracleMuA(z=0),
    "muA1": make_rf_factory(regression=False, n_estimators=100, max_depth=3, min_samples_leaf=30),
    "s0": make_rf_factory(regression=True, n_estimators=100, max_depth=5, min_samples_leaf=5),
    "s1": make_rf_factory(regression=True, n_estimators=100, max_depth=5, min_samples_leaf=5)
}
trunc_schedule = lambda t : 2/(0.999)**t

a2ipw_factories = {
    "muY0": make_rf_factory(regression=True, n_estimators=100, max_depth=5, min_samples_leaf=100),
    "muY1": make_rf_factory(regression=True, n_estimators=100, max_depth=5, min_samples_leaf=100),
    "s0": make_rf_factory(regression=True, n_estimators=100, max_depth=5, min_samples_leaf=5),
    "s1": make_rf_factory(regression=True, n_estimators=100, max_depth=5, min_samples_leaf=5)
}

oracle_factories = {
    "muA0": lambda **_: OracleMuA(z=0),
    "muA1": lambda **_: OracleMuA(z=1),
    "muY0": lambda **_: OracleMuY(z=0),
    "muY1": lambda **_: OracleMuY(z=1),
    "s0"  : lambda **_: OracleSigma(z=0),
    "s1"  : lambda **_: OracleSigma(z=1)
}

class MSY:
    """μ^Y(z,x)  using the user’s f."""
    def __init__(self, c: float):
        self.c = c
        self.Y_model = OracleMuY(z=1)
    def fit(self, X, y=None): return self
    def predict(self, X):
        #return np.ones_like(X[:, 0])* self.c
        return self.Y_model.predict(X) + self.c
    
data_gen = make_synthetic_iv_dgp(seed=0)
X_test = np.array([data_gen()[0] for i in range(50000)])
ms_const1 = OracleMuA(z=1).predict(X_test).mean()
miss_factories = {
    "muY0": make_rf_factory(regression=True, n_estimators=100, max_depth=5, min_samples_leaf=5),
    "muY1": lambda **_: MSY(c=1.5),
    "muA0": lambda **_: OracleMuA(z=0),
    "muA1": make_rf_factory(regression=False, n_estimators=100, max_depth=3, min_samples_leaf=30),
    "s0": make_rf_factory(regression=True, n_estimators=100, max_depth=5, min_samples_leaf=5),
    "s1": make_rf_factory(regression=True, n_estimators=100, max_depth=5, min_samples_leaf=5)
}

N_REPS = 1000
N_CPU  = cpu_count()
n_rounds_raw = 2000
burn_in = 200
batch_size = 200
n_rounds = n_rounds_raw + burn_in+1

def main():
    parser = argparse.ArgumentParser(description="Run AMRIV experiments on synthetic data.")
    parser.add_argument("--plot_only", action="store_true", help="Whether to load results from previous experiments and only produce plots.", default=False)
    args = parser.parse_args()
    plot_only = args.plot_only
    if not plot_only:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        X_sample = np.array([data_gen()[0] for _ in range(50000)])
        true_tau = true_ate(X_sample)        
        ##################
        # Run experiment #
        ##################
        #### Oracle-NA ####
        print(f"Running {N_REPS} experiments for Oracle-NA… ", end='')
        taus_oracle_random = Parallel(n_jobs=N_CPU, verbose=0)(
                   delayed(one_run)(
                       seed,
                       n_rounds = n_rounds,
                       burn_in = burn_in,
                       adaptive = False,
                       deltaA_eps = 1e-2,
                       batch_size = 200, 
                       trunc_schedule = lambda t: 100,
                       factories = oracle_factories)
                   for seed in range(N_REPS)
               )
        taus_oracle_random_amriv = np.asarray([tau[0] for tau in taus_oracle_random])
        taus_oracle_random_amriv_dm = np.asarray([tau[1] for tau in taus_oracle_random])
        np.save(os.path.join(results_dir, "taus_oracle_random_amriv.npy"), taus_oracle_random_amriv)
        np.save(os.path.join(results_dir, "taus_oracle_random_amriv_dm.npy"), taus_oracle_random_amriv_dm)
        print("DONE")
        
        #### Oracle ####
        print(f"Running {N_REPS} experiments for Oracle… ", end='')
        taus_oracle = Parallel(n_jobs=N_CPU, verbose=10)(
                   delayed(one_run)(
                       seed,
                       n_rounds = n_rounds,
                       burn_in = burn_in,
                       adaptive = True,
                       deltaA_eps = 1e-2,
                       batch_size = 200, 
                       trunc_schedule = lambda t: 100,
                       factories = oracle_factories)
                   for seed in range(N_REPS)
               )
        
        taus_oracle_amriv = np.asarray([tau[0] for tau in taus_oracle])
        taus_oracle_amriv_dm = np.asarray([tau[1] for tau in taus_oracle])
        np.save(os.path.join(results_dir, "taus_oracle_amriv.npy"), taus_oracle_amriv)
        np.save(os.path.join(results_dir, "taus_oracle_amriv_dm.npy"), taus_oracle_amriv_dm)
        print("DONE")
        
        #### AMRIV-NA ####
        print(f"Running {N_REPS} experiments for AMRIV-NA…", end='')
        taus_random = Parallel(n_jobs=N_CPU, verbose=10)(
                   delayed(one_run)(
                       seed,
                       n_rounds = n_rounds,
                       burn_in = burn_in,
                       adaptive = False,
                       deltaA_eps = 1e-2,
                       batch_size = 200, 
                       trunc_schedule = trunc_schedule,
                       factories = factories)
                   for seed in range(N_REPS)
               )
        taus_random_amriv = np.asarray([tau[0] for tau in taus_random])
        taus_random_amriv_dm = np.asarray([tau[1] for tau in taus_random])
        np.save(os.path.join(results_dir, "taus_random_amriv.npy"), taus_random_amriv)
        np.save(os.path.join(results_dir, "taus_random_amriv_dm.npy"), taus_random_amriv_dm)
        print("DONE")

        #### AMRIV ####
        print(f"Running {N_REPS} experiments for AMRIV…", end="")
        taus = Parallel(n_jobs=N_CPU, verbose=10)(
                   delayed(one_run)(
                       seed,
                       n_rounds = n_rounds,
                       burn_in = burn_in,
                       adaptive = True,
                       deltaA_eps = 1e-2,
                       batch_size = 200, 
                       trunc_schedule = trunc_schedule,
                       factories = factories)
                   for seed in range(N_REPS)
               )
        taus_amriv = np.asarray([tau[0] for tau in taus])
        taus_amriv_dm = np.asarray([tau[1] for tau in taus])
        np.save(os.path.join(results_dir, "taus_amriv.npy"), taus_amriv)
        np.save(os.path.join(results_dir, "taus_amriv_dm.npy"), taus_amriv_dm)
        print("DONE")
        #### AMRIV-MS ####
        print(f"Running {N_REPS} experiments for AMRIV-MS…", end="")
        taus_ms = Parallel(n_jobs=N_CPU, verbose=10)(
           delayed(one_run)(
               seed,
               n_rounds = n_rounds,
               burn_in = burn_in,
               adaptive = True,
               deltaA_eps = 1e-2,
               batch_size = 200, 
               trunc_schedule = trunc_schedule,
               factories = miss_factories)
           for seed in range(N_REPS))
        taus_amriv_ms = np.asarray([tau[0] for tau in taus_ms])
        taus_amriv_dm_ms = np.asarray([tau[1] for tau in taus_ms])
        np.save(os.path.join(results_dir, "taus_amriv_ms.npy"), taus_amriv_ms)
        np.save(os.path.join(results_dir, "taus_amriv_dm_ms.npy"), taus_amriv_dm_ms)
        print("DONE")
        # A2IPW
        print(f"Running {N_REPS} experiments for A2IPW…", end="")

        taus_a2ipw = Parallel(n_jobs=N_CPU, verbose=10)(
                   delayed(one_run_a2ipw)(
                       seed,
                       n_rounds = n_rounds,
                       burn_in = burn_in,
                       adaptive = True,
                       batch_size = 200, 
                       trunc_schedule = trunc_schedule,
                       factories = a2ipw_factories)
                   for seed in range(N_REPS)
               )
        taus_a2ipw = np.array(taus_a2ipw)
        np.save(os.path.join(results_dir, "taus_a2ipw.npy"), taus_a2ipw)
        print("DONE")
    else:
        taus_amriv = np.load(os.path.join(results_dir, "taus_amriv.npy"))
        taus_amriv_dm = np.load(os.path.join(results_dir, "taus_amriv_dm.npy"))
        taus_random_amriv = np.load(os.path.join(results_dir, "taus_random_amriv.npy"))
        taus_random_amriv_dm = np.load(os.path.join(results_dir, "taus_random_amriv_dm.npy"))
        taus_oracle_random_amriv = np.load(os.path.join(results_dir, "taus_oracle_random_amriv.npy"))
        taus_oracle_random_amriv_dm = np.load(os.path.join(results_dir, "taus_oracle_random_amriv_dm.npy"))
        taus_oracle_amriv = np.load(os.path.join(results_dir, "taus_oracle_amriv.npy"))
        taus_oracle_amriv_dm = np.load(os.path.join(results_dir, "taus_oracle_amriv_dm.npy"))
        taus_amriv_ms = np.load(os.path.join(results_dir, "taus_amriv_ms.npy"))
        taus_amriv_dm_ms = np.load(os.path.join(results_dir, "taus_amriv_dm_ms.npy"))
        taus_a2ipw = np.load(os.path.join(results_dir, "taus_a2ipw.npy"))
        
    ########
    # Plot #
    ########
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 16
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    # style maps
    colour_of = dict(Oracle="gray", AMRIV="C0", DM="C1", A2IPW="C3")
    style_of  = {         # (marker, linestyle)
        "":      ("o", "-"),      # adaptive (default)
        "-NA":   ("^", "--"),     # non–adaptive
        "-MS":   ("s", ":"),      # misspecified
    } 
    matrices: Dict[str, np.ndarray] = {
        "Oracle-NA": taus_oracle_random_amriv,   # oracle, non‑adaptive
        "Oracle": taus_oracle_amriv,             # oracle, adaptive
        "AMRIV-NA": taus_random_amriv,           # AMRIV, non-adaptive
        "AMRIV": taus_amriv,                    # AMRIV
        "AMRIV-MS": taus_amriv_ms,              # AMRIV, misspecified
        "DM-NA": taus_random_amriv_dm,          # Direct Method, non-adaptive
        "DM": taus_amriv_dm,                    # Direct Method, adaptive
        "DM-MS": taus_amriv_dm_ms,              # Direct Method, misspecified
        "A2IPW": taus_a2ipw,                    # A2IPW 
    }
    # ---------- compute -------------------------------------------------
    X_sample = np.array([data_gen()[0] for _ in range(500000)])
    true_tau = true_ate(X_sample)
    print(f"True tau: {true_tau}")
    t_grid = np.array(range(200, 2001, 200))
    scaled_t_grid = t_grid / 1000
    curves = {name: mse_curve(mat, t_grid, true_tau)[0] for name, mat in matrices.items()}
    curves_sd = {name: mse_curve(mat, t_grid, true_tau)[1] for name, mat in matrices.items()}
    label_fontsize = 20
    marker_size = 8

    # ---------- plot: Adaptivity -------------------------------------
    names_to_plot = ["Oracle-NA", "AMRIV-NA", "AMRIV", "DM", "DM-NA"]#, "DM-MS"]
    
    plt.figure(figsize=(6, 4), dpi=100)
    for name in names_to_plot:
        root, spec             = parse_label(name)
        colour                  = colour_of[root]
        marker, linestyle       = style_of[spec]
        vals                    = curves[name]          # your pre-computed MSE curve
        oracle_vals             = curves["Oracle"]      # for normalisation
        plt.plot(
            t_grid, np.array(vals) / np.array(oracle_vals),
            marker=marker, linestyle=linestyle, markersize=marker_size, markeredgecolor='black',
            color=colour, label=name,
        )
    #plt.axhline(1.0, color=colour_of["Oracle"], linestyle=style_of[""][1], marker=style_of[""][0])
    plt.plot(t_grid, np.ones_like(t_grid), color=colour_of["Oracle"], linestyle=style_of[""][1], marker=style_of[""][0], 
             markersize=marker_size, markeredgecolor='black')
    plt.xticks(t_grid, scaled_t_grid)
    plt.xlabel(r"$T \,/\, 1000$", fontsize=label_fontsize)
    plt.ylabel(r"$\mathrm{MSE}(\hat\tau)\;/\;\mathrm{MSE}(\mathrm{Oracle})$", fontsize=label_fontsize)
    plt.title("Normalized MSE vs. horizon", fontsize=label_fontsize)
    plt.grid(alpha=.3)
    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(figs_dir, "Adaptivity_semi_synthetic.pdf"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(figs_dir, "Adaptivity_semi_synthetic.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ---------- plot: Consistency  -------------------------------------
    names_to_plot = ["AMRIV", "AMRIV-MS", "DM", "DM-MS", "A2IPW"]   # which curves to show
    
    plt.figure(figsize=(6, 4), dpi=100)
    for name in names_to_plot:
        root, spec          = parse_label(name)
        colour              = colour_of[root]
        marker, linestyle   = style_of[spec]
    
        mean_vals = np.asarray(curves[name])                # MSE curve
        sd_vals   = np.asarray(curves_sd[name])             # SD at each horizon
    
        plt.plot(t_grid, mean_vals, marker=marker, markersize=marker_size, markeredgecolor='black',
                 linestyle=linestyle, color=colour, label=name)
    
        plt.fill_between(
            t_grid,
            mean_vals - sd_vals / np.sqrt(t_grid),
            mean_vals + sd_vals / np.sqrt(t_grid),
            color=colour, alpha=.15
        )
    
    plt.xticks(t_grid, scaled_t_grid)
    plt.xlabel(r"$T \,/\, 1000$", fontsize=label_fontsize)
    plt.ylabel("MSE ± SE", fontsize=label_fontsize)
    plt.title("MSE vs. horizon", fontsize=label_fontsize)
    plt.grid(alpha=.3)
    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(figs_dir, "Consistency_semi_synthetic.pdf"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(figs_dir, "Consistency_semi_synthetic.png"), dpi=200, bbox_inches="tight")
    plt.close()


    # ---------- coverage helper  (unchanged) ----------------------------
    alpha  = 0.05
    z_crit = 1.96
    t_grid = list(range(200, 2001, 200))
    
    def coverage_within_exp(mat, t_grid):
        cover = []
        for T in t_grid:
            phi_T = mat[:, :T]                         # pseudo-outcomes ϕ₁…ϕ_T
            est   = phi_T.mean(axis=1)                 # \hat{τ}_T per replicate
            se    = phi_T.std(axis=1, ddof=1) / np.sqrt(T)
            half  = z_crit * se
            ok    = (true_tau >= est - half) & (true_tau <= est + half)
            cover.append(ok.mean())                    # fraction covered
        return cover
    
    cover_curves = {name: coverage_within_exp(mat, t_grid)
                    for name, mat in matrices.items()}
    
    # ---------- plot: Coverage  -----------------------------------------
    names_to_plot = ["AMRIV", "AMRIV-NA", "AMRIV-MS", "DM", "DM-NA", "DM-MS", "A2IPW"]
    
    plt.figure(figsize=(6, 4), dpi=100)
    handles = []
    labels = []
    for name in names_to_plot:
        root, spec = parse_label(name)
        colour = colour_of[root]
        marker, linestyle = style_of[spec]
        (line,) = plt.plot(
            t_grid, cover_curves[name],
            marker=marker, linestyle=linestyle, markersize=marker_size, markeredgecolor='black',
            color=colour, label=name
        )
        handles.append(line)
        labels.append(name)

    # --- Add Oracle-NA (from earlier plot) manually ---
    oracle_colour = colour_of["Oracle"]
    oracle_marker, oracle_linestyle = style_of["-NA"]
    oracle_handle = Line2D([], [], color=oracle_colour, marker=oracle_marker, markersize=marker_size, markeredgecolor='black',
                        linestyle=oracle_linestyle, label="Oracle-NA")
    handles.append(oracle_handle)
    labels.append("Oracle-NA")
    # --- Add Oracle (from earlier plot) manually ---
    oracle_marker, oracle_linestyle = style_of[""]
    oracle_handle = Line2D([], [], color=oracle_colour, marker=oracle_marker, markersize=marker_size, markeredgecolor='black',
                        linestyle=oracle_linestyle, label="Oracle-NA")
    handles.append(oracle_handle)
    labels.append("Oracle")
    
    plt.axhline(1 - alpha, c="black", ls=":", lw=3, label="95% target", zorder=0)
    #target_handle = Line2D([], [], color="black", linestyle=":", linewidth=2, label="95 % target")
    #handles.append(target_handle)
    #labels.append("95% target")

    plt.xticks(t_grid, scaled_t_grid)
    plt.xlabel(r"$T \,/\, 1000$", fontsize=label_fontsize)
    plt.ylabel("Empirical coverage", fontsize=label_fontsize)
    plt.title("95% CI coverage vs. horizon", fontsize=label_fontsize)
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=.3)
    plt.tight_layout(pad=0.3)
    # Use legend from all three plots
    legend = plt.legend(handles=handles, labels=labels, fontsize=16, loc="center left", bbox_to_anchor=(1.00, 0.50))
    plt.savefig(os.path.join(figs_dir, "Coverage_semi_synthetic.pdf"), dpi=200,
                bbox_inches="tight", bbox_extra_artists=(legend,))
    plt.savefig(os.path.join(figs_dir, "Coverage_semi_synthetic.png"), dpi=200,
                bbox_inches="tight", bbox_extra_artists=(legend,))
    plt.close()

if __name__ == "__main__":
    main()


