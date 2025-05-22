"""Trip-Advisor data-generating process (DGP) and analytic oracles."""
import locale as _loc
import numpy as np
import pandas as pd
import scipy.special
from typing import Callable
from sklearn.base import RegressorMixin

X_colnames = {                           # keep the same order!
    'days_visited_exp_pre':  'day_count_pre',
    'days_visited_free_pre': 'day_count_pre',
    'days_visited_fs_pre':   'day_count_pre',
    'days_visited_hs_pre':   'day_count_pre',
    'days_visited_rs_pre':   'day_count_pre',
    'days_visited_vrs_pre':  'day_count_pre',
    'is_existing_member':    'binary',
    'locale_en_US':          'binary',
    'os_type':               'os',
    'revenue_pre':           'revenue',
}

LOCALE_LIST   = sorted(_loc.locale_alias.keys())            # ~700 items
LOCALE_TO_INT = {loc: i for i, loc in enumerate(LOCALE_LIST)}
NUM_CATS      = len(LOCALE_LIST)

coef_Z   = 0.8                       # from your dgp_binary()
p0_const = 0.006                     # Pr[C0=1]  when Z=0 (non-complier rate)
U_low, U_high = -5.0, 5.0            # support of ν  (unobs. heterog.)
sigma_0 = 1.50                       # std of ε₀
sigma_1 = 0.25                       # std of ε₁

def true_fn(X: np.ndarray) -> np.ndarray | float:
    """
    Structural CATE function that works for
      • X shape (d,)  – single unit
      • X shape (n,d) – batch
    """
    X = np.asarray(X)
    single = (X.ndim == 1)
    if single:                         # make it (1,d) for uniform code
        X = X[None, :]

    x1  = X[:, 0]                      # first covariate
    x7  = X[:, 6]                      # seventh covariate (index 6)

    piecewise = (
          5 * (x1 > 5)
        + 10 * (x1 > 15)
        + 5 * (x1 > 20)
    )

    vals = 0.8 + 0.5 * piecewise - 3.0 * x7    # shape (n,)
    return vals[0] if single else vals

# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------
def encode_locale(str_array: np.ndarray) -> np.ndarray:
    """
    Map each locale string to its fixed integer code.
    Unknown locales get code = NUM_CATS (optional).
    """
    vec_map = np.vectorize(LOCALE_TO_INT.get, otypes=[float])
    codes   = vec_map(str_array, NUM_CATS)      # fallback for unseen
    return codes.astype(float)

def gen_data(data_type, n, rng):
    if data_type == 'day_count_pre':
        return rng.integers(0, 29, size=n)          # 0–28
    if data_type == 'os':
        # os_type = {0: 'osx', 1: 'windows', 2: 'linux'}
        return rng.choice([0, 1, 2], size=n)
    if data_type == 'locale':
        import locale as _loc
        return rng.choice(list(_loc.locale_alias.keys()), size=n)
    if data_type == 'binary':
        return rng.binomial(1, .5, size=n)
    if data_type == 'revenue':
        return np.round(rng.lognormal(0, 3, size=n), 2)
    raise ValueError(f"Unknown data_type {data_type!r}")

def x_factory(seed: int | None = None) -> Callable[[int], np.ndarray]:
    """
    Returns a callable draw_x(n) that samples n iid rows from N(0,I_d).
    """
    rng = np.random.default_rng(seed)
    col_names = list(X_colnames.keys())
    
    def draw_x(n: int = 1) -> np.ndarray:
        col_arrays = []
        for col in col_names:
            dtype = X_colnames[col]
            arr   = gen_data(dtype, n, rng)
            # convert categorical strings to numeric codes
            if dtype == 'locale':                     
                arr = encode_locale(arr)
            col_arrays.append(arr.astype(float))

        X = np.column_stack(col_arrays)           # shape (n, d)
        return X[0] if n == 1 else X
    
    return draw_x
        
# ----------------
# synthetic IV–DGP 
# ----------------
def ln_mean(s): return np.exp(0.5 * s**2)
def ln_var(s):  return (np.exp(s**2) - 1.0) * np.exp(s**2)

def make_synthetic_iv_dgp(
    seed: int | None = None,
) -> Callable[[], tuple[np.ndarray, int, int, float, float]]:
    """
    Returns g() that draws one observation and outputs
    (X , A(0), A(1), Y(0), Y(1)).
    """
    rng = np.random.default_rng(seed)
    draw_x   = x_factory(seed)              # reuse same RNG for X

    def g():
        # 1. covariate
        X = draw_x()
        nu = np.random.uniform(U_low, U_high)
        
        A1 = np.random.binomial(1, coef_Z*scipy.special.expit(0.4*X[0] + nu))
        A0 = np.random.binomial(1, .006)

        eps0 = np.random.lognormal(0, sigma_0)
        eps1 = np.random.lognormal(0, sigma_1)

        Y1 = true_fn(X) + 2*nu + 5*(X[0]>0) + eps1
        Y0 = 2*nu + 5*(X[0]>0) + eps0

        return X, A0, A1, Y0, Y1
    return g

# -------
# Oracles
# -------

def _p1(x1: np.ndarray) -> np.ndarray:
    """
    p₁(x) = E[C | X=x]  when Z=1  (compliance prob. under encouragement)

    C |X=x, Z=1 ~ Bern{ coef_Z * σ(0.4*x₁ + ν) },
    """
    a = 0.4 * x1
    term = (np.log1p(np.exp(a + U_high)) - np.log1p(np.exp(a + U_low))) / (U_high - U_low)
    return coef_Z * term

def _p0(_: np.ndarray) -> np.ndarray:
    """
    p₀(x) = Pr[C0=1] is constant (≈ 0.006) in your generator.
    """
    return np.full_like(_, p0_const, dtype=float)

# ---------- oracle μ^A -----------------------------------------------
class OracleMuA(RegressorMixin):
    """E[A | Z=z, X=x]  (treatment take-up rate)"""
    def __init__(self, z: int):
        self.z = int(z)
    def fit(self, X, y=None):  # no-op
        return self
    def predict(self, X):
        x1 = np.asarray(X)[:, 0]
        return _p1(x1) if self.z == 1 else _p0(x1)

# ---------- oracle μ^Y -----------------------------------------------
class OracleMuY(RegressorMixin):
    """
    μ^Y(z,x) = E[Y | Z=z, X=x] under the data-generating process
    Needs user-supplied structural f(x,a).
    """
    def __init__(self, z: int):
        self.z = int(z)
    def fit(self, X, y=None):
        return self
    def predict(self, X):
        X   = np.asarray(X)
        x1  = X[:, 0]
        p1  = _p1(x1)
        p0  = _p0(x1)
        mean_nu  = (U_high + U_low) / 2.0
        if self.z == 1:
            return true_fn(X)*p1 + 2*mean_nu + 5*(X[:, 0]>0) + ln_mean(sigma_1) * p1 + ln_mean(sigma_0) * (1 - p1)
        else:
            return true_fn(X)*p0 + 2*mean_nu + 5*(X[:, 0]>0) + ln_mean(sigma_1) * p0 + ln_mean(sigma_0) * (1 - p0)

# ---------- oracle σ²(z,x) -------------------------------------------
class OracleSigma(RegressorMixin):
    """
    σ²(z,x) = Var[ Y | Z=z, X=x ]   (needed for π*(x))
    """
    def __init__(self, z: int):
        self.z = int(z)
        # pre-compute constants
        self.var_nu  = (U_high - U_low)**2 / 12.0                         
    def fit(self, X, y=None):
        return self
    def predict(self, X):
        X   = np.asarray(X)
        x1  = X[:, 0]
        p1  = _p1(x1)
        p0  = _p0(x1)
        p   = p1 if self.z == 1 else p0
        var0 = ln_var(sigma_0)
        var1 = ln_var(sigma_1)

        return 4 * self.var_nu + var1 * p + var0 *(1-p)

# ---------- τ(x), τ, Σ -----------------------------------------------
def true_cate(x):
    """τ(x) = f(x,1) – f(x,0)  (vectorised)."""
    x = np.asarray(x)
    single = x.ndim == 1
    if single:
        x = x[None, :]
    res = true_fn(x) + ln_mean(sigma_1) - ln_mean(sigma_0)
    return res[0] if single else res

def true_ate(X_sample):
    """E_X[τ(X)] – just empirical average over a large sample."""
    return np.mean(true_cate(X_sample))

def true_sigma(X_sample, use_adaptive = True):
    """
    Efficient-bound Σ_true = E[ (√σ0+√σ1)² / γ² ]   (à la AMRIV paper)
    """
    sigma0 = OracleSigma(0).predict(X_sample)
    sigma1 = OracleSigma(1).predict(X_sample)
    delta_A    = _p1(X_sample[:, 0]) - _p0(X_sample[:, 0])
    true_ate_calc = true_ate(X_sample)  # E[τ(X)]
    if use_adaptive:
        scores = 1/(delta_A**2)*(
        (np.sqrt(sigma0)+np.sqrt(sigma1))**2
        ) + (true_cate(X_sample) - true_ate_calc)**2 
    else:
        scores = 1/(delta_A**2)*(
        (sigma0+sigma1)*2
        ) + (true_cate(X_sample) - true_ate_calc)**2 
    return np.mean(scores)

def true_pi(x):
    x_arr = np.asarray(x)
    single = x_arr.ndim == 1
    if single:
        x_arr = x_arr[None, :]          # make it 2-D for uniform call

    s0 = OracleSigma(z=0)
    s1 = OracleSigma(z=1)
    pi = np.sqrt(s1.predict(x_arr)) / (np.sqrt(s0.predict(x_arr)) + np.sqrt(s1.predict(x_arr))) 
    return pi[0] if single else pi
