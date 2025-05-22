import numpy as np
from typing import Callable
from sklearn.base import RegressorMixin

def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))

def _gamma(x1):
    """Compliance probability γ(x) = σ(x₁)."""
    gamma_bar = 0.0
    return gamma_bar + sigmoid(2*x1) * (1 - gamma_bar)

def _p0(gamma):   
    return 0.5 * (1 - gamma)

def _p1(gamma):  
    return 0.5 * (1 + gamma)

# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------
def x_factory(d: int = 1, seed: int | None = None) -> Callable[[int], np.ndarray]:
    """
    Returns a callable draw_x(n) that samples n iid rows from N(0,I_d).
    """
    rng = np.random.default_rng(seed)
    def draw_x(n: int = 1) -> np.ndarray:
        #return rng.normal(size=(n, d)) if n > 1 else rng.normal(size=d)
        return rng.uniform(0, 2, size=(n, d)) if n > 1 else rng.uniform(0, 2, size=d)
    return draw_x

# ---------------------------------------------------
# synthetic IV–DGP  (mirrors every line of the paper)
# ---------------------------------------------------
def make_synthetic_iv_dgp(
    f: Callable[[np.ndarray, int], float],
    d: int = 1,
    v0: float = 4.0,
    v1: float = 0.25,
    seed: int | None = None,
) -> Callable[[], tuple[np.ndarray, int, int, float, float]]:
    """
    Returns g() that draws one observation and outputs
    (X , A(0), A(1), Y(0), Y(1)).
    """
    rng      = np.random.default_rng(seed)
    draw_x   = x_factory(d, seed)              # reuse same RNG for X

    def g():
        # 1. covariate
        X = draw_x()                           # X ~ N(0,I_d)
        x1 = X[0]

        # 2. compliance prob  γ(x) = σ(x1)
        gamma = _gamma(x1)
        C     = rng.binomial(1, gamma)         # compliance indicator

        # 3. potential treatments
        A0 = 0                # Z = 0  => A = 0 always
        A1 = int(C)           # Z = 1  => A = C

        # 4. helper to draw Y(a)
        def y_potential(a: int) -> float:
            # unobserved confounder U
            mean_u = 0.0 if C == 1 else -2.0     # note: a is 0 when C=0
            #U      = rng.normal(loc=mean_u, scale=0.5)
            U = mean_u # constant confounder. randomness can be absobed in ε_A

            # measurement noise ε_A
            #var_eps = v1 * a + v0 * (1 - a)      # variance as specified
            var_eps = v1 * a + (v0 * x1 + v1) * (1 - a)
            unif_bound = np.sqrt(var_eps*12) / 2
            eps = rng.uniform(-unif_bound, unif_bound) # uniform noise
            #eps     = rng.normal(0.0, np.sqrt(var_eps))
            return f(X, a) + U + eps

        Y0, Y1 = y_potential(0), y_potential(1)
        return X, A0, A1, Y0, Y1

    return g

class OracleMuA:
    """μ^A(z,x)  (no fitting needed)."""
    def __init__(self, z: int):
        self.z = int(z)
    def fit(self, X, y=None): return self
    def predict(self, X):
        gamma = _gamma(X[:, 0])
        return gamma if self.z == 1 else np.zeros_like(gamma)

class OracleMuY:
    """μ^Y(z,x)  using the user’s f."""
    def __init__(self, z: int, f):
        self.z, self.f = int(z), f
    def fit(self, X, y=None): return self
    def predict(self, X):
        gamma = _gamma(X[:, 0])
        f0    = self.f(X, 0)
        f1    = self.f(X, 1)
        core  = f0                     # constant term
        return core + gamma * (f1 - f0) if self.z == 1 else core

class OracleSigma:
    """σ²(z,x)  (residual variance needed for π*)."""
    def __init__(self, z: int, v0: float = 4.0, v1: float = 0.25):
        self.z, self.v0, self.v1 = int(z), v0, v1
    def fit(self, X, y=None): return self
    def predict(self, X):
        gamma = _gamma(X[:, 0])
        #base  = 0.25 + 0.25 * gamma * (1 - gamma) + self.v0
        v0 = self.v0 * X[:, 0] + self.v1
        #base = np.ones_like(gamma) * self.v0
        #base = np.ones_like(gamma) * v0
        if self.z == 1:
            return v0 + (self.v1 - v0) * gamma
        return v0

class OracleS(RegressorMixin):
    """
    g(z,x) = E[(Y - A δ(x))² | Z=z, X=x]
           = σ²(z,x) + f(0,x)²
    Requires: structural function f, noise scale α, instrument arm z.
    """
    def __init__(self, z: int, f, v0: float = 4.0, v1: float = 0.25):
        self.z      = int(z)
        self.f      = f
        self.v0     = v0
        self.v1     = v1

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        X = np.asarray(X)
        single = X.ndim == 1
        if single:
            X = X[None, :]

        gamma = _gamma(X[:, 0])
        #base = 0.25 + 0.25*gamma*(1 - gamma) + self.v0
        base = self.v0
        if self.z == 1:
            sigma_z = base + (self.v1 - self.v0) * gamma
        else:
            sigma_z = base
        
        f0 = self.f(X, 0)   
        g_val = sigma_z + f0**2
        
        return g_val[0] if single else g_val
    

def true_cate(x, f) -> np.ndarray:
    """
    Compute the true conditional ATE τ(x) for the DGP
    """
    x_arr = np.asarray(x)
    single = x_arr.ndim == 1
    if single:
        x_arr = x_arr[None, :]          # make it 2-D for uniform call

    tau = f(x_arr, 1) - f(x_arr, 0)  # vectorised call on batch
    return tau[0] if single else tau

def true_ate(f, d, n=50000) -> float:
    """
    Compute the true ATE τ = E[τ(X)] for the DGP
    """
    # E[τ(X)] = E[f(X,1) - f(X,0)]
    #         = E[f(X,1)] - E[f(X,0)]
    X = x_factory(d=d, seed=0)(n)  # X ~ x_factory
    return np.mean(f(X, 1) - f(X, 0))  # vectorised call on batch

def true_ate_var(f, d, v0: float = 4.0, v1: float = 0.25, n=50000, use_adaptive=True) -> float:
    """
    Compute the true ATE variance σ²(τ) = E[τ(X)²] - τ²
    """
    X = x_factory(d=d, seed=0)(n)  # X ~ x_factory
    gamma = _gamma(X[:, 0])

    #sigma1 = 0.25 + 0.25*gamma*(1 - gamma) + (v1 - v0) * gamma + v0
    #sigma0 = 0.25 + 0.25*gamma*(1 - gamma) + v0

    sigma1 = OracleSigma(z=1, v0=v0, v1=v1).predict(X)
    sigma0 = OracleSigma(z=0, v0=v0, v1=v1).predict(X)
    true_ate_calc = true_ate(f, d, n=n)  # E[τ(X)]
    if use_adaptive:
        scores = 1/(gamma**2)*(
        (np.sqrt(sigma0)+np.sqrt(sigma1))**2
        ) + (true_cate(X, f) - true_ate_calc)**2 
    else:
        scores = 1/(gamma**2)*(
        (sigma0+sigma1)*2
        ) + (true_cate(X, f) - true_ate_calc)**2 
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