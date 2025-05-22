"""
AMRIV Experiment class. Runs the AMRIV algorithm on a given data generator.
"""
from __future__ import annotations
import dataclasses, numpy as np
from sklearn.base import RegressorMixin
from utils import DataGenerator, LearnerFactory, make_rf_factory, make_knn_factory

@dataclasses.dataclass
class AMRIVExperiment:
    generator:           DataGenerator
    n_rounds:            int
    burn_in:             int = 100
    adaptive:            bool = True
    trunc_schedule:      callable[[int], int] = lambda t: 100
    CV:                  bool = False  # Whether to use cross-validation for nuisance learners
    batch_size:          int = 1
    deltaA_eps:          float = 1e-3  # minimum deltaA to avoid division by zero
    factories:           dict[str, LearnerFactory | RegressorMixin] = dataclasses.field(default_factory=dict)
    learner_kwargs:      dict[str, dict] = dataclasses.field(default_factory=dict)   # per-nuisance kwargs

    # nuisance learners (filled in __post_init__)
    muY0:       RegressorMixin = dataclasses.field(init=False)
    muA0:       RegressorMixin = dataclasses.field(init=False)
    muY1:       RegressorMixin = dataclasses.field(init=False)
    muA1:       RegressorMixin = dataclasses.field(init=False)
    muY0_fold0: RegressorMixin = dataclasses.field(init=False)
    muY0_fold1: RegressorMixin = dataclasses.field(init=False)
    muY1_fold0: RegressorMixin = dataclasses.field(init=False)
    muY1_fold1: RegressorMixin = dataclasses.field(init=False)
    muA0_fold0: RegressorMixin = dataclasses.field(init=False)
    muA0_fold1: RegressorMixin = dataclasses.field(init=False)
    muA1_fold0: RegressorMixin = dataclasses.field(init=False)
    muA1_fold1: RegressorMixin = dataclasses.field(init=False)
    s0:         RegressorMixin = dataclasses.field(init=False)
    s1:         RegressorMixin = dataclasses.field(init=False)

    # data - NumPy arrays
    X: np.ndarray = dataclasses.field(init=False)   # shape (0, d) initially
    Z: np.ndarray = dataclasses.field(init=False)   
    A: np.ndarray = dataclasses.field(init=False)
    Y: np.ndarray = dataclasses.field(init=False)
    pi_t: np.ndarray = dataclasses.field(init=False)  # keeps track of estimated pi_t
    phi: np.ndarray = dataclasses.field(init=False) 
    delta_dm: np.ndarray = dataclasses.field(init=False)  # for direct method ATE estimation

    # ────────────────────────────────────────────────────────────
    def __post_init__(self):
        # default learners if user did not supply anything
        default = lambda **kw: make_rf_factory(regression=True)(**kw)
        default_A = lambda **kw: make_rf_factory(regression=False)(**kw)

        def build(name, fallback):
            spec = self.factories.get(name, fallback)
            kw   = self.learner_kwargs.get(name, {})
            # if user passed an *instance* use it directly
            if isinstance(spec, RegressorMixin):
                return spec
            # else assume it's a factory
            return spec(**kw)

        self.muY0 = build("muY0", default)
        self.muY1 = build("muY1", default)
        self.muA0 = build("muA0", default_A)
        self.muA1 = build("muA1", default_A)
        self.s0   = build("s0",   default)
        self.s1   = build("s1",   default)

        # fold learners (if CV)
        if self.CV:
            self.muY0_fold0 = build("muY0", default)
            self.muY1_fold0 = build("muY1", default)
            self.muA0_fold0 = build("muA0", default_A)
            self.muA1_fold0 = build("muA1", default_A)
            self.muY0_fold1 = build("muY0", default)
            self.muY1_fold1 = build("muY1", default)
            self.muA0_fold1 = build("muA0", default_A)
            self.muA1_fold1 = build("muA1", default_A)

        # allocate data arrays (unchanged)
        self.X = np.empty((0,))
        self.Z = np.empty(self.n_rounds, dtype=int)
        self.A = np.empty(self.n_rounds, dtype=int)
        self.Y = np.empty(self.n_rounds, dtype=float)
        self.phi = np.empty(self.n_rounds, dtype=float)
        self.delta_dm = np.empty(self.n_rounds, dtype=float)
        self.pi_t = np.empty(self.n_rounds, dtype=float)

    # helper: append one observation
    def _append_obs(self, t: int, x: np.ndarray, z: int, a: int, y: float):
        if t == 0:                 # first sample ⇒ know d
            self.X = np.empty((self.n_rounds, x.shape[0]))
        self.X[t] = x
        self.Z[t] = z
        self.A[t] = a
        self.Y[t] = y

    # instrument assignment
    def _assign(self, x: np.ndarray, t: int) -> int:
        if t <= self.burn_in or self.adaptive == False:
            self.pi_t[t] = 0.5
            return 0.5, np.random.binomial(1, 0.5)
        v0, v1 = self._plug_in_sigmas(x)
        raw = np.sqrt(v1) / (np.sqrt(v0) + np.sqrt(v1))
        k_t = max(2, self.trunc_schedule(t))
        """
        if t % self.batch_size == 0:
            print("x", x[0])
            print("Raw pi: ", raw) 
            print("Truncated pi: ",  1 / k_t, 1 - 1 / k_t)
        """
        pi   = np.clip(raw, 1 / k_t, 1 - 1 / k_t)
        self.pi_t[t] = pi
        return pi, np.random.binomial(1, pi)
    
    # plug-in sigmas
    def _plug_in_sigmas(self, x: np.ndarray) -> tuple[float, float]:
        x_input = x.reshape(1, -1)
        deltaA = np.maximum(self.muA1.predict(x_input)[0] - self.muA0.predict(x_input)[0], self.deltaA_eps)
        #deltaA = self.muA1.predict(x_input)[0] - self.muA0.predict(x_input)[0]
        delta = (self.muY1.predict(x_input)[0] - self.muY0.predict(x_input)[0]) / deltaA
        #v_mean = (self.muY0.predict(x_input)[0] - self.muA0.predict(x_input)[0] * delta) ** 2
        v0 = self.s0.predict(x_input)[0] #- v_mean
        v1 = self.s1.predict(x_input)[0] #- v_mean
        """
        print("----------")
        print("x_input: ", x_input[0])
        print("s0: ", self.s0.predict(x_input)[0], "s1: ", self.s1.predict(x_input)[0])
        print("deltaA: ", deltaA, "delta: ", delta, "v_mean: ", v_mean)
        print("v0: ", v0, "v1: ", v1)
        print("----------")
        """

        # ensure positive variance
        v0 = v0 if v0 > 0 else 1e-3
        v1 = v1 if v1 > 0 else 1e-3
        return v0, v1
    
    def _calculate_phi(self, t: int, x: np.ndarray, z: int, a: int, y: float, pi: float):
        if t <= self.burn_in: # might want another condition here
            self.phi[t] = 0.0
            self.delta_dm[t] = 0.0
            return
        else:
            x_input = x.reshape(1, -1)
            deltaA = np.maximum(self.muA1.predict(x_input)[0] - self.muA0.predict(x_input)[0], self.deltaA_eps)
            delta = (self.muY1.predict(x_input)[0] - self.muY0.predict(x_input)[0]) / deltaA
            v_mean = self.muY0.predict(x_input)[0] - self.muA0.predict(x_input)[0] * delta

            phi_calc = ((2 * z - 1) / (z * pi + (1 - z) * (1 - pi))) \
                * (1 / deltaA) * (y - a * delta - v_mean) + delta
            self.phi[t] = phi_calc
            self.delta_dm[t] = delta

    # online collection of data
    def collect(self, restart: bool = False) -> None:
        if restart:
            self.X = np.empty((0,))
            self.Z = np.empty(self.n_rounds, dtype=int)
            self.A = np.empty(self.n_rounds, dtype=int)
            self.Y = np.empty(self.n_rounds, dtype=float)
            self.phi = np.empty(self.n_rounds, dtype=float)  # for ATE estimation
            self.delta_dm = np.empty(self.n_rounds, dtype=float)  # for direct method ATE estimation
            self.pi_t = np.empty(self.n_rounds, dtype=float)  # keeps track of estimated pi_t
        else:
            if self.X.size > 0:
                raise ValueError("Data already collected. Use restart=True to reset.")
        for t in range(self.n_rounds):
            # draw a fresh unit with all potentials
            x, a0, a1, y0, y1 = self.generator()

            # instrument assignment
            pi, z = self._assign(x, t)

            # realised treatment A(Z)
            a_realised = a1 if z == 1 else a0

            # realised outcome  Y(A)
            y_realised = y1 if a_realised == 1 else y0

            # log the observation
            self._append_obs(t, x, z, a_realised, y_realised)
            self._calculate_phi(t, x, z, a_realised, y_realised, pi)

            # periodic nuisance refit
            if t >= self.burn_in and t%self.batch_size == 0:
                #print(f"Refitting at round {t}...")
                self._refit(t)

    # ───────────────────────── nuisance refitting ─────────────────────────
    def _refit(self, t: int):
        """
        Re-train all nuisance learners using data indices 0 … t (inclusive).
        Works with the pre-allocated arrays X, Z, A, Y of length n_rounds.
        """
        # slice of currently-available data
        idx   = slice(0, t + 1)          # 0 … t
        X_t   = self.X[idx]
        Z_t   = self.Z[idx]
        A_t   = self.A[idx]
        Y_t   = self.Y[idx]

        # ---------------- base learners ----------------
        self.muY0.fit(X_t[Z_t == 0], Y_t[Z_t == 0])
        self.muY1.fit(X_t[Z_t == 1], Y_t[Z_t == 1])
        self.muA0.fit(X_t[Z_t == 0], A_t[Z_t == 0])
        self.muA1.fit(X_t[Z_t == 1], A_t[Z_t == 1])

        # ---------------- cross-validation branch ----------------
        if self.CV and t >= 1:                     
            fold0 = np.zeros(t + 1, dtype=bool)
            fold0[::2] = True                  
            fold1 = ~fold0

            # train on opposite folds
            self.muY0_fold0.fit(X_t[fold0 & (Z_t == 0)], Y_t[fold0 & (Z_t == 0)])
            self.muY1_fold0.fit(X_t[fold0 & (Z_t == 1)], Y_t[fold0 & (Z_t == 1)])
            self.muA0_fold0.fit(X_t[fold0 & (Z_t == 0)], A_t[fold0 & (Z_t == 0)])
            self.muA1_fold0.fit(X_t[fold0 & (Z_t == 1)], A_t[fold0 & (Z_t == 1)])

            self.muY0_fold1.fit(X_t[fold1 & (Z_t == 0)], Y_t[fold1 & (Z_t == 0)])
            self.muY1_fold1.fit(X_t[fold1 & (Z_t == 1)], Y_t[fold1 & (Z_t == 1)])
            self.muA0_fold1.fit(X_t[fold1 & (Z_t == 0)], A_t[fold1 & (Z_t == 0)])
            self.muA1_fold1.fit(X_t[fold1 & (Z_t == 1)], A_t[fold1 & (Z_t == 1)])

            # cross-fitted δ(x) on opposite folds
            deltaA_f1 = np.maximum(self.muA1_fold0.predict(X_t[fold1]) - self.muA0_fold0.predict(X_t[fold1]), self.deltaA_eps)
            delta_f1  = (self.muY1_fold0.predict(X_t[fold1]) - self.muY0_fold0.predict(X_t[fold1])) / deltaA_f1

            deltaA_f0 = np.maximum(self.muA1_fold1.predict(X_t[fold0]) - self.muA0_fold1.predict(X_t[fold0]), self.deltaA_eps)
            delta_f0  = (self.muY1_fold1.predict(X_t[fold0]) - self.muY0_fold1.predict(X_t[fold0])) / deltaA_f0

            delta = np.empty(t + 1)
            delta[fold0] = delta_f0
            delta[fold1] = delta_f1

            # TODO: fix this
            # residual-variance learners
            v_mean = np.empty(t + 1)
            v_mean[fold0] = self.muY0_fold1.predict(X_t[fold0]) - self.muA0_fold1.predict(X_t[fold0]) * delta_f0
            v_mean[fold1] = self.muY0_fold0.predict(X_t[fold1]) - self.muA0_fold0.predict(X_t[fold1]) * delta_f1
            self.s0.fit(X_t[Z_t == 0], (Y_t[Z_t == 0] - A_t[Z_t == 0] * delta[Z_t == 0] - v_mean[Z_t == 0]) ** 2)
            self.s1.fit(X_t[Z_t == 1], (Y_t[Z_t == 1] - A_t[Z_t == 1] * delta[Z_t == 1] - v_mean[Z_t == 1]) ** 2)

        # ---------------- simpler branch (no CV) ----------------
        else:
            deltaA = np.maximum(self.muA1.predict(X_t) - self.muA0.predict(X_t), self.deltaA_eps)
            #print("self.muA1.predict(X_t)", self.muA1.predict(X_t))
            #print("self.muA0.predict(X_t)", self.muA0.predict(X_t))
            #print("deltaA: ", deltaA)
            delta  = (self.muY1.predict(X_t) - self.muY0.predict(X_t)) / deltaA
            #print("delta: ", delta)
            #print("Maximum delta: ", np.max(delta))
            v_mean = self.muY0.predict(X_t) - self.muA0.predict(X_t) * delta
            self.s0.fit(X_t[Z_t == 0], (Y_t[Z_t == 0] - A_t[Z_t == 0] * delta[Z_t == 0] - v_mean[Z_t == 0]) ** 2)
            self.s1.fit(X_t[Z_t == 1], (Y_t[Z_t == 1] - A_t[Z_t == 1] * delta[Z_t == 1] - v_mean[Z_t == 1]) ** 2)
      
    # ate estimate
    def estimate_tau_mriv(self) -> float:
        return float(np.mean(self.phi[self.burn_in+1:]))
    
    def estimate_tau_dm(self) -> float:
        return float(np.mean(self.delta_dm[self.burn_in+1:]))


@dataclasses.dataclass
class A2IPWExperiment:
    generator:       DataGenerator
    n_rounds:        int
    burn_in:         int = 100
    adaptive:        bool = True
    trunc_schedule:  callable[[int], int] = lambda t: 100
    batch_size:      int = 1
    factories:       dict[str, LearnerFactory | RegressorMixin] = dataclasses.field(default_factory=dict)
    learner_kwargs:  dict[str, dict] = dataclasses.field(default_factory=dict)

    # nuisance learners
    mu0: RegressorMixin = dataclasses.field(init=False)
    mu1: RegressorMixin = dataclasses.field(init=False)
    s0:  RegressorMixin = dataclasses.field(init=False)
    s1:  RegressorMixin = dataclasses.field(init=False)

    # logs
    X: np.ndarray = dataclasses.field(init=False)
    A: np.ndarray = dataclasses.field(init=False)
    Y: np.ndarray = dataclasses.field(init=False)
    pi_t: np.ndarray = dataclasses.field(init=False)  # keeps track of estimated pi_t
    phi: np.ndarray = dataclasses.field(init=False) 

    def __post_init__(self):
        default = lambda **kw: make_rf_factory(regression=True)(**kw)
        def build(name, fallback):
            spec = self.factories.get(name, fallback)
            kw   = self.learner_kwargs.get(name, {})
            return spec(**kw) if not isinstance(spec, RegressorMixin) else spec
        self.mu0 = build("mu0", default)
        self.mu1 = build("mu1", default)
        self.s0  = build("s0", default)
        self.s1  = build("s1", default)

        self.X = np.empty((self.n_rounds, 0))   # will shape on first obs
        self.A = np.empty(self.n_rounds, dtype=int)
        self.Y = np.empty(self.n_rounds, dtype=float)
        self.phi = np.empty(self.n_rounds, dtype=float)
        self.pi_t = np.empty(self.n_rounds, dtype=float)

    # helper: append one observation
    def _append_obs(self, t: int, x: np.ndarray, a: int, y: float):
        if t == 0:                 # first sample ⇒ know d
            self.X = np.empty((self.n_rounds, x.shape[0]))
        self.X[t] = x
        self.A[t] = a
        self.Y[t] = y

    # crude Neyman allocation for *treatment* A (ignoring instrument!)
    def _assign(self, x: np.ndarray, t: int):
        if t <= self.burn_in or self.adaptive == False:
            self.pi_t[t] = 0.5
            return 0.5, np.random.binomial(1, 0.5)
        v0, v1 = self._plug_in_sigmas(x)
        raw = np.sqrt(v1) / (np.sqrt(v0) + np.sqrt(v1))
        k_t = max(2, self.trunc_schedule(t))
        pi   = np.clip(raw, 1 / k_t, 1 - 1 / k_t)
        self.pi_t[t] = pi
        return pi, np.random.binomial(1, pi)
    
    def _plug_in_sigmas(self, x: np.ndarray) -> tuple[float, float]:
        x_input = x.reshape(1, -1)
        v0 = self.s0.predict(x_input)[0]
        v1 = self.s1.predict(x_input)[0]
        # ensure positive variance
        v0 = v0 if v0 > 0 else 1e-3
        v1 = v1 if v1 > 0 else 1e-3
        return v0, v1

    def _calculate_phi(self, t: int, x: np.ndarray, a: int, y: float, pi: float):
        if t <= self.burn_in:
            self.phi[t] = 0.0
            return
        else:
            x_input = x.reshape(1, -1)
            delta = self.mu1.predict(x_input)[0] - self.mu0.predict(x_input)[0]
            v_mean = self.mu1.predict(x_input)[0] if a==1 else self.mu0.predict(x_input)[0]
            phi_calc = ((2 * a - 1) / (a * pi + (1 - a) * (1 - pi))) \
                * (y - v_mean) + delta
            self.phi[t] = phi_calc

    def collect(self, restart: bool = False) -> None:
        if restart:
            self.X = np.empty((0,))
            self.A = np.empty(self.n_rounds, dtype=int)
            self.Y = np.empty(self.n_rounds, dtype=float)
            self.phi = np.empty(self.n_rounds, dtype=float)  # for ATE estimation
            self.pi_t = np.empty(self.n_rounds, dtype=float)  # keeps track of estimated pi_t
        else:
            if self.X.size > 0:
                raise ValueError("Data already collected. Use restart=True to reset.")
        for t in range(self.n_rounds):
            # draw a fresh unit with all potentials
            x, a0, a1, y0, y1 = self.generator()

            # instrument assignment
            pi, z = self._assign(x, t)

            # realised treatment A(Z)
            a_realised = a1 if z == 1 else a0

            # realised outcome  Y(A)
            y_realised = y1 if a_realised == 1 else y0

            # log the observation
            self._append_obs(t, x, a_realised, y_realised)
            self._calculate_phi(t, x, a_realised, y_realised, pi)

            # periodic nuisance refit
            if t >= self.burn_in and t%self.batch_size == 0:
                #print(f"Refitting at round {t}...")
                self._refit(t)

    def _refit(self, t: int):
        """
        Re-train all nuisance learners using data indices 0 … t (inclusive).
        Works with the pre-allocated arrays X, Z, A, Y of length n_rounds.
        """
        # slice of currently-available data
        idx   = slice(0, t + 1)          # 0 … t
        X_t   = self.X[idx]
        A_t   = self.A[idx]
        Y_t   = self.Y[idx]

        # ---------------- base learners ----------------
        self.mu0.fit(X_t[A_t == 0], Y_t[A_t == 0])
        self.mu1.fit(X_t[A_t == 1], Y_t[A_t == 1])
        v_mean0 = self.mu0.predict(X_t)
        v_mean1 = self.mu1.predict(X_t)
        self.s0.fit(X_t[A_t == 0], (Y_t[A_t == 0] - v_mean0[A_t == 0]) ** 2)
        self.s1.fit(X_t[A_t == 1], (Y_t[A_t == 1] - v_mean1[A_t == 1]) ** 2)

    def estimate_tau(self):
        return self.phi[self.burn_in:].mean()
