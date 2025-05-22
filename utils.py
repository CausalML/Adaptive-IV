"""
Utility functions for generating data and creating learners.
"""
from __future__ import annotations
from typing import Callable, Protocol, Tuple
import numpy as np
from sklearn.base import RegressorMixin, clone, ClassifierMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

# ─────────────────────────
# Protocols / type aliases
# ─────────────────────────
class DataGenerator(Protocol):
    def __call__(self) -> Tuple[
        np.ndarray,        # X      : covariates
        int, int,          # A(0), A(1) : potential treatments
        float, float       # Y(0), Y(1) : potential outcomes
    ]: ...

LearnerFactory = Callable[..., RegressorMixin]   # returns *unfitted* learner

# ─────────────────────────
#   tiny wrappers / helpers
# ─────────────────────────
class _ProbaWrapper(RegressorMixin):
    """
    Wraps any SK-Learn classifier so `.predict(X)` returns class-1
    probabilities – making it interchangeable with a regressor that
    estimates E[A | Z,X].
    """
    def __init__(self, clf: ClassifierMixin):
        self.clf = clf
    def fit(self, X, y): self.clf.fit(X, y); return self
    def predict(self, X): return self.clf.predict_proba(X)[:, 1]

def _sk_factory(base_cls) -> LearnerFactory:
    """Return a no-arg factory that clones a fresh estimator each call."""
    return lambda **kw: clone(base_cls(**kw))

# ready-made regressor factories
def make_rf_factory(regression: bool = True, **defaults) -> LearnerFactory:
    """
    Random-forest factory:
      • regression=True  -> returns fresh RandomForestRegressor
      • regression=False -> returns fresh _ProbaWrapper(RandomForestClassifier)
    Extra kwargs are passed to the underlying forest.
    """
    if regression:
        return lambda **kw: RandomForestRegressor(**(defaults | kw))
    else:
        return lambda **kw: _ProbaWrapper(RandomForestClassifier(**(defaults | kw)))

def make_knn_factory(regression: bool = True, k: int = 10, **defaults) -> LearnerFactory:
    if regression:
        return lambda **kw: KNeighborsRegressor(n_neighbors=k, **(defaults | kw))
    else:
        return lambda **kw: _ProbaWrapper(KNeighborsClassifier(n_neighbors=k, **(defaults | kw)))

# PyTorch regressor
try:
    import torch, torch.nn as nn, torch.optim as optim
except ImportError:
    torch = None    # allows code to import utils without torch installed

class TorchRegressor(RegressorMixin):
    """Tiny 1-hidden-layer MLP wrapped as a Scikit-Learn regressor."""
    def __init__(self, d:int, epochs:int=300, lr:float=1e-3, width:int=64):
        self.d, self.epochs, self.lr, self.width = d, epochs, lr, width
        self.net = None
    def fit(self, X, y):
        if torch is None:
            raise ImportError("Install torch to use TorchRegressor.")
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1,1)
        self.net = nn.Sequential(
            nn.Linear(self.d, self.width), nn.ReLU(),
            nn.Linear(self.width, 1))
        opt = optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        for _ in range(self.epochs):
            opt.zero_grad(); loss_fn(self.net(X), y).backward(); opt.step()
        return self
    def predict(self, X):
        with torch.no_grad():
            return self.net(torch.tensor(X, dtype=torch.float32)).cpu().numpy().ravel()

def make_torch_factory(input_dim:int, **kw) -> LearnerFactory:
    if torch is None:
        raise ImportError("PyTorch not installed.")
    return lambda **_: TorchRegressor(d=input_dim, **kw | _)
