import numpy as np
from sklearn.tree import DecisionTreeRegressor


def build_model(
    max_depth: int = 30,
    random_state: int = 42,
    min_samples_leaf: int = 1,
    min_samples_split: int = 2,
    max_features=None,
    splitter: str = "best",
    separate_heads: bool = False,
):
    """
    Returns
    -------
    model : DecisionTreeRegressor or dict(pos=DecisionTreeRegressor, ori=DecisionTreeRegressor)
        If separate_heads=True, train 2 trees: xyz and cos-angles.
    backend : str
    """

    def _make():
        return DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=random_state,
            criterion="friedman_mse",
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            splitter=splitter,
        )

    if separate_heads:
        model = {"pos": _make(), "ori": _make()}
    else:
        model = _make()

    print(
        "[Model] sklearn DecisionTreeRegressor"
        f"  max_depth={max_depth}"
        f"  min_leaf={min_samples_leaf}"
        f"  min_split={min_samples_split}"
        f"  max_features={max_features}"
        f"  splitter={splitter}"
        f"  separate_heads={separate_heads}"
    )
    return model, "sklearn"


def fit(model, backend: str, X_train: np.ndarray, y_train: np.ndarray):
    if isinstance(model, dict):
        model["pos"].fit(X_train, y_train[:, :3])
        model["ori"].fit(X_train, y_train[:, 3:])
        return model

    model.fit(X_train, y_train)
    return model


def predict(model, backend: str, X: np.ndarray) -> np.ndarray:
    if isinstance(model, dict):
        p = model["pos"].predict(X)
        o = model["ori"].predict(X)
        return np.concatenate([p, o], axis=1)

    return model.predict(X)
