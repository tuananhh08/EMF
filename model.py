import numpy as np
from sklearn.tree import DecisionTreeRegressor


def build_model(max_depth: int = 30, random_state: int = 42):
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=random_state,
        criterion="friedman_mse",
    )
    print(f"[Model] sklearn DecisionTreeRegressor  max_depth={max_depth}")
    return model, "sklearn"


def fit(model, backend: str, X_train: np.ndarray, y_train: np.ndarray):
    
    model.fit(X_train, y_train)
    return model


def predict(model, backend: str, X: np.ndarray) -> np.ndarray:

    return model.predict(X)