def build_model(max_depth: int = 30, random_state: int = 42):
    try:
        from cuml.tree import DecisionTreeRegressor
        import cupy as cp
        cp.zeros(1)   
        model   = DecisionTreeRegressor(max_depth=max_depth,
                                        random_state=random_state)
        backend = "cuml"
        print(f"[Model] cuML DecisionTreeRegressor  max_depth={max_depth}  (GPU)")

    except Exception:
        from sklearn.tree import DecisionTreeRegressor
        model   = DecisionTreeRegressor(max_depth=max_depth,
                                        random_state=random_state,
                                        criterion="friedman_mse")
        backend = "sklearn"
        print(f"[Model] sklearn DecisionTreeRegressor  max_depth={max_depth}  (CPU)")

    return model, backend


def fit(model, backend: str, X_train, y_train):
    import numpy as np
    if backend == "cuml":
        import cudf
        X_cu = cudf.DataFrame(X_train.astype(np.float32))
        y_cu = cudf.DataFrame(y_train.astype(np.float32))
        model.fit(X_cu, y_cu)
    else:
        model.fit(X_train, y_train)
    return model


def predict(model, backend: str, X) -> "np.ndarray":

    import numpy as np
    if backend == "cuml":
        import cudf, cupy as cp
        X_cu = cudf.DataFrame(X.astype(np.float32))
        out  = model.predict(X_cu)
        return cp.asnumpy(out.values) if hasattr(out, "values") else cp.asnumpy(out)
    else:
        return model.predict(X)