from time import time, perf_counter


class FitResult:
    __slots__ = ("_result", "_wall", "_clock")

    _result: dict
    _wall: float
    _clock: float

    def __init__(self):
        self._result = {}

    @property
    def result(self):
        return self._result

    def __enter__(self):
        self._wall = time()
        self._clock = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._clock = perf_counter() - self._clock
        self._wall = time() - self._wall

    def set(self, x, errors, fun, success, message, minimizer, nfev, **kwargs):
        result = self._result

        result["fun"] = fun
        result["success"] = success
        result["message"] = message
        result["nfev"] = nfev
        result["minimizer"] = minimizer
        result["clock"] = self._clock
        result["wall"] = self._wall
        result["x"] = x
        result["errors"] = errors
        hess_inv = result["hess_inv"] = kwargs.pop("hess_inv", None)
        result["jac"] = kwargs.pop("jac", None)

        if errors is None and hess_inv is not None:
            from numpy import diag  # fmt: skip
            result["errors"] = diag(hess_inv) * 2.0

        result.update(kwargs)
