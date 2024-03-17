from collections.abc import Sequence
from typing import Any

from dagflow.exception import InitializationError
from dagflow.output import Output

from .fitresult import FitResult
from .Minimizable import Minimizable
from .minpars import MinPars

# if we cannot import runtime_error from root we use DagflowError to avoid any exception capture,
# i.e., if CppRuntimeError==DagflowError, the exception will be not raised
try:
    import ROOT  # fmt: skip
    CppRuntimeError = ROOT.std.runtime_error
except Exception:
    from dagflow.exception import DagflowError  # fmt:skip
    CppRuntimeError = DagflowError


class MinimizerBase:
    __slots__ = (
        "_name",
        "_label",
        "_minimizable",
        "_minimizer",
        "_parspecs",
        "_startvalues",
        "_result",
        "_statistic",
        "_verbose",
    )

    _name: str
    _label: str
    _minimizable: Minimizable | None
    _parspecs: MinPars
    _startvalues: list[float] | None
    _result: dict
    _minimizer: Any
    _verbose: bool

    def __init__(
        self,
        statistic: Output,
        minpars: MinPars,
        name: str,
        label: str,
        verbose: bool = False,
        startvalues: list[float] | None = None,
    ):
        if not isinstance(statistic, Output):
            raise InitializationError(
                f"arg 'statistic' must be an Output, but given {type(statistic)=}, {statistic=}."
            )
        self._statistic = statistic
        if not isinstance(minpars, MinPars):
            raise InitializationError(
                f"arg 'minpars' must be a MinPars, but given {type(minpars)=}, {minpars=}."
            )
        self._parspecs = minpars
        if startvalues is not None:
            if not isinstance(startvalues, Sequence) or not all(
                isinstance(val, float) for val in startvalues
            ):
                raise InitializationError(
                    f"'startvalues' must be Sequence[float], but given {startvalues=}"
                )
            if len(startvalues) != len(minpars):
                raise InitializationError(
                    "Sizes of 'startvalues' and 'minpars' must coincide, "
                    f"but given {len(startvalues)=}, {len(minpars)=}"
                )
        self._startvalues = startvalues
        self._name = name
        self._label = label
        self._verbose = verbose
        self._minimizable = None

    @property
    def statistic(self) -> Output:
        return self._statistic

    @property
    def name(self) -> str:
        return self._name

    @property
    def label(self) -> str:
        return self._label

    @statistic.setter
    def statistic(self, statistic) -> None:
        self._statistic = statistic
        self._minimizable = None

    @property
    def parspecs(self) -> MinPars:
        return self._parspecs

    @parspecs.setter
    def parspecs(self, parspecs) -> None:
        self._parspecs = parspecs

    @property
    def result(self) -> dict:
        return self._result

    def fit(self, **kwargs) -> dict:
        if len(self.parspecs) == 0:
            return self.evalstatistic()
        return self._child_fit(**kwargs)

    def evalstatistic(self) -> dict:
        with FitResult() as fr:
            fun = self._statistic.data

            fr.set(
                x=[],
                errors=[],
                fun=fun,
                success=True,
                message="stastitics evaluation (no parameters)",
                minimizer="none",
                nfev=1,
            )
            self._result = fr.result
            self.patchresult()

        return self.result

    def _child_fit(
        self,
        *,
        profile_errors: Sequence | None = None,
        scan: Sequence | None = None,
        covariance: bool = False,
    ) -> dict:
        raise NotImplementedError("The method must be overriden!")

    def patchresult(self) -> None:
        names = list(self._parspecs.names())
        result = self._result
        # result["npars"] = self._parspecs.nvariable()
        # result["nfree"] = self._parspecs.nfree()
        # result["nconstrained"] = self._parspecs.nconstrained()
        # fixed = result["fixed"] = self._parspecs.fixed()
        # result["nfixed"] = len(fixed)
        result["x"] = result.pop("x")
        result["errors"] = result.pop("errors")
        result["names"] = names
        result["xdict"] = dict(zip(names, (float(x) for x in self.result["x"])))
        if self.result["errors"] is not None:
            result["errorsdict"] = dict(zip(names, (float(e) for e in self.result["errors"])))
        else:
            result["errorsdict"] = {}

    def update_minimizable(self) -> Minimizable:
        if self._minimizable is None or self.parspecs.resized:
            self._minimizable = Minimizable(self.statistic, verbose=self._verbose)
            for parspec in self.parspecs.specs():
                self._minimizable.append_par(parspec.par)
        return self._minimizable
