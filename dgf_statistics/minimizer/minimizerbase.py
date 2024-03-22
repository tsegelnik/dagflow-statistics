from typing import Any

from dagflow.exception import InitializationError

from .fitresult import FitResult
from .Minimizable import Minimizable
from collections.abc import Sequence
from dagflow.parameters import Parameter
from dagflow.output import Output

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
        "_parameters",
        "_result",
        "_statistic",
        "_verbose",
    )

    _name: str
    _label: str
    _minimizable: Minimizable | None
    _parameters: list[Parameter]
    _result: dict
    _minimizer: Any
    _verbose: bool
    _statistic: Output

    def __init__(
        self,
        statistic: Output,
        parameters: Sequence[Parameter],
        name: str,
        label: str,
        verbose: bool = False,
    ):
        if not isinstance(statistic, Output):
            raise InitializationError(
                f"arg 'statistic' must be an Output, but given {type(statistic)=}, {statistic=}."
            )
        self._statistic = statistic
        self._parameters = []  # pyright: ignore
        if parameters:
            if not isinstance(parameters, Sequence):
                raise InitializationError(
                    f"parameters must be a sequence of GaussianParameter, but given {parameters=},"
                    f" {type(parameters)=}!"
                )
            for par in parameters:
                self.append_par(par)
        self._name = name
        self._label = label
        self._verbose = verbose
        self._minimizable = None

    @property
    def statistic(self) -> Output:
        return self._statistic

    @property
    def parameters(self) -> list[Parameter]:
        return self._parameters

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
    def parameters(self) -> list[Parameter]:
        return self._parameters

    @parameters.setter
    def parameters(self, parameters) -> None:
        self.parameters = parameters

    @property
    def result(self) -> dict:
        return self._result

    def append_par(self, par: Parameter) -> None:
        if not isinstance(par, Parameter):
            raise RuntimeError(f"par must be a Parameter, but given {par=}, {type(par)=}!")
        self._parameters.append(par)

    def fit(self, **kwargs) -> dict:
        if len(self.parameters) == 0:
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

    def _child_fit(self, *, covariance: bool = False) -> dict:
        raise NotImplementedError("The method must be overriden!")

    def patchresult(self) -> None:
        names = [par.output.node.name for par in self.parameters]
        result = self._result
        result["npars"] = len(self.parameters)
        # result["nfree"] = self.parameters.nfree()
        # result["nconstrained"] = self.parameters.nconstrained()
        # fixed = result["fixed"] = self.parameters.fixed()
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
        if self._minimizable is None:
            self._minimizable = Minimizable(self.statistic, verbose=self._verbose)
            for par in self.parameters:
                self._minimizable.append_par(par)
        return self._minimizable
