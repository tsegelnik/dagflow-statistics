from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from dagflow.core.exception import InitializationError
from dagflow.tools.logger import Logger, get_logger
from dagflow.core.output import Output
from dagflow.parameters import Parameter

from .fitresult import FitResult
from .Minimizable import Minimizable

# if we cannot import runtime_error from root we use DagflowError to avoid any exception capture,
# i.e., if CppRuntimeError==DagflowError, the exception will be not raised
try:
    import ROOT  # fmt: skip
    CppRuntimeError = ROOT.std.runtime_error
except Exception:
    from dagflow.core.exception import DagflowError  # fmt:skip
    CppRuntimeError = DagflowError

if TYPE_CHECKING:
    from typing import Any


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
        "_logger",
        "_initial_parameters",
    )

    _name: str
    _label: str
    _minimizable: Minimizable | None
    _parameters: list[Parameter]
    _result: dict
    _minimizer: Any
    _verbose: bool
    _statistic: Output
    _logger: Logger
    _initial_parameters: dict[Parameter, float] | None

    def __init__(
        self,
        statistic: Output,
        parameters: Sequence[Parameter],
        name: str,
        label: str,
        verbose: bool = False,
        logger: Logger | None = None,
    ):
        if not isinstance(statistic, Output):
            raise InitializationError(
                f"arg 'statistic' must be an Output, but given {type(statistic)=}, {statistic=}."
            )
        self._statistic = statistic
        self._initial_parameters = {}

        self._parameters = []  # pyright: ignore
        if parameters:
            if not isinstance(parameters, Sequence):
                raise InitializationError(
                    f"'parameters' must be a sequence of Parameter, but given {parameters=},"
                    f" {type(parameters)=}!"
                )
            for par in parameters:
                self.append_par(par)
                self.copy_initial_values(par)

        if isinstance(logger, Logger):
            self._logger = logger
        elif logger is not None:
            raise InitializationError(f"Cannot initialize a Minimizable class with logger={logger}")
        else:
            self._logger = get_logger()

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

    @property
    def logger(self) -> Logger:
        return self._logger

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

    def copy_initial_values(self, par: Parameter) -> None:
        self._initial_parameters.update({par: par.value.copy()})

    def push_initial_values(self) -> None:
        for par, value in self._initial_parameters.items():
            par.push(value)

    def append_par(self, par: Parameter) -> None:
        if not isinstance(par, Parameter):
            raise RuntimeError(f"'par' must be a Parameter, but given {par=}, {type(par)=}!")
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
            summary="stastitics evaluation (no parameters)",
            minimizer="none",
            nfev=1,
        )
        self._result = fr.result
        self.patchresult()

        return self.result

    def _child_fit(self, **_) -> dict:
        raise NotImplementedError("The method must be overriden!")

    def patchresult(self) -> None:
        names = [par.output.node.labels.path for par in self.parameters]
        result = self._result
        result["npars"] = len(self.parameters)
        result["names"] = names
        result["xdict"] = dict(zip(names, (float(x) for x in self.result["x"])))
        if self.result["errors"] is not None:
            result["errorsdict"] = dict(zip(names, (float(e) for e in self.result["errors"])))
        else:
            result["errorsdict"] = {}

    def init_minimizable(self) -> Minimizable:
        if self._minimizable is None:
            self._minimizable = Minimizable(self.statistic, verbose=self._verbose, logger=self._logger)
            for par in self.parameters:
                self._minimizable.append_par(par)
        return self._minimizable
