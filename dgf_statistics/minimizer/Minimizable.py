from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from dagflow.core.exception import InitializationError
from dagflow.tools.logger import Logger, get_logger
from dagflow.core.output import Output
from dagflow.parameters import Parameter

if TYPE_CHECKING:
    from typing import Callable

    from numpy.typing import NDArray


class Minimizable:
    __slots__ = ("_statistic", "_parameters", "_verbose", "_functions", "_function", "_ncall", "_logger")

    _statistic: Output
    _parameters: list[Parameter]
    _verbose: bool
    _functions: dict
    _function: Callable
    _ncall: int
    _logger: Logger

    def __init__(
        self,
        statistic: Output,
        parameters: Sequence[Parameter] | None = None,
        verbose: bool = False,
        logger: Logger | None = None,
    ) -> None:
        if not isinstance(statistic, Output):
            raise InitializationError(
                f"'statistic' must be an Output, but given {statistic=}, {type(statistic)=}!"
            )
        self._statistic = statistic

        self._parameters = []  # pyright: ignore
        if parameters is not None:
            if not isinstance(parameters, Sequence):
                raise InitializationError(
                    f"'parameters' must be a sequence of Parameter, but given {parameters=},"
                    f" {type(parameters)=}!"
                )
            for par in parameters:
                self.append_par(par)

        if isinstance(logger, Logger):
            self._logger = logger
        elif logger is not None:
            raise InitializationError(f"Cannot initialize a Minimizable class with logger={logger}")
        else:
            self._logger = get_logger()

        self._verbose = verbose
        self._functions = {"verbose": self._function_verbose, "default": self._function_default}
        self._function = self._functions["verbose" if verbose else "default"]
        self._ncall = 0

    def append_par(self, par: Parameter) -> None:
        if not isinstance(par, Parameter):
            raise RuntimeError(f"par must be a Parameter, but given {par=}, {type(par)=}!")
        self._parameters.append(par)

    def _function_default(self, values: "NDArray") -> float:
        for param, val in zip(self._parameters, values):
            param.value = val
        self._ncall += 1
        return self._statistic.data[0]

    def _function_verbose(self, values: "NDArray") -> float:
        for param, val in zip(self._parameters, values):
            param.value = val
            self._logger.info(f"Parameter(name='{param.output.node.name}', value={val})")
        self._ncall += 1
        ret = self._statistic.data[0]
        self._logger.info(f"Statistic({self._ncall}) = {ret}")
        return ret

    def __call__(self, values: "NDArray") -> float:
        return self._function(values)
