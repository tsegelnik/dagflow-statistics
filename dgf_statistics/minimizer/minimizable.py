from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from numpy import full_like, where

from dagflow.core.exception import InitializationError
from dagflow.core.output import Output
from dagflow.parameters import Parameter
from dagflow.tools.logger import Logger, get_logger

if TYPE_CHECKING:
    from typing import Callable

    from numpy.typing import NDArray


class Minimizable:
    __slots__ = (
        "_statistic",
        "_parameters",
        "_verbose",
        "_functions",
        "_function",
        "_ncall",
        "_nstep",
        "_previous_fcn_step",
        "_previous_fcn_call",
        "_previous_parameters_step",
        "_previous_parameters_call",
        "_logger",
    )

    _statistic: Output
    _parameters: list[Parameter]
    _verbose: bool
    _functions: dict
    _function: Callable
    _ncall: int
    _nstep: int
    _previous_fcn_step: float | None
    _previous_fcn_call: float | None
    _previous_parameters_step: NDArray | None
    _previous_parameters_call: NDArray | None
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
            raise InitializationError(
                f"Cannot initialize a Minimizable class with logger={logger}"
            )
        else:
            self._logger = get_logger()

        self._verbose = verbose
        self._functions = {
            "verbose": self._function_verbose,
            "default": self._function_default,
        }
        self._function = self._functions["verbose" if verbose else "default"]
        self._ncall = 0
        self._nstep = 0
        self._previous_fcn_step = None
        self._previous_fcn_call = None
        self._previous_parameters_step = None
        self._previous_parameters_call = None

    def append_par(self, par: Parameter) -> None:
        if not isinstance(par, Parameter):
            raise RuntimeError(
                f"par must be a Parameter, but given {par=}, {type(par)=}!"
            )
        self._parameters.append(par)

    def _function_default(self, values: NDArray) -> float:
        for param, val in zip(self._parameters, values):
            param.value = val
        self._ncall += 1
        return self._statistic.data[0]

    def _function_verbose(self, values: NDArray) -> float:
        n_parameters = len(self._parameters)
        if self._previous_parameters_step is None:
            parameters_change_step = full_like(values, 0.0)
            modified_parameters_count = 0
            call_type = "initial"
            self._nstep+=1
            modified_parameter_numbers = ""
            modified_parameters = ""
        else:
            parameters_change_step = values-self._previous_parameters_step
            modified_parameters_idx = where(parameters_change_step)[0]
            modified_parameters_count = modified_parameters_idx.size

            if n_parameters>1:
                if modified_parameters_count==n_parameters:
                    call_type = "step"
                    self._nstep+=1
                elif modified_parameters_count==1:
                    call_type = "derivative"
                else:
                    call_type = "call type determination failed"
            else:
                call_type = "undefined"

            if modified_parameters_idx.size==n_parameters:
                modified_parameter_numbers = "all"
                modified_parameters = "all"

            if (modified_parameters_truncated:=modified_parameters_count>2):
                modified_parameters_idx = modified_parameters_idx[:2]

            modified_parameter_numbers = f"{', '.join(map(str, modified_parameters_idx))}"
            modified_parameters = f"{', '.join(self._parameters[i].name for i in modified_parameters_idx)}"
            if modified_parameters_truncated:
                modified_parameter_numbers = f"{modified_parameter_numbers}, ..."
                modified_parameters = f"{modified_parameters}, ..."

        for i, (param, val, change) in enumerate(zip(self._parameters, values, parameters_change_step)):
            param.value = val
            if change!=0.0:
                self._logger.info(f"{i: 3d} {param!s} change={change:g}")
            else:
                self._logger.info(f"{i: 3d} {param!s}")

        self._ncall += 1

        ret = self._statistic.data[0]

        self._logger.info(f"Modified parameters {modified_parameters_count}/{n_parameters} ({call_type}): {modified_parameter_numbers}; {modified_parameters}")
        self._logger.info(f"Statistic {self._ncall} (step {self._nstep}): {ret}")
        if call_type!="initial":
            fcn_change = ret - self._previous_fcn_step
            self._logger.info(f"Statistic change: {fcn_change:g}")
        self._logger.info("")

        if not call_type=="derivative":
            self._previous_fcn_step = ret
            self._previous_parameters_step = values.copy()

        return ret

    def __call__(self, values: NDArray) -> float:
        return self._function(values)
