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
        n_modified_pars = 0
        n_pars = len(self._parameters)
        modified_pars = []
        modified_pars_numbers = []
        for i, (param, val) in enumerate(zip(self._parameters, values)):
            if (param_modified:=(param.value!=val)):
                n_modified_pars+=1

                if n_modified_pars<3:
                    modified_pars.append(param.name)
                    modified_pars_numbers.append(i)
            param.value = val
            self._logger.info(f"{param!s}{param_modified and ' *' or ''}")

        self._ncall += 1

        ret = self._statistic.data[0]

        if n_modified_pars<n_pars:
            calctype = ": derivative"
        else:
            if self._ncall==1:
                calctype = n_pars>1 and ": step/hessian" or ""
            else:
                calctype = n_pars>2 and ": step/hessian" or ""

        modified_pars_numbers = list(map(str, modified_pars_numbers))
        if n_modified_pars>len(modified_pars):
            modified_pars.append("...")
            modified_pars_numbers.append("...")

        self._logger.info(f"Modified parameters {n_modified_pars}/{n_pars}{calctype}")
        self._logger.info(f"Modified parameters: {', '.join(modified_pars)}")
        self._logger.info(f"Modified parameters: {', '.join(modified_pars_numbers)}")
        self._logger.info(f"Statistic {self._ncall}: {ret}")
        self._logger.info("")

        return ret

    def __call__(self, values: "NDArray") -> float:
        return self._function(values)
