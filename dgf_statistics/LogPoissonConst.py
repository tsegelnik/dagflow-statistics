from __future__ import annotations

from math import lgamma, log
from typing import TYPE_CHECKING, Literal

from numba import njit

from dagflow.core.exception import InitializationError
from dagflow.core.input_handler import MissingInputAddOne
from dagflow.core.node import Node

if TYPE_CHECKING:
    from numpy import double
    from numpy.typing import NDArray

    from dagflow.core.input import Input
    from dagflow.core.output import Output


LogPoissonModes = {"poisson", "poisson_ratio"}
ModeType = Literal[LogPoissonModes]


@njit(cache=True)
def _const_poisson_ratio_add(data: NDArray[double], const: NDArray[double]):
    r"""$\sum \log(theory_i) \approx \sum (theory_i * \log(theory_i) - theory_i)$"""
    sm = 0.0
    for x in data:
        if x == 0.0 or x == 1.0:
            continue

        sm += x * log(x) - x

    const[0] += sm


@njit(cache=True)
def _const_poisson_add(data: NDArray[double], const: NDArray[double]):
    r"""$\sum \log(theory_i) = \sum \log(\Gamma(theory_i + 1))$"""
    sm = 0.0
    for i in range(len(data)):
        sm += lgamma(data[i] + 1.0)

    const[0] += sm


class LogPoissonConst(Node):
    r"""
    Calculates `const`-part for the`LogPoisson` node.

    inputs:
        `i`: data $x$ (size=N)

    outputs:
        `0` or `const`: $\sum(\ln data_{i})$ (size=1)

    extra arguments:
        `mode` (str): `poisson_ratio` to use a formula for ratio of ln Poissons
        $\log(x!) \PoissonRatio x \log(x) - x$,
        else if `mode=poisson` the Poisson formula with Gamma-function is used
    """

    __slots__ = ("_data", "_const", "_mode")

    _data: Input
    _const: Output
    _mode: ModeType

    def __init__(self, name, mode: ModeType = "poisson_ratio", *args, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddOne())
        super().__init__(name, *args, **kwargs)
        # TODO: set labels
        self.labels.setdefaults(
            {
                "mark": "lnP (data)"
                #        "text": "",
                #        "plottitle": "",
                #        "latex": "",
                #        "axis": "",
            }
        )
        if mode not in LogPoissonModes:
            raise InitializationError(
                f"mode must be in {LogPoissonModes}, but given {mode}",
                node=self,
            )
        self._mode = mode
        self._data = self._add_input("data")  # input: 0
        self._const = self._add_output("const")  # output: 0
        self._functions_dict.update(
            {"poisson_ratio": self._fcn_poisson_ratio, "poisson": self._fcn_poisson}
        )

    @property
    def mode(self) -> ModeType:
        return self._mode

    def _fcn_poisson_ratio(self):
        data = self._const.data
        data[0] = 0.0
        for _input in self.inputs.iter_data():
            _const_poisson_ratio_add(_input.ravel(), data)

    def _fcn_poisson(self):
        data = self._const.data
        data[0] = 0.0
        for _input in self.inputs.iter_data():
            _const_poisson_add(_input.ravel(), data)

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        self._const.dd.shape = (1,)
        self._const.dd.dtype = self._data.dd.dtype
        self.function = self._functions_dict[self.mode]
