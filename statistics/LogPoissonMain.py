from math import log
from typing import TYPE_CHECKING

from numba import float64, njit, void
from numpy import double
from numpy.typing import NDArray

from dagflow.input_extra import MissingInputAddOne
from dagflow.nodes import FunctionNode
from dagflow.typefunctions import (
    check_inputs_multiplicity,
    check_inputs_same_shape,
)

if TYPE_CHECKING:
    from dagflow.input import Input
    from dagflow.output import Output


@njit(void(float64[:], float64[:], float64[:]), cache=True)
def _poisson_main(
    theory: NDArray[double],
    data: NDArray[double],
    poisson: NDArray[double],
) -> None:
    r"""$\sum (theory_i - data_i * log(theory_i))$"""
    func = lambda x, y: x - log(x) * y
    for i in range(len(theory)):
        poisson[0] += func(theory[i], data[i])


class LogPoissonMain(FunctionNode):
    r"""
    Calculates the Poisson loglikelihood function value.

    inputs:
        `const`: $\sum \ln data_{i}$ (1 element)
        `i`: theory $\mu$ (N elements)
        `i+1`: data $x$ (N elements)

    outputs:
        `poisson`: $\ln Poisson$ (1 element)

    .. note:: The node must take 2N inputs!

    .. note:: To prepair a `const` input use the `LogPoissonConst`-node.
    """

    __slots__ = ("_theory", "_data", "_const", "_poisson")

    _theory: "Input"
    _data: "Input"
    _const: "Input"
    _poisson: "Output"

    def __init__(self, name, *args, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddOne())
        super().__init__(name, *args, **kwargs)
        # TODO: set labels
        # self.labels.setdefaults(
        #    {
        #        "text": "",
        #        "plottitle": "",
        #        "latex": "",
        #        "axis": "",
        #    }
        # )
        self._theory = self._add_input("theory")  # input: 0
        self._data = self._add_input("data")  # input: 1
        self._const = self._add_input("const", positional=False)  # input
        self._poisson = self._add_output("poisson")  # output: 0

    def _fcn(self):
        self._poisson.data[0] = 0.0
        i = 0
        while i < self.inputs.len_pos():
            _poisson_main(
                self.inputs[i].data.ravel(),
                self.inputs[i + 1].data.ravel(),
                self._poisson.data,
            )
            i += 2
        self._poisson.data[0] = 2.0 * (self._poisson.data[0] + self._const.data[0])  # fmt:skip

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_inputs_multiplicity(self, 2)
        i = 0
        while i < self.inputs.len_pos():
            check_inputs_same_shape(self, (i, i + 1))
            i += 2
        self._poisson.dd.shape = (1,)
        self._poisson.dd.dtype = self._data.dd.dtype
