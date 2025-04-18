from __future__ import annotations

from math import log
from typing import TYPE_CHECKING

from numba import njit

from dagflow.core.input_strategy import AddNewInputAddAndKeepSingleOutput
from dagflow.core.node import Node
from dagflow.core.type_functions import check_inputs_number_is_divisible_by_N, check_inputs_have_same_shape

if TYPE_CHECKING:
    from numpy import double
    from numpy.typing import NDArray

    from dagflow.core.input import Input
    from dagflow.core.output import Output


@njit(cache=True)
def _poisson_main_add(
    data: NDArray[double],
    theory: NDArray[double],
    poisson: NDArray[double],
) -> None:
    r"""$\sum (theory_i - data_i * log(theory_i))$"""
    sm = 0.0
    for i in range(len(theory)):
        t = theory[i]
        sm += t - log(t) * data[i]

    poisson[0] += sm


class LogPoissonMain(Node):
    r"""
    Calculates the Poisson loglikelihood function value.

    inputs:
        `const`: $\sum \ln data_{i}$ (1 element)
        `i`: data $x$ (N elements)
        `i+1`: theory $\mu$ (N elements)

    outputs:
        `poisson`: $\ln Poisson$ (1 element)

    .. note:: The node must take 2N inputs!

    .. note:: To prepair a `const` input use the `LogPoissonConst`-node.
    """

    __slots__ = ("_data", "_theory", "_const", "_poisson")

    _data: Input
    _theory: Input
    _const: Input
    _poisson: Output

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs, input_strategy=AddNewInputAddAndKeepSingleOutput())
        # TODO: set labels
        self.labels.setdefaults(
            {
                "mark": "lnP"
                #        "text": "",
                #        "plottitle": "",
                #        "latex": "",
                #        "axis": "",
            }
        )
        self._data = self._add_input("data")  # input: 0
        self._theory = self._add_input("theory")  # input: 1
        self._const = self._add_input("const", positional=False)  # input
        self._poisson = self._add_output("poisson")  # output: 0

    def _function(self):
        data = self._poisson._data
        data[0] = 0.0
        i = 0
        while i < self.inputs.len_pos():
            _poisson_main_add(
                self.inputs[i].data.ravel(),
                self.inputs[i + 1].data.ravel(),
                data,
            )
            i += 2
        data[0] = 2.0 * (data[0] + self._const.data[0])  # fmt:skip

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_inputs_number_is_divisible_by_N(self, 2)
        i = 0
        while i < self.inputs.len_pos():
            check_inputs_have_same_shape(self, (i, i + 1))
            i += 2
        self._poisson.dd.shape = (1,)
        self._poisson.dd.dtype = self._data.dd.dtype
