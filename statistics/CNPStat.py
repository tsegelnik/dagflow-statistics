from math import sqrt
from numba import float64, njit, void
from numpy import double
from numpy.typing import NDArray

from dagflow.input_extra import MissingInputAddEachN
from dagflow.nodes import FunctionNode
from dagflow.typefunctions import (
    AllPositionals,
    check_input_dimension,
    copy_from_input_to_output,
    check_inputs_same_shape,
    check_inputs_multiplicity,
)


@njit(void(float64[:], float64[:], float64[:]), cache=True)
def _cnp(
    data: NDArray[double],
    theory: NDArray[double],
    result: NDArray[double],
) -> None:
    coeff = sqrt(3.0)
    for i in range(len(data)):
        result[i] = coeff / sqrt(1.0 / data[i] + 2.0 / theory[i])


class CNPStat(FunctionNode):
    r"""
    Combined Neyman–Pearson statistic uncertainty:
        errors = sqrt(3) / sqrt(1/dataᵢ+2/theoryᵢ), so if we connect it to `Chi2` node we get
        χ² = (1/3) Σᵢ [(1/dataᵢ+2/theoryᵢ)·(theoryᵢ-dataᵢ)²]

    inputs:
        `i`: data 1d array
        `i+1`: theory 1d array

    outputs:
        `i`: the resulting array with errors.

    .. note:: The node must have only 2N inputs!
    """

    def __init__(self, name, *args, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddEachN(2))
        super().__init__(name, *args, **kwargs)
        self.labels.setdefaults(
            {
                "text": r"CNP stat. uncertainty",
                "plottitle": r"CNP stat. uncertainty",
                "latex": r"CNP stat. uncertainty",
                "axis": r"CNP stat. uncertainty",
            }
        )

    def _fcn(self) -> None:
        i = 0
        while i < self.inputs.len_pos():
            _cnp(
                self.inputs[i].data,
                self.inputs[i + 1].data,
                self.outputs[i // 2].data,
            )
            i += 2

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_inputs_multiplicity(self, 2)
        check_input_dimension(self, AllPositionals, 1)
        i = 0
        while i < self.inputs.len_pos():
            check_inputs_same_shape(self, (i, i + 1))
            copy_from_input_to_output(self, i, i // 2)
            i += 2
