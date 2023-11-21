from numba import njit, void, float64
from numpy import double, empty, square, subtract
from numpy.typing import NDArray
from scipy.linalg import solve_triangular
from typing import TYPE_CHECKING

from dagflow.exception import TypeFunctionError
from dagflow.input_extra import MissingInputAddOne
from dagflow.nodes import FunctionNode
from dagflow.typefunctions import check_inputs_multiplicable_mat

if TYPE_CHECKING:
    from dagflow.input import Input
    from dagflow.output import Output


@njit(
    void(float64[:], float64[:], float64[:], float64[:]),
    cache=True,
)
def _chi2_1d(
    data: NDArray[double],
    theory: NDArray[double],
    errors: NDArray[double],
    result: NDArray[double],
) -> None:
    result[0] = 0.0
    for i in range(len(data)):
        result[0] += ((theory[i] - data[i]) / errors[i]) ** 2


class Chi2(FunctionNode):
    r"""
    $\Chi^{2}$ node

    inputs:
        `0` or `data`: data (1d),
        `1` or `theory`: theory (1d),
        `2` or `errors`: errors (1d or 2d).

    outputs:
        `0` or `result`: the resulting array with only one element.

    extra arguments:
        `lower` (bool): True if the errors is lower triangular matrix else upper.
    """

    __slots__ = ("_data", "_theory", "_errors", "_result", "_lower", "_buffer")

    _data: "Input"
    _theory: "Input"
    _errors: "Input"
    _result: "Output"
    _buffer: NDArray
    _lower: bool

    def __init__(self, name, *args, lower: bool = True, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddOne())
        super().__init__(name, *args, **kwargs)
        self.labels.setdefaults(
            {
                "text": r"$\Chi^{2}$",
                "plottitle": r"$\Chi^{2}$",
                "latex": r"$\Chi^{2}$",
                "axis": r"$\Chi^{2}$",
            }
        )
        self._lower = lower
        self._data = self._add_input("data")  # input: 0
        self._theory = self._add_input("theory")  # input: 1
        self._errors = self._add_input("errors")  # input: 2
        self._result = self._add_output("result")  # output: 0
        self._functions.update({1: self._fcn_1d, 2: self._fcn_2d})

    @property
    def lower(self) -> bool:
        return self._lower

    def _fcn_1d(self) -> None:
        _chi2_1d(
            self._theory.data,
            self._data.data,
            self._errors.data,
            self._result.data,
        )

    def _fcn_2d(self) -> None:
        data = self._data.data
        theory = self._theory.data
        errors = self._errors.data
        buffer = self._buffer
        # errors is triangular decomposition of covariance matrix (L)
        subtract(theory.data, data.data, out=buffer)
        solve_triangular(errors.data, buffer, lower=self.lower, overwrite_b=True)
        square(buffer, out=buffer)
        self._result.data[0] = buffer.sum()

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        from dagflow.typefunctions import (
            check_input_dimension,
            check_input_square,
            check_inputs_same_shape,
        )  # fmt: skip

        check_input_dimension(self, ("data", "theory"), 1)
        check_inputs_same_shape(self, ("data", "theory"))
        errors = self._errors
        dim = errors.dd.dim
        if dim == 2:
            check_input_square(self, "errors")
            check_inputs_multiplicable_mat(self, "errors", "data")
        elif dim == 1:
            check_inputs_same_shape(self, ("data", "errors"))
        else:
            raise TypeFunctionError(
                f"errors must be 1d or 2d, but given {dim}d!",
                node=self,
                input=errors,
            )
        self.fcn = self._functions[dim]

        self._result.dd.shape = (1,)
        self._result.dd.dtype = self._data.dd.dtype

    def _post_allocate(self) -> None:
        # NOTE: buffer is needed only for 2d case
        if self._errors.dd.dim == 2:
            datadd = self._data.dd
            self._buffer = empty(shape=datadd.shape, dtype=datadd.dtype)
