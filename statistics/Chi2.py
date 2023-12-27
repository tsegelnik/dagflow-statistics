from typing import TYPE_CHECKING, Tuple

from numpy import double, empty, square, subtract
from numpy.typing import NDArray
from scipy.linalg import solve_triangular

from dagflow.exception import TypeFunctionError
from dagflow.inputhandler import MissingInputAdd, SequentialFormatter
from dagflow.node import Input, Output
from dagflow.lib import ManyToOneNode
from dagflow.typefunctions import check_inputs_multiplicable_mat

if TYPE_CHECKING:
    from dagflow.input import Input
    from dagflow.output import Output

from numba import njit


@njit(cache=True)
def _chi2_1d(
    data: NDArray[double],
    theory: NDArray[double],
    errors: NDArray[double],
    result: NDArray[double],
) -> None:
    res = 0.0
    for idata, itheory, ierror in zip(data, theory, errors):
        res += ((itheory - idata) / ierror) ** 2
    result[0] += res


class Chi2(ManyToOneNode):
    r"""
    $\Chi^{2}$ node

    inputs:
        `0` or `data`: data (1d),
        `1` or `theory`: theory (1d),
        `2` or `errors`: errors (1d or 2d).

    outputs:
        `0` or `result`: the resulting array with only one element.

    extra arguments:
        `matrix_is_lower` (bool): True if the errors is lower triangular matrix else upper.
    """

    __slots__ = (
        "_data_tuple",
        "_theory_tuple",
        "_errors_tuple",
        "_result",
        "_matrix_is_lower",
        "_buffer",
    )

    _data: Tuple[Input]
    _theory: Tuple[Input]
    _errors: Tuple[Input]
    _result: Output
    _buffer: NDArray
    _matrix_is_lower: bool

    def __init__(self, name, *args, matrix_is_lower: bool = True, **kwargs):
        kwargs.setdefault(
            "missing_input_handler",
            MissingInputAdd(
                input_fmt=SequentialFormatter(("data", "theory", "errors"))
            ),
        )
        super().__init__(name, *args, output_name="result", **kwargs)
        self.labels.setdefaults(
            {
                "text": r"$\Chi^{2}$",
                "plottitle": r"$\Chi^{2}$",
                "latex": r"$\Chi^{2}$",
                "axis": r"$\Chi^{2}$",
            }
        )
        self._matrix_is_lower = matrix_is_lower
        self._data_tuple = ()  # input: 0
        self._theory_tuple = ()  # input: 1
        self._errors_tuple = ()  # input: 2
        self._result = self.outputs[0]
        self._functions.update({1: self._fcn_1d, 2: self._fcn_2d})

    @property
    def matrix_is_lower(self) -> bool:
        return self._matrix_is_lower

    def _fcn_1d(self) -> None:
        ret = self._result.data
        ret[0] = 0.0

        for theory, data, errors in zip(
            self._theory_tuple, self._data_tuple, self._errors_tuple
        ):
            _chi2_1d(theory.data, data.data, errors.data, ret)

    def _fcn_2d(self) -> None:
        buffer = self._buffer
        ret = 0.0
        for theory, data, errors in zip(
            self._theory_tuple, self._data_tuple, self._errors_tuple
        ):
            # errors is triangular decomposition of covariance matrix (L)
            subtract(theory.data, data.data, out=buffer)
            solve_triangular(
                errors.data, buffer, lower=self.matrix_is_lower, overwrite_b=True
            )
            square(buffer, out=buffer)
            ret += buffer.sum()

        self._result.data[0] = ret

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        self._data_tuple = tuple(input for input in self.inputs[::3])  # input: 0
        self._theory_tuple = tuple(input for input in self.inputs[1::3])  # input: 1
        self._errors_tuple = tuple(input for input in self.inputs[2::3])  # input: 1

        from dagflow.typefunctions import (
            check_input_dimension,
            check_input_square,
            check_inputs_same_shape,
        )

        check_input_dimension(self, slice(0, None, 3), 1)
        check_input_dimension(self, slice(1, None, 3), 1)
        check_inputs_same_shape(self, (0, 1))
        check_inputs_same_shape(self, slice(0, None, 3))
        check_inputs_same_shape(self, slice(1, None, 3))
        check_inputs_same_shape(self, slice(2, None, 3))
        errors = self._errors_tuple[0]
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
        self._result.dd.dtype = self._data_tuple[0].dd.dtype

    def _post_allocate(self) -> None:
        # NOTE: buffer is needed only for 2d case
        if self._errors_tuple[0].dd.dim == 2:
            datadd = self._data_tuple[0].dd
            self._buffer = empty(shape=datadd.shape, dtype=datadd.dtype)
