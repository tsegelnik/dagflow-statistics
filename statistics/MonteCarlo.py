from typing import Literal, Tuple

from numba import float64, njit, void
from numpy import add, double, matmul, sqrt
from numpy.random import normal, poisson
from numpy.typing import NDArray

from dagflow.exception import InitializationError
from dagflow.lib import BlockToOneNode
from dagflow.typefunctions import (
    check_input_matrix_or_diag,
    check_inputs_multiplicable_mat,
    check_inputs_multiplicity,
    check_outputs_number,
    copy_from_input_to_output,
)

MonteCarloModes = {"Asimov", "Normal", "NormalStats", "Poisson", "Covariance"}
ModeType = Literal[MonteCarloModes]

MonteCarloModes1 = {"Asimov", "NormalStats", "Poisson"}
ModeType1 = Literal[MonteCarloModes1]

MonteCarloModes2 = {"Normal", "Covariance"}
ModeType2 = Literal[MonteCarloModes2]


def _covariance_L(
    mean: NDArray[double], cov_L: NDArray[double], result: NDArray[double]
) -> None:
    if cov_L.ndim == 1:
        _covariance_L_1d(mean, cov_L, result)
    else:
        _covariance_L_2d(mean, cov_L, result)


@njit(void(float64[:], float64[:], float64[:]), cache=True)
def _covariance_L_1d(
    mean: NDArray[double], cov_L: NDArray[double], result: NDArray[double]
) -> None:
    for i in range(len(result)):
        result[i] = mean[i] + cov_L[i] * normal()


@njit(void(float64[:]), cache=True)
def _fill_normal(data: NDArray[double]) -> None:
    for i in range(len(data)):
        data[i] = normal()


def _covariance_L_2d(
    mean: NDArray[double], cov_L: NDArray[double], result: NDArray[double]
) -> None:
    _fill_normal(result)
    matmul(cov_L, result, out=result)
    add(result, mean, out=result)


@njit(void(float64[:], float64[:], float64[:]), cache=True)
def _normal(
    mean: NDArray[double], errors: NDArray[double], result: NDArray[double]
) -> None:
    for i in range(len(result)):
        result[i] = mean[i] + errors[i] * normal()


@njit(void(float64[:], float64[:]), cache=True)
def _normal_stats(mean: NDArray[double], result: NDArray[double]) -> None:
    func = lambda x: x + sqrt(x) * normal()
    for i in range(len(result)):
        result[i] = func(mean[i])


@njit(void(float64[:], float64[:]), cache=True)
def _poisson(mean: NDArray[double], result: NDArray[double]):
    for i in range(len(result)):
        result[i] = poisson(mean[i])


class MonteCarlo(BlockToOneNode):
    r"""
    Generates a random sample distributed according different modes.

    inputs:
        `i`: average model vector
        `i+1`: model uncertainties vector (sigma); *only for `Normal` and `Covariance`*

    outputs:
        `i`: generated sample

    extra arguments:
        `mode`:
            * `Asimov`: store input data without fluctuations
            * `Normal`: normal distribution without correlations (2 inputs)
            * `NormalStats`: normal distribution without correlations (1 input)
            * `Poisson`: uses Poisson distribution
            * `Covariance`: multivariate normal distribution using L-decomposition of the covariance matrix
    """

    __slots__ = ("_mode",)

    _mode: ModeType

    def __new__(cls, name, mode: ModeType, *args, _baseclass: bool = True, **kwargs):
        if not _baseclass:
            return super().__new__(cls, *args, **kwargs)
        if mode in MonteCarloModes1:
            return MonteCarlo1(name, mode, *args, _baseclass=False, **kwargs)
        elif mode in MonteCarloModes2:
            return MonteCarlo2(name, mode, *args, _baseclass=False,  **kwargs)

        raise RuntimeError(f"Invalid montecarlo mode {mode}. Expect: {MonteCarloModes}")

    @property
    def mode(self) -> str:
        return self._mode

    def next_sample(self) -> None:
        self.unfreeze()
        self.taint(force=True)


class MonteCarlo1(MonteCarlo):
    r"""
    Generates a random sample distributed according different modes.

    inputs:
        `i`: average model vector

    outputs:
        `i`: generated sample

    extra arguments:
        `mode`:
            * `Asimov`: store input data without fluctuations
            * `NormalStats`: normal distribution without correlations (1 input)
            * `Poisson`: uses Poisson distribution
    """

    __slots__ = ()

    def __init__(
        self,
        name,
        mode: ModeType1,
        *args,
        _baseclass: bool = True,
        **kwargs,
    ):
        if not mode in MonteCarloModes1:
            raise RuntimeError(f"Invalid montecarlo mode {mode}. Expect: {MonteCarloModes1}")

        self._mode = mode
        super().__init__(name, *args, **kwargs)
        # TODO: set lables
        # self.labels.setdefaults(
        #    {
        #        "text": "MonteCarlo sample",
        #        "plottitle": "MonteCarlo sample",
        #        "latex": "MonteCarlo sample",
        #        "axis": "MonteCarlo sample",
        #    }
        # )
        self._functions.update(
            {
                "Asimov": self._fcn_asimov,
                "NormalStats": self._fcn_normal_stats,
                "Poisson": self._fcn_poisson,
            }
        )

    @staticmethod
    def _input_names() -> Tuple[str, ...]:
        return "data",

    def _fcn_asimov(self) -> None:
        for _input, _output in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            _output[:] = _input[:]

    def _fcn_normal_stats(self) -> None:
        for _input, _output in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            _normal_stats(_input, _output)

    def _fcn_poisson(self) -> None:
        for _input, _output in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            _poisson(_input, _output)

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        n = self.inputs.len_pos()
        check_outputs_number(self, n)
        for i in range(n):
            copy_from_input_to_output(self, i, i)

        self.fcn = self._functions[self.mode]

class MonteCarlo2(MonteCarlo):
    r"""
    Generates a random sample distributed according different modes.

    inputs:
        `i`: average model vector
        `i+1`: model uncertainties vector (sigma); *only for `Normal` and `Covariance`*

    outputs:
        `i`: generated sample

    extra arguments:
        `mode`:
            * `Normal`: normal distribution without correlations (2 inputs)
            * `Covariance`: multivariate normal distribution using L-decomposition of the covariance matrix
    """

    def __init__(
        self,
        name,
        mode: ModeType2,
        *args,
        _baseclass: bool = True,
        **kwargs,
    ):
        if not mode in MonteCarloModes2:
            raise RuntimeError(f"Invalid montecarlo mode {mode}. Expect: {MonteCarloModes2}")

        self._mode = mode
        super().__init__(name, *args, **kwargs)
        # TODO: set lables
        # self.labels.setdefaults(
        #    {
        #        "text": "MonteCarlo sample",
        #        "plottitle": "MonteCarlo sample",
        #        "latex": "MonteCarlo sample",
        #        "axis": "MonteCarlo sample",
        #    }
        # )
        if mode not in MonteCarloModes:
            raise InitializationError(
                f"mode must be in {MonteCarloModes}, but given {mode}",
                node=self,
            )
        self._functions.update(
            {
                "Normal": self._fcn_normal,
                "Covariance": self._fcn_covariance_L,
            }
        )

    @staticmethod
    def _input_names() -> Tuple[str, ...]:
        return "data", "errors"

    def _fcn_covariance_L(self) -> None:
        i = 0
        while i < self.inputs.len_pos():
            _covariance_L(
                self.inputs[i].data, self.inputs[i + 1].data, self.outputs[i // 2].data
            )
            i += 2

    def _fcn_normal(self) -> None:
        i = 0
        while i < self.inputs.len_pos():
            _normal(
                self.inputs[i].data, self.inputs[i + 1].data, self.outputs[i // 2].data
            )
            i += 2

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_inputs_multiplicity(self, 2)
        n = self.inputs.len_pos()
        check_outputs_number(self, n // 2)

        if self.mode == "Covariance":
            check_input_matrix_or_diag(self, slice(1, n, 2), check_square=True)

        for i in range(n // 2):
            check_inputs_multiplicable_mat(self, i, i + 1)
            copy_from_input_to_output(self, 2 * i, i)

        self.fcn = self._functions[self.mode]

