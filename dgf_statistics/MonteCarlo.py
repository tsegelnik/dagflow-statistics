from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from numba import njit
from numpy import add, matmul, sqrt

from dagflow.exception import InitializationError
from dagflow.lib import BlockToOneNode
from dagflow.typefunctions import (
    check_input_matrix_or_diag,
    check_inputs_multiplicable_mat,
    check_inputs_multiplicity,
    check_outputs_number,
    copy_from_input_to_output,
)

if TYPE_CHECKING:
    from numpy import double
    from numpy.random import Generator
    from numpy.typing import NDArray

MonteCarloModes = {"asimov", "normal", "normalstats", "poisson", "covariance"}
ModeType = Literal[MonteCarloModes]

MonteCarloModes1 = {"asimov", "normalstats", "poisson"}
ModeType1 = Literal[MonteCarloModes1]

MonteCarloModes2 = {"normal", "covariance"}
ModeType2 = Literal[MonteCarloModes2]


def _covariance_L(
    mean: NDArray[double],
    cov_L: NDArray[double],
    result: NDArray[double],
    gen: Generator,
) -> None:
    if cov_L.ndim == 1:
        _covariance_L_1d(mean, cov_L, result, gen)
    else:
        _covariance_L_2d(mean, cov_L, result, gen)


@njit(cache=True)
def _covariance_L_1d(
    mean: NDArray[double],
    cov_L: NDArray[double],
    result: NDArray[double],
    gen: Generator,
) -> None:
    for i in range(len(result)):
        result[i] = mean[i] + cov_L[i] * gen.normal()


@njit(cache=True)
def _fill_normal(data: NDArray[double], gen: Generator) -> None:
    for i in range(len(data)):
        data[i] = gen.normal()


def _covariance_L_2d(
    mean: NDArray[double],
    cov_L: NDArray[double],
    result: NDArray[double],
    gen: Generator,
) -> None:
    _fill_normal(result, gen)
    matmul(cov_L, result, out=result)
    add(result, mean, out=result)


@njit(cache=True)
def _normal(
    mean: NDArray[double],
    errors: NDArray[double],
    result: NDArray[double],
    gen: Generator,
) -> None:
    for i in range(len(result)):
        result[i] = mean[i] + errors[i] * gen.normal()


@njit(cache=True)
def _normal_stats(
    mean: NDArray[double], result: NDArray[double], gen: Generator
) -> None:
    func = lambda x: x + sqrt(x) * gen.normal()
    for i in range(len(result)):
        result[i] = func(mean[i])


@njit(cache=True)
def _poisson(mean: NDArray[double], result: NDArray[double], gen: Generator):
    for i in range(len(result)):
        result[i] = gen.poisson(mean[i])


class MonteCarlo(BlockToOneNode):
    r"""
    Generates a random sample distributed according different modes.

    inputs:
        `i`: average model vector
        `i+1`: model uncertainties vector (sigma); *only for `normal` and `covariance`*

    outputs:
        `i`: generated sample

    extra arguments:
        `generator`: generator of pseudorandom sequence
        `mode`:
            * `asimov`: store input data without fluctuations
            * `normal`: normal distribution without correlations (2 inputs)
            * `normalstats`: normal distribution without correlations (1 input)
            * `poisson`: uses Poisson distribution
            * `covariance`: multivariate normal distribution using L-decomposition of the covariance matrix
    """

    __slots__ = (
        "_mode",
        "_generator",
    )

    _mode: ModeType
    _generator: Generator

    def __new__(
        cls,
        name,
        mode: ModeType,
        *args,
        generator: Generator | None = None,
        _baseclass: bool = True,
        **kwargs,
    ):
        if not _baseclass:
            return super().__new__(cls, *args, **kwargs)
        if mode in MonteCarloModes1:
            return MonteCarlo1(name, mode, *args, generator=generator, _baseclass=False, **kwargs)
        elif mode in MonteCarloModes2:
            return MonteCarlo2(name, mode, *args, generator=generator, _baseclass=False, **kwargs)

        raise RuntimeError(f"Invalid montecarlo mode {mode}. Expect: {MonteCarloModes}")

    def __init__(self, *args, generator: Generator | None = None, **kwargs):
        self._generator = self._create_generator() if generator is None else generator
        super().__init__(*args, auto_freeze=True, **kwargs)

    @property
    def mode(self) -> str:
        return self._mode

    def next_sample(self) -> None:
        self.unfreeze()
        self.taint(force_computation=True)

    @staticmethod
    def _create_generator() -> Generator:
        from numpy.random import MT19937
        algo = MT19937(seed=0)
        return Generator(algo)


class MonteCarlo1(MonteCarlo):
    r"""
    Generates a random sample distributed according different modes.

    inputs:
        `i`: average model vector

    outputs:
        `i`: generated sample

    extra arguments:
        `mode`:
            * `asimov`: store input data without fluctuations
            * `normalstats`: normal distribution without correlations (1 input)
            * `poisson`: uses Poisson distribution
    """

    __slots__ = ()

    def __init__(
        self,
        name,
        mode: ModeType1,
        generator: Generator | None = None,
        *args,
        _baseclass: bool = True,
        **kwargs,
    ):
        if mode not in MonteCarloModes1:
            raise RuntimeError(
                f"Invalid MonteCarlo mode {mode}. Expect: {MonteCarloModes1}"
            )

        self._mode = mode
        super().__init__(name, *args, generator=generator, **kwargs)
        # TODO: set lables

        self.labels.setdefaults(
            {
                "mark": f"MC:{mode[0].upper()}",
                #        "text": "MonteCarlo sample",
                #        "plottitle": "MonteCarlo sample",
                #        "latex": "MonteCarlo sample",
                #        "axis": "MonteCarlo sample",
            }
        )
        self._functions.update(
            {
                "asimov": self._fcn_asimov,
                "normalstats": self._fcn_normal_stats,
                "poisson": self._fcn_poisson,
            }
        )

    @staticmethod
    def _input_names() -> tuple[str, ...]:
        return ("data",)

    def _fcn_asimov(self) -> None:
        for _input, _output in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            _output[:] = _input[:]

    def _fcn_normal_stats(self) -> None:
        for _input, _output in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            _normal_stats(_input, _output, self._generator)

    def _fcn_poisson(self) -> None:
        for _input, _output in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            _poisson(_input, _output, self._generator)

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
        `i+1`: model uncertainties vector (sigma); *only for `normal` and `covariance`*

    outputs:
        `i`: generated sample

    extra arguments:
        `mode`:
            * `normal`: normal distribution without correlations (2 inputs)
            * `covariance`: multivariate normal distribution using L-decomposition of the covariance matrix
    """

    def __init__(
        self,
        name,
        mode: ModeType2,
        generator: Generator | None = None,
        *args,
        _baseclass: bool = True,
        **kwargs,
    ):
        if mode not in MonteCarloModes2:
            raise RuntimeError(
                f"Invalid montecarlo mode {mode}. Expect: {MonteCarloModes2}"
            )

        self._mode = mode
        super().__init__(name, *args, generator=generator, **kwargs)
        # TODO: set lables
        self.labels.setdefaults(
            {
                "mark": f"MC:{mode[0].upper()}",
                #        "text": "MonteCarlo sample",
                #        "plottitle": "MonteCarlo sample",
                #        "latex": "MonteCarlo sample",
                #        "axis": "MonteCarlo sample",
            }
        )
        if mode not in MonteCarloModes:
            raise InitializationError(
                f"mode must be in {MonteCarloModes}, but given {mode}",
                node=self,
            )
        self._functions.update(
            {
                "normal": self._fcn_normal,
                "covariance": self._fcn_covariance_L,
            }
        )

    @staticmethod
    def _input_names() -> tuple[str, ...]:
        return "data", "errors"

    def _fcn_covariance_L(self) -> None:
        i = 0
        while i < self.inputs.len_pos():
            _covariance_L(
                self.inputs[i].data,
                self.inputs[i + 1].data,
                self.outputs[i // 2].data,
                self._generator,
            )
            i += 2

    def _fcn_normal(self) -> None:
        i = 0
        while i < self.inputs.len_pos():
            _normal(
                self.inputs[i].data,
                self.inputs[i + 1].data,
                self.outputs[i // 2].data,
                self._generator,
            )
            i += 2

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_inputs_multiplicity(self, 2)
        n = self.inputs.len_pos()
        check_outputs_number(self, n // 2)

        if self.mode == "covariance":
            check_input_matrix_or_diag(self, slice(1, n, 2), check_square=True)

        for i in range(n // 2):
            check_inputs_multiplicable_mat(self, i, i + 1)
            copy_from_input_to_output(self, 2 * i, i)

        self.fcn = self._functions[self.mode]
