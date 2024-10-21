from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Literal

from numba import njit
from numpy import add, matmul, sqrt

from dagflow.exception import InitializationError
from dagflow.lib.BlockToOneNode import BlockToOneNode
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


MonteCarloLocModes = ("asimov", "normal-stats", "poisson")

MonteCarloLocScaleModes = ("normal", "covariance")

MonteCarloShapeModes = ("normal-unit",)

MonteCarloModes = (
    *MonteCarloLocModes,
    *MonteCarloLocScaleModes,
    *MonteCarloShapeModes,
)


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
            * `normal-stats`: normal distribution without correlations (1 input)
            * `normal-unit`: normal distribution without correlations (1 input, using for shape)
            * `poisson`: uses Poisson distribution
            * `covariance`: multivariate normal distribution using L-decomposition of the covariance matrix
    """

    __slots__ = (
        "_mode",
        "_generator",
    )

    _mode: Literal["asimov", "normal", "normal-stats", "normal-unit", "poisson", "covariance"]
    _generator: Generator

    def __new__(
        cls,
        name: str,
        mode: Literal["asimov", "normal", "normal-stats", "normal-unit", "poisson", "covariance"],
        *args,
        generator: Generator | None = None,
        _baseclass: bool = True,
        **kwargs,
    ):
        if not _baseclass:
            return super().__new__(cls)

        subclass = cls._determine_subclass(mode)
        return subclass(name, mode, *args, generator=generator, _baseclass=False, **kwargs)

    def __init__(
        self,
        name: str,
        mode: Literal["asimov", "normal", "normal-stats", "normal-unit", "poisson", "covariance"],
        *args,
        generator: Generator = None,
        **kwargs
    ):
        self._generator = self._create_generator() if generator is None else generator
        super().__init__(name, *args, auto_freeze=True, **kwargs)
        self._functions.update(
            {
                "asimov": self._fcn_asimov,
            }
        )

    @property
    def mode(self) -> str:
        return self._mode

    @abstractmethod
    def _fcn_asimov(self) -> None:
        raise NotImplementedError()

    def next_sample(self) -> None:
        self.unfreeze()
        self.touch(force_computation=True)

    def reset(self) -> None:
        self._fcn_asimov()

    @staticmethod
    def _determine_subclass(mode):
        if mode in MonteCarloLocModes:
            return MonteCarloLoc
        elif mode in MonteCarloLocScaleModes:
            return MonteCarloLocScale
        elif mode in MonteCarloShapeModes:
            return MonteCarloShape

        raise RuntimeError(f"Invalid MonteCarlo mode {mode}. Expect: {MonteCarloModes}")

    @staticmethod
    def _create_generator() -> Generator:
        from numpy.random import MT19937, Generator
        algo = MT19937(seed=0)
        return Generator(algo)


class MonteCarloShape(MonteCarlo):
    r"""
    Generates a random sample distributed according normal distribution (0, 1).

    inputs:
        `i`: average model vector

    outputs:
        `i`: generated sample

    extra arguments:
        `mode`:
            * `normal-unit`: normal distribution without correlations (1 input, using for shape)
    """

    def __init__(
        self,
        name: str,
        mode: Literal["normal-unit"],
        *args,
        dtype: Literal["d", "f"] = "d",
        shape: tuple[int, ...] = (),
        generator: Generator = None,
        _baseclass: bool = True,
        **kwargs,
    ):
        if mode not in MonteCarloShapeModes:
            raise RuntimeError(
                f"Invalid MonteCarlo mode {mode}. Expect: {MonteCarloShapeModes}"
            )

        self._mode = mode
        super().__init__(name, mode, *args, generator=generator, **kwargs)
        self._add_output("result", shape=shape, dtype=dtype)
        # TODO: set labels

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
                "normal-unit": self._fcn_normal_unit,
            }
        )

    @staticmethod
    def _input_names() -> tuple:
        return ()

    def _fcn_asimov(self) -> None:
        for _output in self.outputs.iter_data():
            _output[:] = 0.

    def _fcn_normal_unit(self) -> None:
        for _output in self.outputs.iter_data():
            _fill_normal(_output, self._generator)

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        self.fcn = self._functions[self.mode]


class MonteCarloLoc(MonteCarlo):
    r"""
    Generates a random sample distributed according different modes.

    inputs:
        `i`: average model vector

    outputs:
        `i`: generated sample

    extra arguments:
        `mode`:
            * `asimov`: store input data without fluctuations
            * `normal-stats`: normal distribution without correlations (1 input)
            * `poisson`: uses Poisson distribution
    """

    def __init__(
        self,
        name: str,
        mode: Literal["asimov", "normal-stats", "poisson"],
        *args,
        generator: Generator | None = None,
        _baseclass: bool = True,
        **kwargs,
    ):
        if mode not in MonteCarloLocModes:
            raise RuntimeError(
                f"Invalid MonteCarlo mode {mode}. Expect: {MonteCarloLocModes}"
            )

        self._mode = mode
        super().__init__(name, mode, *args, generator=generator, **kwargs)
        # TODO: set labels

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
                "normal-stats": self._fcn_normal_stats,
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


class MonteCarloLocScale(MonteCarlo):
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
        name: str,
        mode: Literal["normal", "covariance"],
        *args,
        generator: Generator | None = None,
        _baseclass: bool = True,
        **kwargs,
    ):
        if mode not in MonteCarloLocScaleModes:
            raise RuntimeError(
                f"Invalid MonteCarlo mode {mode}. Expect: {MonteCarloLocScaleModes}"
            )

        self._mode = mode
        super().__init__(name, mode, *args, generator=generator, **kwargs)
        # TODO: set labels

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

    def _fcn_asimov(self) -> None:
        i = 0
        while i < self.inputs.len_pos():
            self.outputs[i // 2].data[:] = self.inputs[i].data[:]
            i += 2

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
