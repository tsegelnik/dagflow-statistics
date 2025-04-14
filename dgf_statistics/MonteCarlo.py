from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Literal

from numba import njit
from numpy import add, matmul, sqrt

from dagflow.core.exception import InitializationError
from dagflow.core.type_functions import (
    check_inputs_are_matrices_or_diagonals,
    check_inputs_are_matrix_multipliable,
    check_inputs_number_is_divisible_by_N,
    check_number_of_outputs,
    copy_from_inputs_to_outputs,
)
from dagflow.lib.abstract import BlockToOneNode

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


def _random_with_covariance_L(
    mean: NDArray[double],
    cov_L: NDArray[double],
    result: NDArray[double],
    gen: Generator,
) -> None:
    if cov_L.ndim == 1:
        _random_with_covariance_L_1d(mean, cov_L, result, gen)
    else:
        _random_with_covariance_L_2d(mean, cov_L, result, gen)


@njit(cache=True)
def _random_with_covariance_L_1d(
    mean: NDArray[double],
    cov_L: NDArray[double],
    result: NDArray[double],
    gen: Generator,
) -> None:
    for i in range(len(result)):
        result[i] = mean[i] + cov_L[i] * gen.normal()


@njit(cache=True)
def _random_fill_normal(data: NDArray[double], gen: Generator) -> None:
    for i in range(len(data)):
        data[i] = gen.normal()


def _random_with_covariance_L_2d(
    mean: NDArray[double],
    cov_L: NDArray[double],
    result: NDArray[double],
    gen: Generator,
) -> None:
    _random_fill_normal(result, gen)
    matmul(cov_L, result, out=result)
    add(result, mean, out=result)


@njit(cache=True)
def _random_normal(
    mean: NDArray[double],
    errors: NDArray[double],
    result: NDArray[double],
    gen: Generator,
) -> None:
    for i in range(len(result)):
        result[i] = mean[i] + errors[i] * gen.normal()


@njit(cache=True)
def _random_normal_stats(
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
    r"""Generates a random sample distributed according different modes.

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

    _mode: Literal[
        "asimov", "normal", "normal-stats", "normal-unit", "poisson", "covariance"
    ]
    _generator: Generator

    def __new__(
        cls,
        name: str,
        mode: Literal[
            "asimov", "normal", "normal-stats", "normal-unit", "poisson", "covariance"
        ],
        *args,
        dtype: Literal["d", "f"] = "d",
        shape: tuple[int, ...] = (),
        generator: Generator | None = None,
        _baseclass: bool = True,
        **kwargs,
    ):
        if not _baseclass:
            return super().__new__(cls, *args)
        if mode in MonteCarloLocModes:
            return MonteCarloLoc(
                name,
                mode,
                *args,
                generator=generator,
                _baseclass=False,
                **kwargs,
            )
        elif mode in MonteCarloLocScaleModes:
            return MonteCarloLocScale(
                name,
                mode,
                *args,
                generator=generator,
                _baseclass=False,
                **kwargs,
            )
        elif mode in MonteCarloShapeModes:
            return MonteCarloShape(
                name,
                mode,
                *args,
                dtype=dtype,
                shape=shape,
                generator=generator,
                _baseclass=False,
                **kwargs,
            )

        raise RuntimeError(f"Invalid MonteCarlo mode {mode}. Expect: {MonteCarloModes}")

    def __init__(
        self,
        name: str,
        mode: Literal[
            "asimov", "normal", "normal-stats", "normal-unit", "poisson", "covariance"
        ],
        *args,
        dtype: Literal["d", "f"] = "d",
        shape: tuple[int, ...] = (),
        generator: Generator | None = None,
        **kwargs,
    ):
        self._generator = self._create_generator() if generator is None else generator
        super().__init__(name, *args, **kwargs)
        self._functions_dict.update({"asimov": self._function_asimov})

    @property
    def mode(self) -> str:
        return self._mode

    @abstractmethod
    def _function_asimov(self) -> None:
        raise NotImplementedError()

    def next_sample(self) -> None:
        self.unfreeze()
        self.touch(force_computation=True)
        # We need to set the flag frozen manually
        self.fd.frozen = True

    def reset(self) -> None:
        self._function_asimov()
        # We need to set the flag frozen manually
        self.fd.frozen = True

    @staticmethod
    def _create_generator() -> Generator:
        from numpy.random import MT19937, Generator

        algo = MT19937(seed=0)
        return Generator(algo)


class MonteCarloShape(MonteCarlo):
    r"""Generates a random sample distributed according normal distribution (0,
    1).

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
        generator: Generator | None = None,
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
        self._functions_dict.update(
            {
                "normal-unit": self._function_normal_unit,
            }
        )

    @staticmethod
    def _input_names() -> tuple:
        return ()

    def _function_asimov(self) -> None:
        for callback in self._input_nodes_callbacks:
            callback()

        for outdata in self._output_data:
            outdata[:] = 0.0
        # We need to set the flag frozen manually
        self.fd.frozen = True

    def _function_normal_unit(self) -> None:
        for callback in self._input_nodes_callbacks:
            callback()

        for outdata in self._output_data:
            _random_fill_normal(outdata, self._generator)

        # We need to set the flag frozen manually
        self.fd.frozen = True

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        self.function = self._functions_dict[self.mode]


class MonteCarloLoc(MonteCarlo):
    r"""Generates a random sample distributed according different modes.

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
        self._functions_dict.update(
            {
                "asimov": self._function_asimov,
                "normal-stats": self._function_normal_stats,
                "poisson": self._function_poisson,
            }
        )

    @staticmethod
    def _input_names() -> tuple[str, ...]:
        return ("data",)

    def _function_asimov(self) -> None:
        for callback in self._input_nodes_callbacks:
            callback()

        for indata, outdata in zip(self._input_data, self._output_data):
            outdata[:] = indata[:]

        # We need to set the flag frozen manually
        self.fd.frozen = True

    def _function_normal_stats(self) -> None:
        for callback in self._input_nodes_callbacks:
            callback()

        for indata, outdata in zip(self._input_data, self._output_data):
            _random_normal_stats(indata, outdata, self._generator)

        # We need to set the flag frozen manually
        self.fd.frozen = True

    def _function_poisson(self) -> None:
        for callback in self._input_nodes_callbacks:
            callback()

        for indata, outdata in zip(self._input_data, self._output_data):
            _poisson(indata, outdata, self._generator)

        # We need to set the flag frozen manually
        self.fd.frozen = True

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        n = self.inputs.len_pos()
        check_number_of_outputs(self, n)
        for i in range(n):
            copy_from_inputs_to_outputs(self, i, i)

        self.function = self._functions_dict[self.mode]


class MonteCarloLocScale(MonteCarlo):
    r"""Generates a random sample distributed according different modes.

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
        self._functions_dict.update(
            {
                "normal": self._function_normal,
                "covariance": self._function_covariance_L,
            }
        )

    @staticmethod
    def _input_names() -> tuple[str, ...]:
        return "data", "errors"

    def _function_asimov(self) -> None:
        for callback in self._input_nodes_callbacks:
            callback()

        for (indata, _), outdata in zip(self._blocks_input_data, self._output_data):
            outdata[:] = indata

        # We need to set the flag frozen manually
        self.fd.frozen = True

    def _function_covariance_L(self) -> None:
        for callback in self._input_nodes_callbacks:
            callback()

        for (indata, covariance_L), outdata in zip(
            self._blocks_input_data, self._output_data
        ):
            _random_with_covariance_L(
                indata,
                covariance_L,
                outdata,
                self._generator,
            )

        # We need to set the flag frozen manually
        self.fd.frozen = True

    def _function_normal(self) -> None:
        for callback in self._input_nodes_callbacks:
            callback()

        for (indata, std), outdata in zip(self._blocks_input_data, self._output_data):
            _random_normal(
                indata,
                std,
                outdata,
                self._generator,
            )
        # We need to set the flag frozen manually
        self.fd.frozen = True

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        check_inputs_number_is_divisible_by_N(self, 2)
        n = self.inputs.len_pos()
        check_number_of_outputs(self, n // 2)

        if self.mode == "covariance":
            check_inputs_are_matrices_or_diagonals(
                self, slice(1, n, 2), check_square=True
            )

        for i in range(n // 2):
            check_inputs_are_matrix_multipliable(self, i, i + 1)
            copy_from_inputs_to_outputs(self, 2 * i, i)

        self.function = self._functions_dict[self.mode]
