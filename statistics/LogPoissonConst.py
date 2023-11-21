from math import lgamma, log
from typing import Literal, TYPE_CHECKING

from numba import float64, njit, void
from numpy import double
from numpy.typing import NDArray

from dagflow.exception import InitializationError
from dagflow.input_extra import MissingInputAddOne
from dagflow.nodes import FunctionNode

if TYPE_CHECKING:
    from dagflow.input import Input
    from dagflow.output import Output


LogPoissonModes = {"poisson", "poisson_ratio"}
ModeType = Literal[LogPoissonModes]


@njit(void(float64[:], float64[:]), cache=True)
def _const_poisson_ratio(data: NDArray[double], const: NDArray[double]):
    r"""$\sum \log(theory_i) \approx \sum (theory_i * \log(theory_i) - theory_i)$"""
    func = lambda x: x * log(x) - x if x not in {0.0, 1.0} else 0
    for i in range(len(data)):
        const[0] += func(data[i])


@njit(void(float64[:], float64[:]), cache=True)
def _const_poisson(data: NDArray[double], const: NDArray[double]):
    r"""$\sum \log(theory_i) = \sum \log(\Gamma(theory_i + 1))$"""
    for i in range(len(data)):
        const[0] += lgamma(data[i] + 1)


class LogPoissonConst(FunctionNode):
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

    _data: "Input"
    _const: "Output"
    _mode: ModeType

    def __init__(self, name, mode: ModeType = "poisson_ratio", *args, **kwargs):
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
        if mode not in LogPoissonModes:
            raise InitializationError(
                f"mode must be in {LogPoissonModes}, but given {mode}",
                node=self,
            )
        self._mode = mode
        self._data = self._add_input("data")  # input: 0
        self._const = self._add_output("const")  # output: 0
        self._functions.update({"poisson_ratio": self._fcn_poisson_ratio, "poisson": self._fcn_poisson})

    @property
    def mode(self) -> ModeType:
        return self._mode

    def _fcn_poisson_ratio(self):
        self._const.data[0] = 0.0
        for _input in self.inputs.iter_data():
            _const_poisson_ratio(_input.ravel(), self._const.data)

    def _fcn_poisson(self):
        self._const.data[0] = 0.0
        for _input in self.inputs.iter_data():
            _const_poisson(_input.ravel(), self._const.data)

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        self._const.dd.shape = (1,)
        self._const.dd.dtype = self._data.dd.dtype
        self.fcn = self._functions[self.mode]
