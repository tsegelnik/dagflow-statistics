from typing import TYPE_CHECKING

from dagflow.exception import InitializationError
from dagflow.inputhandler import MissingInputAdd
from dagflow.nodes import FunctionNode
from dagflow.parameters import Parameter

if TYPE_CHECKING:
    from dagflow.input import Input

from .iminuitminimizer import IMinuitMinimizer


class MinimizerNode(FunctionNode):
    __slots__ = ("_statistic", "_minimizer", "_params")

    _statistic: "Input"
    _params: list[Parameter]
    _minimizer: IMinuitMinimizer

    def __init__(self, name, *args, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAdd())
        super().__init__(name, *args, **kwargs)
        self.labels.setdefaults({"mark": "IMinuit Minimizer"})
        self._statistic = self._add_input("statistic", positional=False)  # input: 0

    def _fcn(self) -> None:
        self._params = [Parameter(parent=None, value_output=inp.parent_output) for inp in self.inputs]
        self._minimizer = IMinuitMinimizer(
            statistic=self._statistic.parent_output, parameters=self._params
        )
        return self._minimizer.fit()

    def profile_errors(
        self,
        confidence_level: float | None = None,
        ncall: int | None = None,
    ) -> dict:
        if self._minimizer is None:
            raise InitializationError("Evaluate the node before the errors profiling!", node=self)
        names = [param.output.node.name for param in self._params]
        return self._minimizer.profile_errors(names, confidence_level, ncall)

    def _typefunc(self) -> None:
        from dagflow.typefunctions import check_input_size, check_inputs_equivalence  # fmt: skip
        check_input_size(self, 0, exact=1)
        check_inputs_equivalence(self)
