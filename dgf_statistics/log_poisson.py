from __future__ import annotations

from typing import TYPE_CHECKING

from dagflow.core.meta_node import MetaNode

from dgf_statistics.log_poisson_const import LogPoissonConst
from dgf_statistics.log_poisson_main import LogPoissonMain

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dagflow.core.node import Node

    from dgf_statistics.LogPoissonConst import ModeType


class LogPoisson(MetaNode):
    __slots__ = ("_LogPoissonConst", "_LogPoissonMain")

    _LogPoissonConst: Node
    _LogPoissonMain: Node

    def __init__(
        self,
        mode: ModeType = "poisson_ratio",
        *,
        bare: bool = False,
        labels: Mapping = {},
    ):
        super().__init__()
        if bare:
            return

        self._init_LogPoissonConst(
            mode=mode,
            name="LogPoissonConst",
            label=labels.get("LogPoissonConst", {}),
        )
        self._init_LogPoissonMain("LogPoisson", labels.get("LogPoisson", {}))

    def _init_LogPoissonConst(
        self,
        mode: ModeType,
        name: str = "LogPoissonConst",
        label: Mapping = {},
    ) -> None:
        self._LogPoissonConst = LogPoissonConst(name=name, mode=mode, label=label)

        self._add_node(
            self._LogPoissonConst,
            kw_inputs=["data"],
            kw_outputs=["const"],
            merge_inputs=["data"],
        )

    def _init_LogPoissonMain(
        self,
        name: str = "LogPoisson",
        label: Mapping = {},
    ) -> LogPoissonMain:
        logPoisson = LogPoissonMain(name, label=label)
        self._LogPoissonConst._const >> logPoisson("const")

        self._add_node(
            logPoisson,
            kw_inputs=["data", "theory"],
            kw_outputs=["poisson"],
            merge_inputs=["data"],
            missing_inputs=True,
            also_missing_outputs=True,
        )
        return logPoisson

