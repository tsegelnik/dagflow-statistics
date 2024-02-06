from dgf_statistics.LogPoissonConst import LogPoissonConst
from dgf_statistics.LogPoissonConst import ModeType
from dgf_statistics.LogPoissonMain import LogPoissonMain
from collections.abc import Mapping
from typing import TYPE_CHECKING

from dagflow.metanode import MetaNode

if TYPE_CHECKING:
    from dagflow.node import Node


class LogPoisson(MetaNode):
    __slots__ = ("_LogPoissonConst", "_LogPoissonMain")

    _LogPoissonConst: "Node"
    _LogPoissonMain: "Node"

    def __init__(
        self,
        mode: ModeType = "poisson",
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
            kw_inputs=["theory", "data"],
            kw_outputs=["poisson"],
            merge_inputs=["data"],
            missing_inputs=True,
            also_missing_outputs=True,
        )
        return logPoisson


# TODO: fix this method
#    @classmethod
#    def replicate(
#        cls,
#        mode: ModeType = "poisson",
#        name_LogPoissonConst: str = "LogPoissonConst",
#        name_LogPoissonMain: str = "LogPoisson",
#        labels: Mapping = {},
#        *,
#        replicate: Tuple[KeyLike, ...] = ((),),
#    ) -> Tuple["LogPoisson", "NodeStorage"]:
#        storage = NodeStorage(default_containers=True)
#        nodes = storage("nodes")
#        inputs = storage("inputs")
#        outputs = storage("outputs")
#
#        logPoissons = cls(mode=mode, bare=True)
#        key_LogPoissonMain = (name_LogPoissonMain,)
#        key_LogPoissonConst = (name_LogPoissonConst,)
#
#        logPoissons._init_LogPoissonConst(mode, name_LogPoissonConst, labels.get("LogPoissonConst", {}))
#        outputs[key_LogPoissonConst + ("const",)] = logPoissons._LogPoissonConst.outputs["const"]
#
#        label_int = labels.get("LogPoisson", {})
#        for key in replicate:
#            if isinstance(key, str):
#                key = (key,)
#            name = ".".join(key_LogPoissonMain + key)
#            logPoissonMain = logPoissons._add_LogPoissonMain(name, label_int, positionals=False)
#            logPoissonMain()
#            nodes[key_LogPoissonMain + key] = logPoissonMain
#            inputs[key_LogPoissonMain + key] = logPoissonMain.inputs[0]
#            outputs[key_LogPoissonMain + key] = logPoissonMain.outputs[0]
#
#        NodeStorage.update_current(storage, strict=True)
#        return logPoissons, storage
