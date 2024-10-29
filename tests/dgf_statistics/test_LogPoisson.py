#!/usr/bin/env python
from pytest import mark
from numpy import allclose, arange, log, finfo
from scipy.special import gammaln

from dagflow.core.graph import Graph
from dagflow.plot.graphviz import savegraph
from dagflow.lib.common import Array

from dgf_statistics.LogPoisson import LogPoisson


@mark.parametrize("mode", ("poisson", "poisson_ratio"))
def test_LogPoisson_01(debug_graph, testname, mode):
    n = 10
    start = 10
    offset = 1.0
    dataArr = arange(start, start + n, dtype="d")
    theoryArr = dataArr + offset

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        data = Array("data", dataArr, mark="Data")
        theory = Array("theory", theoryArr, mark="Theory")
        logPoisson = LogPoisson(mode=mode)
        data >> logPoisson.inputs["data"]
        theory >> logPoisson.inputs["theory"]

    res = logPoisson.outputs["poisson"].data[0]
    if mode == "poisson_ratio":
        truth = 2.0 * ((theoryArr - dataArr) + dataArr * log(dataArr / theoryArr)).sum()
    else:
        truth = 2.0 * ((theoryArr - dataArr * log(theoryArr)).sum() + gammaln(dataArr + 1).sum())
    assert allclose(res, truth, atol=finfo("d").resolution)

    savegraph(graph, f"output/{testname}.png")
