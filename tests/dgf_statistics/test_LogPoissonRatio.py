#!/usr/bin/env python
from numpy import allclose, arange, finfo, log
from pytest import mark
from scipy.special import gammaln

from dagflow.core.graph import Graph
from dagflow.lib.common import Array
from dagflow.plot.graphviz import savegraph
from dgf_statistics.LogPoissonRatio import LogPoissonRatio


def test_LogPoissonRatio_01(debug_graph, testname):
    n = 10
    start = 10
    offset = 1.0
    dataArr = arange(start, start + n, dtype="d")
    theoryArr = dataArr + offset

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        data = Array("data", dataArr, mark="Data")
        theory = Array("theory", theoryArr, mark="Theory")
        logPoisson = LogPoissonRatio("lr")
        (data, theory) >> logPoisson

    res = logPoisson.outputs["result"].data[0]
    truth = 2.0 * ((theoryArr - dataArr) + dataArr * log(dataArr / theoryArr)).sum()
    assert allclose(res, truth, atol=0, rtol=0)

    savegraph(graph, f"output/{testname}.png")
