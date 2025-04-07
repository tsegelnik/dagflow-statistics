#!/usr/bin/env python
from numpy import allclose, arange, array, diag, finfo, matmul, sqrt
from numpy.linalg import cholesky, inv
from pytest import mark

from dagflow.core.graph import Graph
from dagflow.lib.common import Array
from dagflow.plot.graphviz import savegraph
from dgf_statistics.Chi2 import Chi2
from dgf_statistics.CNPStat import CNPStat


def test_Chi2_01(debug_graph, testname):
    n = 10
    start = 10
    offset = 1.0
    dataArr = arange(start, start + n, dtype="d")
    theoryArr = dataArr + offset
    statArr = dataArr**0.5

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        data = Array("data", dataArr, mark="Data", mode="fill")
        theory = Array("theory", theoryArr, mark="Theory", mode="fill")
        stat = Array("staterr", statArr, mark="Stat errors", mode="fill")
        chi2 = Chi2("chi2")
        (data, theory, stat) >> chi2

    res = chi2.outputs["result"].data[0]
    truth1 = (((dataArr - theoryArr) / statArr) ** 2).sum()
    truth2 = ((offset / statArr) ** 2).sum()
    assert (res == truth1).all()
    assert (res == truth2).all()

    savegraph(graph, f"output/{testname}.png")


def test_Chi2_02(debug_graph, testname):
    n = 15
    start = 10
    offset = 1.0
    dataArr = arange(start, start + n, dtype="d")
    theoryArr = dataArr + offset
    covmat = diag(dataArr)
    Lmat = cholesky(covmat)

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        data = Array("data", dataArr, mark="Data", mode="fill")
        theory = Array("theory", theoryArr, mark="Theory", mode="fill")
        L = Array("L", Lmat, mark="Stat errors (cholesky)", mode="fill")
        chi2 = Chi2("chi2")
        data >> chi2
        theory >> chi2
        L >> chi2

    res = chi2.outputs["result"].data[0]
    truth = (offset**2 / dataArr).sum()
    assert allclose(res, truth, rtol=0, atol=finfo("d").resolution)

    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("duplicate", (False, True))
def test_Chi2_03(duplicate: bool, debug_graph, testname):
    n = 10
    start = 10
    offset = 1.0
    dataArr = arange(start, start + n, dtype="d")
    theoryArr = dataArr + offset
    covmat = diag(dataArr) + 2.0
    Lmat = cholesky(covmat)

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        data = Array("data", dataArr, mark="Data", mode="fill")
        theory = Array("theory", theoryArr, mark="Theory", mode="fill")
        L = Array("L", Lmat, mark="Stat errors (cholesky)", mode="fill")
        chi2 = Chi2("chi2")
        (data, theory, L) >> chi2
        if duplicate:
            (data, theory, L) >> chi2
    res = chi2.outputs["result"].data[0]

    scale = duplicate and 2.0 or 1.0
    diff = array(dataArr - theoryArr).T
    truth1 = scale * matmul(diff.T, matmul(inv(covmat), diff))
    ndiff = matmul(inv(Lmat), diff)
    truth2 = scale * matmul(ndiff.T, ndiff)

    assert allclose(res, truth1, rtol=0, atol=finfo("d").resolution)
    assert allclose(res, truth2, rtol=0, atol=finfo("d").resolution)

    savegraph(graph, f"output/{testname}.png")


def test_Chi2CNPStat_v01(debug_graph, testname):
    n = 10
    start = 10
    offset = 1.0
    dataa = arange(start, start + n, dtype="d") + 1
    theorya = dataa + offset

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        data = Array("data", dataa, mark="Data", mode="fill")
        theory = Array("theory", theorya, mark="Theory", mode="fill")

        cnp = CNPStat(name="cnp", label="CNP stat. uncertainty")

        (data, theory) >> cnp
        chi2 = Chi2(name="chi2", label="chi2")
        data >> chi2("data")
        theory >> chi2("theory")
        cnp.outputs[0] >> chi2("errors")
    res = chi2.outputs["result"].data[0]
    res_expected = ((1.0 / dataa + 2.0 / theorya) * (theorya - dataa) ** 2).sum() / 3.0
    assert allclose(res, res_expected, atol=finfo("d").resolution)

    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("mode", (None, "uncertainty", "variance"))
def test_Chi2CNPStat_v01(mode: str | None, debug_graph, testname):
    n = 10
    start = 10
    offset = 1.0
    dataa = arange(start, start + n, dtype="d") + 1
    theorya = dataa + offset

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        data = Array("data", dataa, mark="Data", mode="fill")
        theory = Array("theory", theorya, mark="Theory", mode="fill")

        if mode:
            cnp = CNPStat(name="cnp", mode=mode, label=f"CNP stat. {mode}")
        else:
            cnp = CNPStat(name="cnp", label=f"CNP stat. {mode}")
        (data, theory) >> cnp

    res = cnp.outputs["result"].data
    if mode == "uncertainty" or mode is None:
        res_expected = sqrt(3.0 / (1 / dataa + 2 / theorya))
    elif mode == "variance":
        res_expected = 3.0 / (1 / dataa + 2 / theorya)

    assert allclose(res, res_expected, atol=finfo("d").resolution)

    savegraph(graph, f"output/{testname}.png")
