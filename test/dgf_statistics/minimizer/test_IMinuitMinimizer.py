from numpy import allclose, linspace
from pytest import mark
from scipy.stats import norm

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.input import Input
from dagflow.lib import Array

from dagflow.parameters import Parameter
from dgf_statistics.MonteCarlo import MonteCarlo
from dgf_statistics.CNPStat import CNPStat
from dgf_statistics.Chi2 import Chi2
from dgf_statistics.minimizer.iminuitminimizer import IMinuitMinimizer
from dgf_statistics.minimizer.minpars import MinPars

from dagflow.lib import OneToOneNode


class NormalPDF(OneToOneNode):
    __slots__ = ("_mu", "_sigma")
    _mu: Input
    _sigma: Input

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mu = self._add_input("mu", positional=False)
        self._sigma = self._add_input("sigma", positional=False)

    def _fcn(self):
        mu = self._mu.data[0]
        sigma = self._sigma.data[0]
        for inp, out in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            out[:] = norm.pdf(inp[:], loc=mu, scale=sigma)


@mark.parametrize("mu", (10.789123, -5.321123))
@mark.parametrize("sigma", (0.543976, 1.967091))
def test_IMinuitMinimizer_normal(mu, sigma, testname):
    size = 11

    x = linspace(-5, 5, size)
    with Graph(close=True) as graph:
        Mu = Array("mu", [mu])
        Sigma = Array("sigma", [sigma])
        X = Array("x", x)

        pdf = NormalPDF("model")
        X >> pdf
        Mu >> pdf("mu")
        Sigma >> pdf("sigma")
        model = pdf.outputs[0]

        mc = MonteCarlo("MC", mode="normalstats")
        model >> mc

        cnp = CNPStat("CNP stat")
        (model, mc) >> cnp

        chi = Chi2("Chi2")
        mc >> chi("data")
        model >> chi("theory")
        cnp.outputs[0] >> chi("errors")

    pars = MinPars(
        {
            "mu": Parameter(parent=None, value_output=Mu.outputs[0]),
            "sigma": Parameter(parent=None, value_output=Sigma.outputs[0]),
        }
    )
    parmu, parsigma = pars.specs()
    parmu.step = 0.1
    parsigma.step = 0.1

    startvalues = [mu / 2, sigma *2]
    minimizer = IMinuitMinimizer(
        statistic=chi.outputs[0], minpars=pars, verbose=False, startvalues=startvalues
    )
    res = minimizer.fit()
    assert allclose(res["x"], [mu, sigma], rtol=0, atol=1e-15)

    savegraph(graph, f"output/{testname}.png")
