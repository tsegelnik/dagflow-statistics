from numpy import linspace
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
        self._add_input("mu", positional=False)
        self._add_input("sigma", positional=False)

    def _fcn(self):
        mu = self._mu.data[0]
        sigma = self._sigma.data[0]
        for inp, out in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            out[:] = norm(inp, loc=mu, scale=sigma)


@mark.parametrize("mu", (0, -2))
@mark.parametrize("sigma", (0.5, 1))
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
    # TODO: solve the problem with parameters properties!
    parmu, parsigma = pars.specs()
    parmu.vmin = mu - 1
    parmu.vmax = mu + 1
    parsigma.vmin = sigma / 2
    parsigma.vmax = sigma * 2
    parmu.step = 0.1
    parsigma.step = 0.1


    minimizer = IMinuitMinimizer(statistic=chi.outputs[0], minpars=pars, verbose=True)
    res = minimizer.fit()
    assert all(res["x"] == [mu, sigma])

    savegraph(graph, f"output/{testname}.png")
