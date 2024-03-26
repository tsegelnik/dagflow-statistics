from math import sqrt

from numpy import allclose, linspace
from pytest import mark
from scipy.stats import norm

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.input import Input
from dagflow.lib import Array, OneToOneNode
from dagflow.parameters import Parameter
from dagflow.plot import plot_array_1d
from dgf_statistics.Chi2 import Chi2
from dgf_statistics.CNPStat import CNPStat
from dgf_statistics.minimizer.iminuitminimizer import IMinuitMinimizer
from dgf_statistics.MonteCarlo import MonteCarlo

_NevScale = 10000


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
            out[:] = _NevScale * norm.pdf(inp[:], loc=mu, scale=sigma)


@mark.parametrize("mu", (-1.531, 2.097))
@mark.parametrize("sigma", (0.567, 1.503))
def test_IMinuitMinimizer_normal(mu, sigma, testname):
    size = 1001
    x = linspace(-8, 8, size)

    # start values of the fitting
    mufit = mu / 2
    sigmafit = sigma * 1.5
    with Graph(close=True) as graph:
        # setting up of true parameters
        Mu0 = Array("mu0", [mu])
        Sigma0 = Array("sigma0", [sigma])
        X = Array("x", x)

        # build model data to do MC
        pdf0 = NormalPDF("model")
        X >> pdf0
        Mu0 >> pdf0("mu")
        Sigma0 >> pdf0("sigma")
        model = pdf0.outputs[0]

        # do MC modelling
        mc = MonteCarlo("MC", mode="normalstats")
        model >> mc

        # build new model to fit MC simulations
        MuFit = Array("mufit", [mufit])
        SigmaFit = Array("sigmafit", [sigmafit])
        pdffit = NormalPDF("modelfit")
        X >> pdffit
        MuFit >> pdffit("mu")
        SigmaFit >> pdffit("sigma")
        modelfit = pdffit.outputs[0]

        # eval errors
        cnp = CNPStat("CNP stat")
        (mc, modelfit) >> cnp

        # eval Chi2
        chi = Chi2("Chi2")
        mc >> chi("data")
        modelfit >> chi("theory")
        cnp.outputs[0] >> chi("errors")

    parmu = Parameter(parent=None, value_output=MuFit.outputs[0])
    parsigma = Parameter(parent=None, value_output=SigmaFit.outputs[0])
    minimizer = IMinuitMinimizer(statistic=chi.outputs[0], parameters=[parmu, parsigma], verbose=False)
    res = minimizer.fit()

    atol = 5.0 / sqrt(_NevScale)
    assert allclose(res["x"], [mu, sigma], rtol=0, atol=atol)
    assert allclose(res["covariance"], minimizer.calculate_covariance(), rtol=0, atol=1e-10)

    # errors checks
    errors = minimizer.profile_errors()
    names = res["names"]
    assert errors["names"] == names
    assert all(abs(err) < atol for errs in errors["errors"] for err in errs)
    assert all(errors["errorsdict"][key] == errors["errors"][i] for i, key in enumerate(names))
    for name in names:
        for key in ("is_valid", "lower_valid", "upper_valid"):
            assert errors["errors_profile_status"][name][key]
        assert errors["errors_profile_status"][name]["message"] == ""

    # save plot and graph
    draw(x, mc, model, modelfit, f"output/{testname}-plot.png")
    savegraph(graph, f"output/{testname}.png")


def draw(x, mc, model, modelfit, figname):
    from matplotlib import pyplot as plt  # fmt:skip
    def _create_fig():
        result = plt.subplot(111)
        result.minorticks_on()
        result.grid()
        result.set_xlabel("x")
        result.set_ylabel("y")
        return result

    ax = _create_fig()
    plot_array_1d(
        mc.outputs[0].data,
        meshes=x,
        color="black",
        label="MC",
    )
    plot_array_1d(
        model.data,
        meshes=x,
        linestyle="--",
        label="model",
    )
    plot_array_1d(
        modelfit.data,
        meshes=x,
        linestyle="--",
        label="fit",
    )
    ax.legend()
    plt.savefig(figname)
    plt.close()
