from math import sqrt

from matplotlib import pyplot as plt  # fmt:skip
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
_Background = 100
_verbose = False


class Model(OneToOneNode):
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


class Shift(OneToOneNode):
    """The node to shift the data by Y axis to avoid negative bins in the MC data"""

    __slots__ = "_shift"
    _shift: float

    def __init__(self, *args, shift: float = _Background, **kwargs):
        super().__init__(*args, **kwargs)
        self._shift = shift

    def _fcn(self):
        for inp, out in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            out[:] = self._shift + inp[:]


@mark.parametrize("mu", (-1.531654, 2.097123))
@mark.parametrize("sigma", (0.567543, 1.503321))
@mark.parametrize("mode", ("asimov", "normalstats"))
def test_IMinuitMinimizer(mu, sigma, mode, testname):
    size = 201
    x = linspace(-10, 10, size)

    # start values of the fitting
    mufit = mu / 2
    sigmafit = sigma * 1.5
    with Graph(close=True) as graph:
        # setting of true parameters
        Mu0 = Array("mu 0", [mu])
        Sigma0 = Array("sigma 0", [sigma])
        X = Array("x", x)

        # build input data for the MC simulation
        pdf0 = Model("normal pdf for MC")
        X >> pdf0
        Mu0 >> pdf0("mu")
        Sigma0 >> pdf0("sigma")
        model = pdf0.outputs[0]

        # perform fluctuations of data within MC and shift the result with constant background
        mc = MonteCarlo("MC", mode=mode)
        model >> mc
        shiftMC = Shift("exp")
        mc >> shiftMC

        # build a model to fit exp data
        MuFit = Array("mu fit", [mufit])
        SigmaFit = Array("sigma fit", [sigmafit])
        pdffit = Model("normal pdf for the Model")
        X >> pdffit
        MuFit >> pdffit("mu")
        SigmaFit >> pdffit("sigma")
        shiftFit = Shift("Model")
        pdffit >> shiftFit
        modelfit = shiftFit.outputs[0]

        # eval errors
        cnp = CNPStat("CNP stat")
        (shiftMC, modelfit) >> cnp

        # eval Chi2
        chi = Chi2("Chi2")
        shiftMC >> chi("data")
        modelfit >> chi("theory")
        cnp.outputs[0] >> chi("errors")

    # check if the MC data is valid: negative events -> wrong model
    assert min(shiftMC.outputs[0].data) > 0

    # perform a minimization
    parmu = Parameter(parent=None, value_output=MuFit.outputs[0])
    parsigma = Parameter(parent=None, value_output=SigmaFit.outputs[0])
    minimizer = IMinuitMinimizer(statistic=chi.outputs[0], parameters=[parmu, parsigma], verbose=_verbose)
    res = minimizer.fit()

    assert res["success"]
    assert res["nfev"] > 1

    names = res["names"]
    assert (
        len(res["x"])
        == len(names)
        == len(res["errorsdict"])
        == len(res["errors"])
        == len(res["xdict"])
        == res["npars"]
        == 2
    )

    atol = 2.0 / sqrt(_NevScale)
    assert allclose(res["x"], [mu, sigma], rtol=0, atol=atol if mode == "normalstats" else 2e-5)
    assert allclose(res["covariance"], minimizer.calculate_covariance(), rtol=0, atol=1e-8)
    assert all(res["errorsdict"][key] == res["errors"][i] for i, key in enumerate(names))

    # errors checks
    errors = minimizer.profile_errors()
    assert errors["names"] == names
    errs = errors["errors"]
    assert all(abs(err) < atol for errParam in errs for err in errParam)
    assert all(errors["errorsdict"][key] == errs[i] for i, key in enumerate(names))
    for name in names:
        for key in ("is_valid", "lower_valid", "upper_valid"):
            assert errors["errors_profile_status"][name][key]
        assert errors["errors_profile_status"][name]["message"] == ""

    # save plot and graph
    draw_params(res["x"], mu, sigma, minimizer, f"output/{testname}-params.png")
    draw_fit(x, shiftMC, model, modelfit, mode, f"output/{testname}-plot.png")
    savegraph(graph, f"output/{testname}.png")


def draw_params(res, mu, sigma, minimizer, figname):
    _, ax = plt.subplots()
    for cl in (1, 2, 3):
        contours = minimizer._minimizer.mncontour(0, 1, cl=cl)
        ax.plot(contours[:, 0], contours[:, 1], label=f"{cl}σ")
    ax.scatter(*[mu, sigma], label="true")
    ax.scatter(*res, label="fit")
    plt.xlabel("mu")
    plt.ylabel("sigma")
    plt.legend()
    plt.xlim([mu - 0.01, mu + 0.01])
    plt.ylim([sigma - 0.01, sigma + 0.01])
    plt.savefig(figname)
    plt.close()


def draw_fit(x, mc, model, modelfit, mode, figname):
    ax = plt.subplot(111)
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plot_array_1d(
        mc.outputs[0].data,
        meshes=x,
        color="black",
        label="data+fluct." if mode != "asimov" else "asimov MC",
    )
    plot_array_1d(model.data+_Background, meshes=x, linestyle="--", label="data")
    plot_array_1d(modelfit.data, meshes=x, linestyle="--", label="fit")
    ax.legend()
    plt.savefig(figname)
    plt.close()
