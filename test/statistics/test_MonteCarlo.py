from os.path import join
from statistics.MonteCarlo import MonteCarlo

from matplotlib import pyplot as plt
from numpy import allclose, arange, diag, dot, eye, fabs, fill_diagonal, ones
from numpy.linalg import cholesky, inv
from numpy.random import seed
from pytest import mark

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array
from dagflow.plot import add_colorbar, closefig, plot_array_1d, plot_auto, savefig

seed(4)

@mark.parametrize("scale", [0.1, 100.0, 10000.0])
@mark.parametrize(
    "mcmode",
    [
        "Asimov",
        "Poisson",
        "NormalStats",
        "Normal",
        "Covariance",
    ],
)
@mark.parametrize("datanum", [0, 1, 2, "all"])
def test_mc(mcmode, scale, datanum, debug_graph, testname, tmp_path):
    size = 20
    data1 = ones(size, dtype="d") * scale
    data2 = (1.0 + arange(size, dtype="d")) * scale
    data3 = (size - arange(size, dtype="d")) * scale
    data = (data1, data2, data3)

    with Graph(close=True, debug=debug_graph) as graph:
        if datanum == "all":
            mcdata_v = tuple(
                MCTestData(data, mcmode, index=-i - 1, scale=scale)
                for i, data in enumerate(data)
            )
        else:
            mcdata_v = (
                MCTestData(data[datanum], mcmode, index=datanum + 1, scale=scale),
            )

        toymc = MonteCarlo(name="MonteCarlo", mode=mcmode)
        for mcdata in mcdata_v:
            mcdata.inputs >> toymc

    list(
        map(
            lambda o_out: MCTestData.set_mc(o_out[0], toymc, o_out[1]),
            zip(
                mcdata_v,
                list(toymc.outputs),
            ),
        )
    )

    tmp_path = join(str(tmp_path), testname)
    list(map(lambda o: MCTestData.plot(o, tmp_path), mcdata_v))

    list(map(MCTestData.check_nextSample, mcdata_v))
    list(map(MCTestData.check_stats, mcdata_v))

    plot_auto(toymc.outputs[0], save=f"output/{testname}_plot.png")
    savegraph(graph, f"output/{testname}.png")


class MCTestData:
    mcdata = None
    corrmat = None
    covmat_L = None
    figures = tuple()
    correlation = 0.95
    syst_unc_rel = 2
    nsigma = 5
    PoissonThreshold = {
        0.1: [1, 8, 8],
        100.0: [17, 17, 17],
        10000.0: [18, 18, 18],
    }

    def __init__(self, data, mctype, index, scale):
        self.index = str(index)
        self.scale = scale
        self.mctype = mctype
        self.data = data
        self.shape2 = (data.size, data.size)
        self.err_stat2 = data.copy()
        self.err_stat = self.err_stat2**0.5

        self.edges = arange(self.data.size + 1, dtype="d")
        edges = Array("edges", self.edges).outputs[0]
        self.hist = Array("hist", self.data, edges=[edges])

        if mctype == "Covariance":
            self.prepare_corrmatrix()
            self.prepare_covmatrix_syst()
            self.prepare_covmatrix_full()
        self.prepare_inputs()

    def prepare_inputs(self):
        if self.mctype in ("Normal", "CovarianceDiag"):
            self.input_err = Array("errors", self.err_stat)
            self.inputs = (self.hist, self.input_err)
        elif self.mctype == "Covariance":
            self.inputs = (self.hist, self.input_L)
        else:
            self.inputs = (self.hist,)

    def prepare_corrmatrix(self):
        self.corrmat = eye(self.data.size, dtype="d")
        for i in range(2, 5):
            for j in range(i + 1, 5):
                self.corrmat[j, i] = self.corrmat[i, j] = self.correlation
        fill_diagonal(self.corrmat, 1.0)

    def prepare_covmatrix_syst(self):
        self.err_syst = self.syst_unc_rel * self.data
        self.err_syst_sqr = diag(self.err_syst**0.5)
        self.covmat_syst = dot(
            dot(self.err_syst_sqr.T, self.corrmat), self.err_syst_sqr
        )

    def prepare_covmatrix_full(self):
        self.covmat_full = diag(self.err_stat2) + self.covmat_syst
        self.covmat_L = cholesky(self.covmat_full)
        self.covmat_L_inv = inv(self.covmat_L)
        self.input_L = Array("L", self.covmat_L)

    def set_mc(self, mcobject, mcoutput):
        self.mcobject = mcobject
        self.mcoutput = mcoutput
        self.mcdata = mcoutput.data
        self.mcdiff = self.mcdata - self.data

        if self.corrmat is None:
            self.mcdiff_norm = self.mcdiff / self.err_stat
        else:
            self.mcdiff_norm = self.covmat_L_inv @ self.mcdiff

    def plot(self, tmp_path):
        assert self.mcdata is not None
        self.tmp_path = tmp_path
        self.plot_hist()
        self.plot_mats()

    def figure(self, *args, **kwargs):
        fig = plt.figure(*args, **kwargs)
        self.figures += (fig,)
        return fig

    def savefig(self, *args):
        path = "_".join((self.tmp_path,) + tuple(args)) + ".png"
        savefig(path, dpi=300)

    def plot_hist(self):
        ax = self._create_fig("Check {index}, input {}, scale {scale}")
        plot_array_1d(
            self.hist.outputs[0].data,
            edges=self.hist.outputs[0].dd.axes_edges[0].data,
            color="black",
            label="input",
        )
        plot_array_1d(
            self.mcoutput.data,
            edges=self.edges,
            yerr=self.err_stat,
            linestyle="--",
            label="output",
        )
        ax.legend()
        self.savefig("hist", self.index)
        closefig()

        ax = self._create_fig("Check diff {index}, input {}, scale {scale}")
        plot_array_1d(
            self.mcdiff_norm,
            edges=self.edges,
            yerr=1.0,
            label="normalized uncorrelated",
        )
        ax.legend()
        self.savefig("diff_norm", self.index)

        ax.set_ylim(-4, 5)
        self.savefig("diff_norm_zoom", self.index)
        closefig()

        ax = self._create_fig("Check diff {index}, input {}, scale {scale}")
        plot_array_1d(
            self.mcdiff, edges=self.edges, yerr=self.err_stat, label="raw difference"
        )
        ax.legend()
        self.savefig("diff", self.index)
        closefig()

    def _create_fig(self, arg0):
        self.figure()
        result = plt.subplot(111)
        result.minorticks_on()
        result.grid()
        result.set_xlabel("")
        result.set_ylabel("")
        result.set_title(arg0.format(self.mctype, index=self.index, scale=self.scale))
        return result

    def matshow(self, mat, title, suffix):
        self.figure()
        ax = plt.subplot(111)
        c = plt.matshow(mat, fignum=False)
        add_colorbar(c)
        ax = plt.gca()
        ax.set_title(title)
        self.savefig(suffix, self.index)
        closefig()

    def plot_mats(self):
        if self.corrmat is None:
            return

        self.matshow(self.corrmat, "Correlation matrix", "corrmat")
        self.matshow(self.covmat_syst, "Covariance matrix (syst)", "covmat_syst")
        self.matshow(self.covmat_full, "Covariance matrix (full)", "covmat_full")
        self.matshow(self.covmat_L, "Covariance matrix decomposed: L", "covmat_L")

    def check_stats(self):
        if self.mctype == "Asimov":
            assert (self.mcdiff == 0.0).all()
        else:
            self._check_stats()
        if self.covmat_L is not None:
            cm_again = self.covmat_L @ self.covmat_L.T

            assert allclose(cm_again, self.covmat_full, atol=1.0e-9, rtol=0)

    def _check_stats(self):
        if self.mctype != "Poisson":
            assert (self.mcdiff != 0.0).all()

        mcdiff_abs = fabs(self.mcdiff_norm)
        assert (mcdiff_abs < self.nsigma).all()

        sum = self.mcdiff_norm.sum()
        sum_abs = fabs(sum)
        assert sum_abs < self.nsigma * self.data.size**0.5

        chi2 = (self.mcdiff_norm**2).sum()
        chi2_diff = chi2 - self.data.size
        assert chi2_diff < self.nsigma * (2.0 * self.data.size) ** 0.5

        diff_norm_abs = fabs(self.mcdiff_norm)
        n1 = (diff_norm_abs > 1).sum()
        n2 = (diff_norm_abs > 2).sum()
        n3 = (diff_norm_abs > 3).sum()
        assert n1 <= self.data.size * 0.6
        assert n2 <= self.data.size * 0.2 + 1
        assert n3 <= 1

    def check_nextSample(self):
        index = int(self.index)
        if index > 0:
            output_index = 0
            threshold_index = index - 1
        else:
            threshold_index = -index - 1
            output_index = -index - 1

        mcobject = self.mcobject
        self.first_data = mcobject.outputs[output_index].data.copy()
        self.mcobject.next_sample()
        self.second_data = mcobject.outputs[output_index].data.copy()
        self.mcdiff_nextSample = self.first_data - self.second_data

        if self.mctype == "Asimov":
            assert (self.mcdiff_nextSample == 0).all()
        elif self.mctype == "Poisson":
            scale = self.scale
            threshold = self.PoissonThreshold[scale][threshold_index]
            assert (self.mcdiff_nextSample != 0).sum() >= threshold
        else:
            assert (self.mcdiff_nextSample != 0).all()
