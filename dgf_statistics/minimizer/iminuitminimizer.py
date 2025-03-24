from __future__ import annotations

from typing import TYPE_CHECKING

from iminuit import Minuit
from numpy import array, ascontiguousarray

from .fitresult import FitResult
from .minimizerbase import MinimizerBase

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from dagflow.core.output import Output
    from dagflow.parameters import Parameter

# if we cannot import runtime_error from root we use DagflowError to avoid any exception capture,
# i.e., if CppRuntimeError==DagflowError, the exception will be not raised
try:
    import ROOT  # fmt: skip
    CppRuntimeError = ROOT.std.runtime_error
except Exception:
    from dagflow.core.exception import DagflowError  # fmt:skip
    CppRuntimeError = DagflowError


class IMinuitMinimizer(MinimizerBase):
    __slots__ = ("_errordef",)
    _errordef: float

    def __init__(
        self,
        statistic: Output,
        parameters: dict[str, Parameter],
        name: str = "iminuit",
        label: str = "iminuit",
        errordef: float = 1.0,  # 1.0: LEAST_SQUARES, 0.5: LIKELIHOOD
        *,
        limits: dict[str, tuple[float | None, float | None]] = {},
        **kwargs,
    ) -> None:
        super().__init__(statistic, parameters, name, label, limits=limits, **kwargs)
        self._errordef = errordef

    def _child_fit(self, **kwargs) -> dict:
        """
        Run Migrad minimization.

        Migrad from the Minuit2 library is a robust minimization algorithm,
        which uses first and approximate second derivatives
        to achieve quadratic convergence near the minimum.
        """
        ncall = kwargs.pop("ncall", None)  #  maximum number of calls inside migrad
        iterate = kwargs.pop(
            "iterate", 5
        )  # N calls if convergence was not reached; default: 5

        result = self.init_minimizer()
        fmin = None
        with FitResult() as fr:
            try:
                result = self._minimizer.migrad(ncall=ncall, iterate=iterate)
            except CppRuntimeError as exc:
                message = f"{exc.what()}"  # pyright: ignore
            except RuntimeError as exc:
                message = repr(exc)
            else:
                # the message by default has str, html and pretty representations,
                # but we build a dict using slots with complete information
                fmin = result.fmin
                message = {key[1:]: getattr(fmin, key) for key in fmin.__slots__}
        if fmin:
            fr.set(
                x=array(result.values),
                errors=array(result.errors),
                fun=float(result.fval) if result.fval is not None else None,
                success=result.valid,
                summary=message,
                minimizer=self._label,
                nfev=result.nfcn,
                errorsdef=self._errordef,
                covariance=(
                    array(result.covariance) if result.covariance is not None else None
                ),
            )
            self._result = fr.result

        self.patchresult()
        return self.result

    def init_minimizer(self) -> Minuit:
        """
        Initializes the Minuit minimizer
        """
        if not self._parameters:
            raise RuntimeError("Pass parameters to minimize!")
        minimizable = self.init_minimizable()

        def fcn(*params):
            params = ascontiguousarray(params, dtype="d")
            return minimizable(params)

        startvalues = []
        names = self._parameters_names
        for par in self._parameters:
            # TODO: wrap path getter into Parameter method
            startvalues.append(par.value)

        self._minimizer = minimizer = Minuit(fcn, *startvalues, name=names)
        minimizer.throw_nan = True
        minimizer.errordef = self._errordef

        limits = []
        for name in names:
            if name in self._limits.keys():
                limits.append(self._limits[name])
            else:
                limits.append((None, None))
        self._minimizer.limits = limits

        return minimizer

    def profile_errors(
        self,
        names: list[str] | None = None,
        confidence_level: float | None = None,
        ncall: int | None = None,
    ) -> dict:
        """
        Calculates errors for parameters within the Minos algorithm.

        The Minos algorithm uses the profile likelihood method to compute (generally asymmetric)
        confidence intervals. It scans the negative log-likelihood or (equivalently)
        the least-squares cost function around the minimum to construct a confidence interval.
        """
        result = {}
        if names:
            self.logger.info(f"Caclulating profile for: {names}")
            _names = result["names"] = names
        else:
            _names = result["names"] = self.result["names"]
        errs = result["errors"] = []
        errsdict = result["errorsdict"] = {}
        statuses = result["errors_profile_status"] = {}

        for name in _names:
            status = statuses[name] = {}
            try:
                res = self._minimizer.minos(name, cl=confidence_level, ncall=ncall)
                stat = res.merrors[name]
                errs.append([stat.lower, stat.upper])
                errsdict[name] = errs[-1]
                for key in stat.__slots__:
                    status[key] = getattr(stat, key)
                status["message"] = ""
            except CppRuntimeError as exc:
                status["is_valid"] = False
                status["message"] = f"{exc.what()}"  # pyright: ignore
            except RuntimeError as exc:
                status["is_valid"] = False
                status["message"] = repr(exc)

        return result

    def calculate_covariance(self, ncall: int | None = None) -> "NDArray":
        """
        Calculates covariance matrix within the Hesse algorithm.

        The Hesse method estimates the covariance matrix by inverting the matrix of second derivatives
        (Hesse matrix) at the minimum. To get parameters correlations, you need to use this.
        The Minos algorithm is another way to estimate parameter uncertainties, see `profile_errors()`.

        .. note:: by default the covariance matrix is already calculated and stored in `result["covariance"]`
        """
        res = self._minimizer.hesse(ncall=ncall)
        return res.covariance

    def get_scans(self, names: list, fitresult: dict):
        """
        Get Minos profile over a specified interval.

        Scans over one parameter and minimises the function with respect to all other parameters
        for each scan point.
        """
        scans = fitresult["scan"] = {}
        if names:
            self.logger.info(f"Caclulating mnprofile for: {names}")
        for name in names:
            scan = scans[name] = {}
            try:
                xout, yout, valid = self._minimizer.mnprofile(name)
            except CppRuntimeError as exc:
                self._on_exception_in_get_scans(
                    scan, f"{exc.what()}"
                )  # pyright: ignore
            except RuntimeError as exc:
                self._on_exception_in_get_scans(scan, repr(exc))
            else:
                scan["x"] = xout.tolist()
                scan["y"] = yout.tolist()
                scan["success"] = valid.tolist()
                scan["message"] = ""

    def _on_exception_in_get_scans(self, scan: dict, msg: str):
        scan["x"] = []
        scan["y"] = []
        scan["success"] = False
        scan["message"] = msg
