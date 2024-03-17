from collections.abc import Sequence
from typing import TYPE_CHECKING

from iminuit import Minuit
from numpy import array, ascontiguousarray

from .fitresult import FitResult
from .minimizerbase import MinimizerBase

if TYPE_CHECKING:
    from dagflow.output import Output
    from .minpars import MinPars, MinPar

# if we cannot import runtime_error from root we use DagflowError to avoid any exception capture,
# i.e., if CppRuntimeError==DagflowError, the exception will be not raised
try:
    import ROOT  # fmt: skip
    CppRuntimeError = ROOT.std.runtime_error
except Exception:
    from dagflow.exception import DagflowError  # fmt:skip
    CppRuntimeError = DagflowError


class IMinuitMinimizer(MinimizerBase):
    def __init__(
        self,
        statistic: "Output",
        minpars: "MinPars",
        name: str = "iminuit",
        label: str = "iminuit",
        **kwargs,
    ):
        super().__init__(statistic, minpars, name, label, **kwargs)

    def _child_fit(
        self,
        *,
        profile_errors: Sequence | None = None,
        scan: Sequence | None = None,
        covariance: bool = False,
    ) -> dict:
        if profile_errors is None:
            profile_errors = []
        if scan is None:
            scan = []
        assert self.parspecs

        self.setuppars()
        with FitResult() as fr:
            try:
                result = self._minimizer.migrad()
            except CppRuntimeError as exc:
                message = f"{exc.what()}"
                success = False
                fun = None
                result = self._minimizer
            except RuntimeError as exc:
                message = repr(exc)
                success = False
                fun = None
                result = self._minimizer
            else:
                success = result.valid
                message = str(result.fmin)
                fun = result.fval
            finally:
                argmin = array(result.values)
                errors = array(result.errors)
                fr.set(
                    x=argmin,
                    errors=errors,
                    fun=fun,
                    success=success,
                    message=message,
                    minimizer=self._label,
                    nfev=result.nfcn,
                )

        self._result = fr.result
        self.patchresult()

        if self._result["success"]:
            if covariance:
                _, status = self.get_covmatrix()
                self._result["covariance"] = {
                    "matrix": array(result.covariance),
                    "status": status,
                }
            if profile_errors:
                self.profile_errors(profile_errors, self.result)
            if scan:
                self.get_scans(scan, self.result)

        return self.result

    def setuppars(self) -> None:
        minimizable = self.update_minimizable()

        def fcn(x):
            x = ascontiguousarray(x, dtype="d")
            return minimizable(x)

        startvalues = self._startvalues if self._startvalues is not None else self.parspecs.values()
        self._minimizer = Minuit(fcn, startvalues, name=self.parspecs.names())
        self._minimizer.throw_nan = True
        self._minimizer.errordef = 1

        for i, parspec in enumerate(self.parspecs.specs()):
            self.setuppar(i, parspec)

        self.parspecs.resetstatus()

    def setuppar(self, i: int, parspec: "MinPar") -> None:
        self._minimizer.values[i] = parspec.value
        self._minimizer.errors[i] = parspec.step

    def get_covmatrix(self, verbose: bool = False):
        status = self._minimizer.fmin.has_covariance
        covmatrix = array(self._minimizer.covariance) if status else None

        if verbose:
            if status:
                print("Covariance matrix:")
                print(covmatrix)
            else:
                print("Covariance matrix not estimated")

        return covmatrix, not status

    def get_scans(self, names: list, fitresult: dict):
        scans = fitresult["scan"] = {}
        if names:
            print("Caclulating profile for:", end=" ")
        for name in names:
            scan = scans[name] = {}
            try:
                xout, yout, valid = self._minimizer.mnprofile(name)
            except CppRuntimeError as exc:
                self._on_exception_in_get_scans(scan, f"{exc.what()}")
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

    def profile_errors(self, names: list, fitresult: dict):
        errs = fitresult["errors_profile"] = {}
        statuses = fitresult["errors_profile_status"] = {}
        if names:
            print("Caclulating profile error for:", names)

        for name in names:
            status = statuses[name] = {}
            try:
                self._minimizer.minos(name)
            except CppRuntimeError as exc:
                self._on_exception_in_profile_errors(errs, name, status, f"{exc.what()}")
            except RuntimeError as exc:
                self._on_exception_in_profile_errors(errs, name, status, repr(exc))
            else:
                stat = self._minimizer.merrors[name]
                errs[name] = [stat.lower, stat.upper]

                for key in stat.__slots__:
                    status[key] = getattr(stat, key)
                status["message"] = ""

    def _on_exception_in_profile_errors(self, errs: dict, name: str, status: dict, msg: str):
        errs[name] = [None, None]
        status["is_valid"] = False
        status["message"] = msg
