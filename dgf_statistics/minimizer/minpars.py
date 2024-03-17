from __future__ import annotations

from typing import TYPE_CHECKING

from dagflow.parameters import GaussianParameter, Parameter

if TYPE_CHECKING:
    from dagflow.output import Output


class MinPar:
    __slots__ = (
        "_parent",
        "_par",
        "_value",
        "_step",
        "_constrained",
        "_scanvalues",
        "_fixed",
    )

    _parent: MinPars
    _par: Parameter
    _value: float
    _step: float

    def __init__(self, parent, par, *, initial_central: bool = True, **kwargs):
        if not isinstance(parent, MinPars):
            raise RuntimeError(
                f"'parent' must be a MinPars object, but given {parent=}, {type(parent)=}"
            )
        self._parent = parent
        self._par = par

        self.setup(initial_central=initial_central, **kwargs)

    @property
    def output(self) -> Output:
        return self._par.output

    @property
    def name(self) -> str | None:
        return self._par.output.name

    def __str__(self):
        return "{name:<25} {_value:8}, step={_step}".format(**self.__dict__)

    def setup(self, *, initial_central: bool = True, **kwargs):
        value = kwargs.pop("value", None)
        if value is None:
            if initial_central and isinstance(self._par, GaussianParameter):
                self.value = self._par.central
            else:
                self.value = self._par.value
        else:
            self.value = value

        step = kwargs.pop("step", None)
        if step == 0.0:
            raise RuntimeError(f"'{self.name}' initial step is undefined. Specify its sigma explicitly.")
        self.step = step

        assert not kwargs, "Unparsed MinPar arguments: {!s}".format(kwargs)

    @property
    def par(self):
        return self._par

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self._parent.modified = True

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, step):
        self._step = step
        self._parent.modified = True


class MinPars:
    _initial_central = True

    def __init__(self, pars: dict, *, initial_central: bool = True):  # check=None,
        self._specs = {}
        self._parmap = {}
        self._modified = True
        self._resized = True
        self._initial_central = initial_central

        self._skippars = []
        for v in pars.values():
            self.addpar(v)

    def __str__(self):
        return str(self._specs)

    def dump(self):
        for i, v in enumerate(self._specs.values()):
            print("% 3d" % i, v)

    def parspec(self, idx):
        if isinstance(idx, str):
            return self._specs[idx]
        elif isinstance(idx, int):
            return list(self._specs.values())[idx]

        # Assume idx is parameter instance
        return self._parmap[idx]

    __getitem__ = parspec

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __len__(self):
        return len(self._specs)

    def names(self):
        return self._specs.keys()

    def specs(self):
        return self._specs.values()

    def values(self):
        return [par.value for par in self._specs.values()]

    def items(self):
        return self._specs.items()

    def npars(self):
        return len(self._specs)

    def resetstatus(self):
        self.modified = False
        self.resized = False

    @property
    def modified(self):
        return self._modified

    @modified.setter
    def modified(self, modified):
        self._modified = modified

    @property
    def resized(self):
        return self._resized

    @resized.setter
    def resized(self, resized):
        self._resized = resized

    def addpar(self, par, **kwargs):
        name = par.output.node.name
        if name in self._specs or par in self._specs.values():
            raise RuntimeError(f"The parameter {name} added twice")

        spec = self._specs[name] = MinPar(self, par, initial_central=self._initial_central, **kwargs)
        self._parmap[par] = spec

        self.modified = True
        self.resized = True


#    def __enter__(self):
#        pass
#
#    def __exit__(self, exc_type, exc_value, traceback):
#        pass
