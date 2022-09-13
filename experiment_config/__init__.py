import os
import warnings


def parameter(name, group="Default", default=None):
    def wrapper(func):
        if group not in BaseConfig.GROUPED_PARAMETERS:
            BaseConfig.GROUPED_PARAMETERS[group] = {}
        BaseConfig.GROUPED_PARAMETERS[group][name] = func
        if default is not None:
            BaseConfig.DEFAULT_PARAM_VALUES[name] = default
        return property(func)
    return wrapper


def deprecate(name):
    def wrapper(func):
        BaseConfig.DEPRECATED_PARAMETERS.append(name)
        return func
    return wrapper


class ConfigError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BaseConfig(object):
    GROUPED_PARAMETERS = {}
    DEFAULT_PARAM_VALUES = {}
    DEPRECATED_PARAMETERS = []

    @classmethod
    def from_yaml_file(cls, filepath):
        raise NotImplementedError()

    def __init__(self, **kwargs):
        seen_kws = set()
        for (group, params) in self.GROUPED_PARAMETERS.items():
            for (param_name, param) in params.items():
                try:
                    param_val = kwargs[param_name]
                except KeyError:
                    if param_name in self.DEFAULT_PARAM_VALUES:
                        param_val = self.DEFAULT_PARAM_VALUES[param_name]
                    else:
                        raise ConfigError(f"Missing keyword argument '{param_name}'")  # noqa
                setattr(self, f"_{param_name}", param_val)
                seen_kws.add(param_name)

        all_kws = set(kwargs.keys())
        unseen_kws = all_kws.difference(seen_kws)
        if len(unseen_kws) > 0:
            warnings.warn(f"The following keyword arguments are unused: {unseen_kws}")  # noqa

        self.validate()

    def parameters(self):
        return {param_name: param
                for group in self.GROUPED_PARAMETERS.values()
                for (param_name, param) in group.items()}

    def validate(self):
        for (param_name, param) in self.parameters().items():
            param(self)  # runs any code in the property
            if param_name in self.DEPRECATED_PARAMETERS:
                warnings.warn(f"{param_name} is deprecated.")

    def __str__(self):
        formatted = ''
        sorted_groups = sorted(self.GROUPED_PARAMETERS.items(),
                               key=lambda x: x[0])
        for (group, params) in sorted_groups:
            formatted += f"{group}:\n"
            sorted_params = sorted(params.items(), key=lambda x: x[0])
            for (name, prop) in sorted_params:
                formatted += f"  {name}: {prop(self)}"
                if name in self.DEPRECATED_PARAMETERS:
                    formatted += " (deprecated)"
                formatted += '\n'
            formatted += '\n'
        return formatted.strip()

    def update(self, param_name, value):
        setattr(self, f"_{param_name}", value)
        self.validate()

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for (param_name, param) in self.parameters().items():
            this_val = param(self)
            that_val = param(other)
            if this_val != that_val:
                return False
        return True

    def copy(self):
        kwargs = {param_name: param(self) for (param_name, param)
                  in self.parameters().items()}
        return self.__class__(**kwargs)

    def save_to_yaml(self, outpath):
        raise NotImplementedError()

    def git_info(self):
        # log commit hash
        with os.popen("git rev-parse --abbrev-ref HEAD") as p:
            branch = p.read().strip()
        with os.popen("git log --pretty=format:'%h' -n 1") as p:
            commit = p.read().strip()
        if branch == '':
            branch = None
        if commit == '':
            commit = None
        if None in (branch, commit):
            warnings.warn("Error getting current git information. Are you in a git repo?")  # noqa
        return {"branch": branch, "commit": commit}
