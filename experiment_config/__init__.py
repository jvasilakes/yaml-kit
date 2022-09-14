import os
import yaml
import warnings


def parameter(group="Default", default=None, type=None):
    """
    A decorator for marking a config parameter.

    ```
    from experiment_config import BaseConfig, parameter

    class MyConfig(BaseConfig):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        @parameter(group="Group1", type=(str, int), default="hello!")
        def thing(self):
            # Any additional initialization/validation code here
            return self._thing
     ```

    :param str group: (Optional) The parameter group to add this parameter to.
                      If not specified, adds the Default group.
    :param default Any: (Optional) If specified, sets a default
                        value for this parameter.
    :param type type: (Optional) Can be a single type or a tuple of
                      types. If specified, restricts the values to
                      these types.
    """
    def wrapper(func):
        if group not in BaseConfig.GROUPED_PARAMETERS:
            BaseConfig.GROUPED_PARAMETERS[group] = {}
        name = func.__name__
        BaseConfig.GROUPED_PARAMETERS[group][name] = func
        if default is not None:
            BaseConfig.DEFAULT_PARAM_VALUES[name] = default
        if type is not None:
            BaseConfig.PARAM_TYPES[name] = type
        return property(func)
    return wrapper


def deprecate():
    """
    Mark the decorated parameter as deprecated.
    This will raise a warning when it is accessed.

    ```
    from experiment_config import BaseConfig, parameter

    class MyConfig(BaseConfig):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        @deprecate
        @parameter()
        def thing(self):
            # Any additional initialization/validation code here
            return self._thing
     ```
    """
    def wrapper(func):
        if isinstance(func, property):
            func = func.fget
        name = func.__name__
        BaseConfig.DEPRECATED_PARAMETERS.append(name)
        return func
    return wrapper


class ConfigError(Exception):
    """
    The default Experiment Config Exception
    """
    pass


class ConfigTypeError(ConfigError):
    """
    Exception for when a parameter value type does
    not match that specified in the @parameter decorator.
    """
    pass


class ConfigWarning(UserWarning):
    """
    The default Experiment Config warning.
    """
    pass


class ConfigVersionWarning(ConfigWarning):
    """
    Warning for when the current git version does not
    match the version read from a yaml file by
    BaseConfig.from_yaml_file().
    """
    pass


class BaseConfig(object):
    GROUPED_PARAMETERS = {}
    DEFAULT_PARAM_VALUES = {}
    PARAM_TYPES = {}
    DEPRECATED_PARAMETERS = []
    VERSION_INFO = {}

    @classmethod
    def from_yaml_file(cls, filepath, **override_kwargs):
        with open(filepath, 'r') as inF:
            config_dict = yaml.safe_load(inF)
        for (key, val) in override_kwargs.items():
            config_dict[key] = val
        return cls(**config_dict)

    @classmethod
    def save_default(cls, outpath):
        with open(outpath, 'w') as outF:
            for (group, params) in cls.GROUPED_PARAMETERS.items():
                outF.write(f"# {group}\n")
                params_with_defaults = {}
                for param_name in params.keys():
                    try:
                        value = cls.DEFAULT_PARAM_VALUES[param_name]
                    except KeyError:
                        value = ''
                    params_with_defaults[param_name] = value
                yaml.dump(params_with_defaults, outF)
                outF.write('\n')

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

        if "branch" in kwargs or "commit" in kwargs:
            assert "branch" in kwargs and "commit" in kwargs, ConfigError("Must include both 'branch' and 'commit'")  # noqa
            seen_kws.update({"branch", "commit"})
            curr_version = self.git_info()
            diff_version = False
            if kwargs["branch"] != curr_version["branch"]:
                diff_version = True
            if kwargs["commit"] != curr_version["commit"]:
                diff_version = True
            if diff_version is True:
                warn_str = "Config versions differ:\n"
                for (k, v) in curr_version.items():
                    warn_str += f"  {k}: {kwargs[k]} -> {curr_version[k]}\n"
                warn_str = warn_str.strip()
                warnings.warn(warn_str, ConfigVersionWarning)

        all_kws = set(kwargs.keys())
        unseen_kws = all_kws.difference(seen_kws)
        if len(unseen_kws) > 0:
            warnings.warn(f"The following keyword arguments are unused: {unseen_kws}",  # noqa
                          ConfigWarning)

        self.validate()

    def group(self, group_name):
        try:
            return self.GROUPED_PARAMETERS[group_name]
        except KeyError:
            raise ConfigError(f"{self.__class__} has no parameter group {group_name}.")  # noqa

    def parameters(self):
        return {param_name: param
                for group in self.GROUPED_PARAMETERS.values()
                for (param_name, param) in group.items()}

    def validate(self):
        for (param_name, param) in self.parameters().items():
            val = param(self)  # runs any code in the property

            # Check if the type is correct
            if param_name in self.PARAM_TYPES.keys():
                param_types = self.PARAM_TYPES[param_name]
                if not isinstance(val, param_types):
                    raise ConfigTypeError(f"Value {val} of parameter '{param_name}' is not of type {param_types}")  # noqa

            # Check if deprecated
            if param_name in self.DEPRECATED_PARAMETERS:
                warnings.warn(f"{param_name} is deprecated.",
                              ConfigWarning)

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
        git = self.git_info()
        if git is not None:
            formatted += "Git:\n"
            for (key, val) in git.items():
                formatted += f"  {key}: {val}\n"
        return formatted.strip()

    def update(self, param_name, value):
        if param_name not in self.parameters().keys():
            raise ConfigError(f"{self.__class__} has no attribute {param_name}")  # noqa
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
        with open(outpath, 'w') as outF:
            for (group, params) in self.GROUPED_PARAMETERS.items():
                outF.write(f"# {group}\n")
                params_with_values = {name: param(self)
                                      for (name, param) in params.items()}
                yaml.dump(params_with_values, outF)
                outF.write('\n')
            git = self.git_info()
            if git is not None:
                outF.write("# Git\n")
                yaml.dump(git, outF)

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
            warnings.warn("Error getting current git information. Are you in a git repo?",  # noqa
                          ConfigWarning)
            return None
        return {"branch": branch, "commit": commit}
