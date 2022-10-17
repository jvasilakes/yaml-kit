import os
import yaml
import warnings
import colorama as cr

cr.init()


def parameter(group="Default", default=None, type=None):
    """
    A decorator for marking a config parameter.

    .. code-block:: python

       from experiment_config import BaseConfig, parameter

       class MyConfig(BaseConfig):

           def __init__(self, **kwargs):
               super().__init__(**kwargs)

           @parameter(group="Group1", type=(str, int), default="hello!")
           def thing(self):
               # Any additional initialization/validation code here
               return self._thing

    :param str group: (Optional) The parameter group to add this parameter to.
                      If not specified, adds the Default group.
    :param Any default: (Optional) If specified, sets a default
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

    .. code-block:: python

       from experiment_config import BaseConfig, parameter

       class MyConfig(BaseConfig):

           def __init__(self, **kwargs):
               super().__init__(**kwargs)

           @deprecate
           @parameter()
           def thing(self):
               # Any additional initialization/validation code here
               return self._thing
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


class ConfigKeyError(ConfigError):
    """
    Exception for when a required parameter is missing.
    """
    pass


class ConfigTypeError(ConfigError):
    """
    Exception for when a parameter value type does
    not match that specified in the `@parameter` decorator.
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
    `BaseConfig.from_yaml_file()`.
    """
    pass


class BaseConfig(object):
    GROUPED_PARAMETERS = {}
    DEFAULT_PARAM_VALUES = {}
    PARAM_TYPES = {}
    DEPRECATED_PARAMETERS = []
    VERSION_INFO = {}

    """
    The base experiment config class, which should be subclassed
    to create your own config. The keyword arguments used to initialize
    your subclass are those methods which you decorate with @parameter.

    .. code-block:: python

       from experiment_config import BaseConfig, parameter

       class MyConfig(BaseConfig):

            # This is always necessary
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            @parameter()
            def my_parameter(self):
                # Add any initialization/validation code here
                # The @parameter decorator creates a _my_parameter variable.
                return self._my_parameter

        config = MyConfig(my_parameter=2)
    """

    @classmethod
    def from_yaml_file(cls, filepath, **override_kwargs):
        """
        Loads a config from the specified yaml file.
        Override the values of any parameters in the yaml file
        using keyword arguments.

        :param str filepath: Path to the .yaml file.
        """
        with open(filepath, 'r') as inF:
            config_dict = yaml.safe_load(inF)
        for (key, val) in override_kwargs.items():
            config_dict[key] = val
        return cls(**config_dict, overridden_kwargs=override_kwargs.keys())

    @classmethod
    def save_default(cls, outpath):
        """
        Create a new yaml file with default parameter values to the
        specified outpath. Any parameters without default values
        will be save with an empty string.

        :param str outpath: Path to a .yaml file to save.
        """
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
                    if param_name in self.PARAM_TYPES.keys():
                        # Try to cast the value to a supported type
                        param_types = self.PARAM_TYPES[param_name]
                        if not isinstance(param_types, (list, tuple)):
                            param_types = [param_types]
                        if not type(param_val) in param_types:
                            success = False
                            for ptype in param_types:
                                try:
                                    param_val = ptype(param_val)
                                    success = True
                                    break
                                except ValueError:
                                    continue
                            if success is False:
                                raise ConfigTypeError(f"Can't cast value {param_value} to supported types {param_types}")  # noqa
                except KeyError:
                    if param_name in self.DEFAULT_PARAM_VALUES:
                        param_val = self.DEFAULT_PARAM_VALUES[param_name]
                    else:
                        raise ConfigKeyError(f"Missing keyword argument '{param_name}'")  # noqa
                setattr(self, f"_{param_name}", param_val)
                seen_kws.add(param_name)

        self.overridden_kwargs = {}
        if "overridden_kwargs" in kwargs.keys():
            self.overridden_kwargs = kwargs.pop("overridden_kwargs")

        if "branch" in kwargs or "commit" in kwargs:
            assert "branch" in kwargs and "commit" in kwargs, ConfigError("Must include both 'branch' and 'commit'")  # noqa
            seen_kws.update({"branch", "commit"})
            curr_version = self._git_info()
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

    def group(self, group_name=None):
        """
        Get the specified parameter group by name.

        :param str group_name: (Optional) The group name to get. If None,
                               returns the Default group.
        :returns: The group of parameters.
        :rtype: dict
        :raises: ConfigError
        """
        if group_name is None:
            group_name = "Default"
        try:
            return {param_name: param(self) for (param_name, param) in
                    self.GROUPED_PARAMETERS[group_name].items()}
        except KeyError:
            raise ConfigError(f"{self.__class__} has no parameter group {group_name}.")  # noqa

    def parameters(self):
        """
        Return the ungrouped parameters.

        :rtype: dict
        """
        return {param_name: param(self)
                for group in self.GROUPED_PARAMETERS.values()
                for (param_name, param) in group.items()}

    def validate(self):
        """
        Validate all parameters by
         1. Running the code specified in the parameter.
         2. Checking if the types are correct.
         3. Checking if the parameter is deprecated.

        :raises: ConfigTypeError, ConfigWarning
        """
        # Calling self.parameters() runs any code in the parameter block.
        for (param_name, param_val) in self.parameters().items():
            # Check if the type is correct
            if param_name in self.PARAM_TYPES.keys():
                param_types = self.PARAM_TYPES[param_name]
                if not isinstance(param_val, param_types):
                    raise ConfigTypeError(f"Value {param_val} of parameter '{param_name}' is not of type {param_types}")  # noqa

            # Check if deprecated
            if param_name in self.DEPRECATED_PARAMETERS:
                warnings.warn(f"{param_name} is deprecated.",
                              ConfigWarning)

    def __str__(self):
        formatted = cr.Style.BRIGHT + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        formatted += cr.Style.RESET_ALL
        sorted_groups = sorted(self.GROUPED_PARAMETERS.items(),
                               key=lambda x: x[0])
        for (group, params) in sorted_groups:
            group_str = cr.Style.BRIGHT + f"{group}\n" + cr.Style.RESET_ALL
            formatted += group_str
            sorted_params = sorted(params.items(), key=lambda x: x[0])
            for (name, prop) in sorted_params:
                format_str = f" • {name}: {prop(self)}"
                if name in self.DEPRECATED_PARAMETERS:
                    format_str = cr.Fore.RED + format_str + " (deprecated)"
                    format_str += cr.Style.RESET_ALL
                if name in self.overridden_kwargs:
                    format_str = cr.Fore.YELLOW + format_str + " (overridden)"
                    format_str += cr.Style.RESET_ALL
                formatted += format_str + '\n'
        git = self._git_info()
        if git is not None:
            group_str = cr.Style.BRIGHT + "Git\n" + cr.Style.RESET_ALL
            formatted += group_str
            for (key, val) in git.items():
                formatted += f" • {key}: {val}\n"
        formatted += cr.Style.BRIGHT + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        formatted += cr.Style.RESET_ALL
        return formatted.strip()

    def update(self, param_name, value):
        """
        Update the value of a parameter.

        :param str param_name: The parameter to update.
        :param Any value: The new value.
        """
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
        """
        Copy this BaseConfig instance.
        """
        kwargs = {param_name: param(self) for (param_name, param)
                  in self.parameters().items()}
        return self.__class__(**kwargs)

    def yaml(self, outpath=None):
        """
        Save this config in yaml format to the specified path.

        :param str outpath: Where to save the config.
        """
        yaml_str = ''
        outF = None
        if outpath is not None:
            outF = open(outpath, 'w')
        for (group, params) in self.GROUPED_PARAMETERS.items():
            yaml_str += f"# {group}\n"
            params_with_values = {name: param(self)
                                  for (name, param) in params.items()}
            yaml_str += yaml.dump(params_with_values) + '\n'
        git = self._git_info()
        if git is not None:
            yaml_str += "# Git\n"
            yaml_str += yaml.dump(git)
        if outpath is not None:
            with open(outpath, 'w') as outF:
                outF.write(yaml_str)
        return yaml_str

    def _git_info(self):
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
