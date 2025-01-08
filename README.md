# Experiment Config

An extensible, flexible, and version-aware package for configuring and logging ML experiments.


## Description

`experiment_config` is a programmatic way to generate and validate YAML config files with custom, user-defined logic. It supports comments, type checking, and git integration so you always know that your experiments are valid, and you can check how parameter values have changed through code iterations.

## Requirements

 * `ruamel.yaml`
 * `colorama`



## Installation

```
git clone https://github.com/jvasilakes/experiment-config.git
cd experiment_config
python setup.py develop
```

### Uninstallation

```
python setup.py develop --uninstall
```


## Usage

At a high level, `experiment_config` helps you generate and read YAML files. It always starts with creating a `.py` file to define the parameters. This file always has the same basic structure:

```python
from experiment_config import Config, get_and_run_config_command

config = Config("MyConfig")

# config logic goes here

if __name__ == "__main__":
    get_and_run_config_command(config)
```

`Config` is the abstract class in which you will define your parameters. It has a name (here "MyConfig") and built-in methods for saving, loading, printing, etc.


### Defining Parameters

Once you have instantiated a `Config` instance, you need to define your parameters.
Parameters are basically key: value pairs with some extra logic to make them easier to use than a basic `dict`. Parameters are also organized into groups, and each group can have an arbitrary number of subgroups. 

A parameter definition is a function, named as you would the parameter, decorated with `@config.parameter` decorator. The decorator takes the following optional arguments:

  * group (str): the parameter group. Subgroups are specified using dot syntax, e.g., "Group.Subgroup.Subsubgroup".
  * default (any): the default value for this parameter. This value will be filled in automatically when generating a template config.
  * types (type or tuple of types): accepted datatype for this parameter.
  * deprecated (bool): If True, raise a DepracationWarning for this parameter.

Additionally, docstrings in the parameter function are saved and rendered in the YAML. For example,

```python
@config.parameter(group="Model.Encoder", default=1, types=int)
def num_layers(val):
    """
    The number of linear layers in the encoder.
    """
    assert val >= 1
```

will generate the  YAML

```
Model:
  Encoder:
    # The number of linear layers in the encoder.
    num_layers: 1
```

The validation logic in the parameter function can be as simple or complex as you wish. If you don't need any validation besides type-checking, a `pass` will suffice.


### Load Hooks

The validation logic in the parameter functions is often sufficient. However, if you need to validate parameters against each other to ensure compatibility (e.g., you want the `Model.Encoder.output_dim` to match `Model.Decoder.input_dim`) you can use the `@config.on_load` decorator after all your parameter definitions. For example,

```python
@config.on_load
def validate_parameters():
    assert Model.Encoder.output_dim == Model.Decoder.input_dim
```

### Command Line Usage

Now that you've defined your `config.py` file with all your parameters and load hooks, you can use it on the command line like so.

```
$> python config.py new myconfig.yaml
```

You can now edit `myconfig.yaml` to your liking. It will already contain any default values you specified in your parameter functions.

Once you're happy with your config file, validate it with 

```
$> python config.py validate myconfig.yaml
```

If nothing is printed, your config is valid according to your logic. You can also validate by pretty-printing the config file to the terminal.

```
$> python config.py print myconfig.yaml
```

In the course of your work, you may end up with a number of similar config files. You can batch update their parameters like so.

```
$> python config.py update -f configs/*.yaml -p "Model.Encoder.output_dim" 10 -p "Model.Decoder.input_dim" 10
```

In the command above `-f` specifies a list of files to edit and `-p` is a single parameter, new value pair to update. The original version of the file will be saved with a `.yaml.orig` extension in case you need to roll back your changes.

The update command also has basic support for string matching and substitution. The string `{}` in a parameter value will be replaced with the current value in the config file. For example, given the contents of `config.yaml`

```
Experiment:
  logdir: /home/user/experiments/logs
```

We can update `logdir` like so

```
$> python config.py update -f config.yaml -p "Experiment.logdir" "{}/test_experiment"
```

resulting in the following YAML

```
Experiment:
  logdir: /home/user/experiments/logs/test_experiment
```

### API

Using your config in Python is simple. Assuming your config definition is saved as `myconfig.py` and the corresponding YAML is `myconfig.yaml`, you can do.

```python
# Import your config instance. This contains all your parameter definitions,
# but no values besides any defaults you defined.
from myconfig import config

# Load your parameter values from the YAML file.
config.load_yaml("myconfig.yaml")

# Access parameter values directly
config.Model.Encoder.input_dim.value

# or iterate over groups and subgroups/parameters
for group in config:
    for subgroup_or_param in group:
        pass

# You can also convert to a dictionary
config.asdict()
```

## Examples

See `examples/`.
