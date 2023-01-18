import os
import argparse

from experiment_config import Config


config = Config("ExampleConfig")


@config.parameter(group="Losses")
def loss_fn(val):
    assert val in ["cross-entropy", "mse"]


@config.parameter(types=str)
def metric(val):
    pass


@config.parameter(group="Training", default=2, types=int)
def batch_size(val):
    """
    Docstrings will be saved into the config and printed
    as comments in the yaml file.
    """
    assert val > 0


@config.parameter(group="Training", default=False, deprecated=True)
def use_old_method(val):
    pass


@config.parameter(group="Model", default=0.0, types=float)
def dropout_prob(val):
    assert 0.0 <= val <= 1.0


@config.parameter(group="Model.Encoder", default=5, types=int)
def input_dim(val):
    assert val > 0


@config.parameter(group="Model.Encoder", default=5, types=int)
def hidden_dim(val):
    """
    Must be the same as Model.Decoder.input_dim
    """
    assert val > 0


# It's okay to overload a parameter name if they belong to different groups.
@config.parameter(group="Model.Decoder", default=5, types=int)
def input_dim(val):
    """
    Must be the same as Model.Encoder.hidden_dim
    """
    assert val > 0


@config.parameter(group="Model.Decoder", default=5, types=int)
def output_dim(val):
    assert val > 0


# The on_load decorator can be used to set restrictions
# on combinations of parameters.
@config.on_load
def validate_parameters():
    assert config.Model.Encoder.hidden_dim == config.Model.Decoder.input_dim


# The following is an example of how you can create a command
# line tool to work with config files.
def update_config(filepath, **updates):
    config.load_yaml(filepath)
    for (key, value) in updates.items():
        tmp = key.split('.')
        group = '.'.join(tmp[:-1])
        param = tmp[-1]
        config.update(param, value, group=group)
    os.rename(filepath, f"{filepath}.orig")
    config.yaml(filepath)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    print_parser = subparsers.add_parser(
        "print", help="Print a config file to the terminal.")
    print_parser.add_argument("filepath", type=str,
                              help="The yaml file to print.")

    newconf_parser = subparsers.add_parser(
        "new", help="Save a new default config file.")
    newconf_parser.add_argument("filepath", type=str,
                                help="Where to save the new config file.")

    update_parser = subparsers.add_parser(
        "update", help="Update one or more config files with new parameter values.")  # noqa
    update_parser.add_argument("-p", "--param", nargs=2, action="append",
                               help="E.g., -p Model.Encoder.input_dim 2")
    update_parser.add_argument("-f", "--files", nargs='+', type=str,
                               help="Config files to update.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "print":
        config.load_yaml(args.filepath)
        print(config)

    elif args.command == "new":
        config.yaml(args.filepath)

    elif args.command == "update":
        if args.param is None:
            update_params = {}
        else:
            update_params = dict(args.param)
        for filepath in args.files:
            update_config(filepath, **update_params)
