import os
import argparse


def update_config(config, filepath, **updates):
    """
    Load the config with parameters from a yaml file at filepath.
    Update parameters using the key, value pairs in updates.

    Keys in updates use the dot syntax to specify groups, subgroups,
    and parameters. E.g., 'Group.Subgroup.parameter'.
    """
    null_values = ["null", "none", '']
    # TODO: This will ignore errors that aren't set during the updates
    # so some non-updated parameters might still be incorrect.
    config.load_yaml(filepath, errors="ignore")
    for (key, value) in updates.items():
        tmp = key.split('.')
        group = '.'.join(tmp[:-1])
        param = tmp[-1]
        if value.lower() in null_values:
            value = None
        if value is not None:
            if '{}' in value:
                curr_val = config[key].value
                value = modify_path_value(curr_val, value)  # value is a pattern
        config.update(param, value, group=group, run_on_load=False)
    # TODO: This is a bit of a hack, to call _post_load_hook directly.
    config._post_load_hook()
    os.rename(filepath, f"{filepath}.orig")
    config.yaml(filepath)


def modify_path_value(current_value, pattern):
    path_head = current_value
    patt_head = pattern
    tail = ''
    while path_head != '':
        patt_head, patt_tail = os.path.split(patt_head)
        if patt_tail == '{}':
            path_head, path_tail = os.path.split(path_head)
            tail = os.path.join(path_tail, tail)
        else:
            tail = os.path.join(patt_tail, tail)
        if patt_head == '':
            tail = os.path.join(path_head, tail)
            break
    return tail


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    print_parser = subparsers.add_parser(
        "print", help="Print a config file to the terminal.")
    print_parser.add_argument("filepath", type=str,
                              help="The yaml file to print.")

    val_parser = subparsers.add_parser(
        "validate", help="Validate a config file.")
    val_parser.add_argument("filepath", type=str,
                            help="The yaml file to validate.")

    newconf_parser = subparsers.add_parser(
        "new", help="Save a new default config file.")
    newconf_parser.add_argument("filepath", type=str,
                                help="Where to save the new config file.")

    update_parser = subparsers.add_parser(
        "update", help="Update one or more config files with new parameter values.")  # noqa
    update_parser.add_argument(
        "-p", "--param", nargs=2, metavar=("GROUP.PARAM", "VALUE"),
        action="append",
        help="""Update PARAM in GROUP with a new VALUE.
        E.g., `-p Model.Encoder.input_dim 2`. Pattern matching on paths
        is supported using '{}' and '/'. E.g., if the current VALUE is
        'log/path' `-p Experiment.logdir {}/new/{}` will update it to
        'log/new/path'.""")
    update_parser.add_argument("-f", "--files", nargs='+', metavar="FILE",
                               type=str, help="Config files to update.")

    return parser.parse_args()


def run_command(config, args):
    if args.command == "print":
        config.load_yaml(args.filepath)
        print(config)

    elif args.command == "validate":
        config.load_yaml(args.filepath, errors="warn")

    elif args.command == "new":
        config.yaml(args.filepath)

    elif args.command == "update":
        if args.param is None:
            update_params = {}
        else:
            update_params = dict(args.param)
        for filepath in args.files:
            update_config(config, filepath, **update_params)


def get_and_run_config_command(config):
    """
    Given a Config instance, get the command line arguments
    passed to this script and run them. Add the following to
    the end of your config definition file to expose the
    command line interface.

    .. code-block:: python

        myconfig = Config("MyConfig")
        @myconfig.parameter()
        # etc.

        if __name__ == "__main__":
            get_and_run_config_command(myconfig)
    """
    args = parse_args()
    run_command(config, args)
