from yaml_kit import Config, get_and_run_config_command


config = Config("ExampleConfig")


@config.parameter(types=str, default="Example")
def name(val):
    pass


@config.parameter(types=str)
def metric(val):
    assert val in ["accuracy", "f1"]


@config.parameter(group="Losses")
def loss_fn(val):
    assert val in ["cross-entropy", "mse"]


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
@config.parameter(group="Model.Decoder", default=6, types=int)
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

@config.on_load
def modify_parameter():
    config.Model.dropout_prob.value = 0.5


if __name__ == "__main__":
    get_and_run_config_command(config)
