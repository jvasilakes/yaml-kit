from yaml_kit import Config, get_and_run_config_command

config = Config("MyConfig")


@config.parameter(group="Experiment", types=str)
def name(val):
    # Code to validate or modify `val` or `pass` to do nothing.
    assert len(val) > 0


@config.parameter(group="Model.Encoder", types=int, default=10)
def input_dim(val):
    assert val > 0


@config.parameter(group="Model.Encoder", types=int, default=5)
def output_dim(val):
    assert val > 0


@config.parameter(group="Model.Decoder", types=int, default=5)  # noqa
def input_dim(val):
    assert val > 0


@config.parameter(group="Model.Decoder", types=int, default=3)  # noqa
def output_dim(val):
    assert val > 0


@config.parameter(group="Model.Decoder", types=list, default=[5, 3], deprecated=True)
def shape(val):
    assert len(val) == 2
    for member in val:
        assert isinstance(member, int)
        assert member > 0


@config.on_load
def validate_model_shapes():
    assert config.Model.Encoder.output_dim == config.Model.Decoder.input_dim


if __name__ == "__main__":
    get_and_run_config_command(config)
