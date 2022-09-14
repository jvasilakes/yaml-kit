from experiment_config import BaseConfig, parameter, deprecate


class ExampleConfig(BaseConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @parameter(type=str)
    def metric(self):
        return self._metric

    @parameter(group="Losses")
    def loss_fn(self):
        assert self._loss_fn in ["cross-entropy", "mse"]
        return self._loss_fn

    @parameter(group="Training", default=2)
    def batch_size(self):
        assert isinstance(self._batch_size, int)
        return self._batch_size

    @deprecate()
    @parameter(group="Training", default=False)
    def use_old_method(self):
        return self._use_old_method


if __name__ == "__main__":
    config = ExampleConfig(metric="accuracy", loss_fn="mse")
    print(config)

    config = ExampleConfig.from_yaml_file("example.yaml")
    print()
    print("From example.yaml")
    print(config)

    config = ExampleConfig.from_yaml_file("example.yaml", batch_size=10,
                                          thing="that")  # Invalid parameter
    print()
    print("From example.yaml with overridden kwargs")
    print(config)
    config.save_to_yaml("example_out.yaml")
