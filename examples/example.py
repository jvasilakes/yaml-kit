from experiment_config import BaseConfig, parameter, deprecate, ConfigKeyError


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

    @parameter(group="Model", default={"input_dim": 0,
                                       "output_dim": 0,
                                       "dropout_prob": 0.0,
                                       "activation_fn": "relu"})
    def model_kwargs(self):
        """
        This parameter showcases possible advanced usage.
        """
        type_map = {
                "input_dim": int,
                "output_dim": int,
                "dropout_prob": float,
                "activation_fn": str
                }
        valid_activation_fns = [
                "relu",
                "tanh",
                "sigmoid"
                ]
        if "input_dim" not in self._model_kwargs.keys():
            raise ConfigKeyError(f"Missing required key 'input_dim' to parameter model_kwargs")  # noqa
        if "output_dim" not in self._model_kwargs.keys():
            raise ConfigKeyError(f"Missing required key 'output_dim' to parameter model_kwargs")  # noqa
        for (key, val) in self._model_kwargs.items():
            try:
                assert isinstance(val, type_map[key]), f"Expected value of type {type_map[key]} for key {key}. Got {type(val)}."  # noqa
            except KeyError:
                raise ValueError(f"Unsupported kwarg {key}.")

        if "activation_fn" in self._model_kwargs:
            assert self._model_kwargs["activation_fn"] in valid_activation_fns, f"activation_fn must be one of {valid_activation_fns}"  # noqa
        else:
            self._model_kwargs["activation_fn"] = \
                    self.DEFAULT_PARAM_VALUES["model_kwargs"]["activation_fn"]
        if "dropout_prob" in self._model_kwargs:
            assert 0.0 <= self._model_kwargs["dropout_prob"] <= 1.0, f"dropout_prob must be in [0, 1]. Got {self._model_kwargs['dropout_prob']}"  # noqa
        else:
            self._model_kwargs["dropout_prob"] = \
                    self.DEFAULT_PARAM_VALUES["model_kwargs"]["dropout_prob"]

        return self._model_kwargs


if __name__ == "__main__":
    model_kwargs = {"input_dim": 2,
                    "output_dim": 5,
                    "activation_fn": "relu"}
    config = ExampleConfig(metric="accuracy", loss_fn="mse",
                           model_kwargs=model_kwargs)
    print(config)
    config.yaml("example.yaml")

    config = ExampleConfig.from_yaml_file("example.yaml")
    print()
    print("From example.yaml")
    print(config)

    config = ExampleConfig.from_yaml_file("example.yaml", batch_size=10,
                                          thing="that")  # Invalid parameter
    print()
    print("From example.yaml with overridden kwargs")
    print(config)
    config.yaml("example_out.yaml")
    print("Saved config to example_out.yaml")
