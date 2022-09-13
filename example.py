from experiment_config import BaseConfig, parameter, deprecate


class ExampleConfig(BaseConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @parameter("batch_size", group="Training", default=2)
    def hello(self):
        assert isinstance(self._batch_size, int)
        return self._batch_size

    @parameter("loss_fn", group="Losses")
    def pet(self):
        assert self._loss_fn in ["cross-entropy", "mse"]
        return self._loss_fn

    @deprecate("use_old_method")
    @parameter("use_old_method", group="Training", default=False)
    def use_old_method(self):
        return self._use_old_method


if __name__ == "__main__":
    config = ExampleConfig(loss_fn="mse")
    print(config)
    print(config.git_info())
