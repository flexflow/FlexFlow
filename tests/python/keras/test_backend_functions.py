import numpy as np
import flexflow
from flexflow.keras.backend import cos, sin
from flexflow.keras.layers import Input


def test_sin():
    inp = Input(shape=(16,), dtype="float32")
    out = sin(inp)

    model = flexflow.keras.models.Model(inp, out)

    opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=opt, loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    model.fit(
        x = np.random.randn(300, 16).astype(np.float32),
        y = np.random.randn(300, 16).astype(np.float32))


if __name__ == '__main__':
  test_sin()
