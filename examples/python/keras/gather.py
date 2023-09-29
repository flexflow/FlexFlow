from flexflow.keras.layers import Dense, Input, Reshape
from flexflow.keras.backend.internal import gather
import flexflow.keras.optimizers
import flexflow.core as ff
import numpy as np


def get_modified_idx(idx, hidden_shape):
    return idx.reshape(-1, 1).repeat(hidden_shape, 1).astype(np.int32)


def gather_example():
    h = 3
    idx = np.array([[5, 7, 10], [8, 4, 0]])
    # Convert idx to that required by torch.gather
    idx = get_modified_idx(idx, h)  # 6,3

    input0 = Input(shape=(10,), dtype="float32")
    input1 = Input(shape=idx.shape, dtype="int32")

    x0 = Dense(60, activation='relu')(input0)  # B,60
    x0 = Reshape((20, h))(x0)  # B,20,3
    f0 = gather(x0, input1, axis=1)  # B,6,3
    f0 = Reshape((18,))(f0)

    out = Dense(1)(f0)  # B,1

    model = flexflow.keras.models.Model([input0, input1], out)

    opt = flexflow.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
      optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
    print(model.summary())
    model.fit(
        x=[
            np.random.randn(300, 10).astype(np.float32),
            idx[None, ...].repeat(300, 0).astype(np.int32)
        ],
        y=np.random.randn(300, 1).astype(np.float32),
        epochs=2
    )


if __name__ == '__main__':
    configs = ff.get_configs()
    ff.init_flexflow_runtime(configs)
    gather_example()
