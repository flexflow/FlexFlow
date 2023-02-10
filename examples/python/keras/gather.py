from flexflow.keras.layers import Dense, Input, Reshape
from flexflow.keras.backend.internal import gather
from flexflow.keras.backend import sum
import flexflow.keras.optimizers
from flexflow.core import FFConfig

import numpy as np

    # idxt = torch.tensor(idx).reshape(1,-1,1).repeat(x.shape[0], 1, x.shape[-1])
    # return torch.gather(x, 1, idxt).reshape(x.shape[0], *idx.shape, -1)


def gather_example():
  ffconfig = FFConfig()

  idx = np.array([[5,7,10],[8,4,0]])
  idx = idx.reshape(-1, 1).repeat(3, -1)  # 6,3

  input0 = Input(shape=(10,), dtype="float32")
  input1 = Input(shape=idx.shape, dtype="int32")

  x0 = Dense(60, activation='relu')(input0)  # B,60
  x0 = Reshape((20, 3))(x0)  # B,20,3
  f0 = gather(x0, input1, axis=1) # B,6,3
  f0 = Reshape((18,))(f0)
#   f1 = Reshape((2,3,3))(f0)  # B,2,3,3
#   f2 = sum(f1, axis=2)  # B,2,3

  out = Dense(1)(f0) # B,1

  model = flexflow.keras.models.Model([input0, input1], out)

  opt = flexflow.keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
  print(model.summary())
  model.fit(
    x = [
      np.random.randn(300, 10).astype(np.float32),
      idx[None, ...].repeat(300, 0).astype(np.int32)
    ],
    y = np.random.randn(300, 1).astype(np.float32),
    epochs = 2
  )


if __name__ == '__main__':
    gather_example()
