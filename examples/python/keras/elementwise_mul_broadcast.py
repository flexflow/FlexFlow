from flexflow.keras.layers import Dense, Input, Reshape, Multiply
import flexflow.keras.optimizers
import flexflow.core as ff
import numpy as np

def broadcast1():
  input0 = Input(shape=(16*2,), dtype="float32")
  input1 = Input(shape=(10*1,), dtype="float32")

  x0 = Dense(20, activation='relu')(input0) # B,20
  x1 = Dense(10, activation='relu')(input1) # B,10

  nx0 = Reshape((10,2))(x0) # B,10,2
  nx1 = Reshape((10,1))(x1) # B,10,1

  m0 = Multiply()([nx1, nx0]) # B,10,2
  f0 = Reshape((20,))(m0) # B,20

  out = Dense(1)(f0) # B,1

  model = flexflow.keras.models.Model([input0, input1], out)

  opt = flexflow.keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
  print(model.summary())
  model.fit(
    x = [
      np.random.randn(300, 16*2).astype(np.float32),
      np.random.randn(300, 10*1).astype(np.float32),
    ],
    y = np.random.randn(300, 1).astype(np.float32),
    epochs = 2
  )


def broadcast2():
  input0 = Input(shape=(16*2,), dtype="float32")
  input1 = Input(shape=(10*1,), dtype="float32")

  x0 = Dense(20, activation='relu')(input0) # B,20
  x1 = Dense(10, activation='relu')(input1) # B,10

  nx0 = Reshape((10,2))(x0) # B,10,2
  nx1 = Reshape((10,1))(x1) # B,10,1

  m0 = Multiply()([nx0, nx1]) # B,10,2
  f0 = Reshape((20,))(m0) # B,20

  out = Dense(1)(f0) # B,1

  model = flexflow.keras.models.Model([input0, input1], out)

  opt = flexflow.keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
  print(model.summary())
  model.fit(
    x = [
      np.random.randn(300, 16*2).astype(np.float32),
      np.random.randn(300, 10*1).astype(np.float32),
    ],
    y = np.random.randn(300, 1).astype(np.float32),
    epochs = 2
  )


def broadcast_both():
  input0 = Input(shape=(16*2,), dtype="float32")
  input1 = Input(shape=(10*1,), dtype="float32")

  x0 = Dense(2, activation='relu')(input0) # B,20
  x1 = Dense(10, activation='relu')(input1) # B,10

  nx0 = Reshape((1,2))(x0) # B,1,2
  nx1 = Reshape((10,1))(x1) # B,10,1

  m0 = Multiply()([nx0, nx1]) # B,10,2
  f0 = Reshape((20,))(m0) # B,20

  out = Dense(1)(f0) # B,1

  model = flexflow.keras.models.Model([input0, input1], out)

  opt = flexflow.keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
  print(model.summary())
  model.fit(
    x = [
      np.random.randn(300, 16*2).astype(np.float32),
      np.random.randn(300, 10*1).astype(np.float32),
    ],
    y = np.random.randn(300, 1).astype(np.float32),
    epochs = 2
  )

if __name__ == '__main__':
    configs = ff.get_configs()
    ff.init_flexflow_runtime(configs)
    broadcast1()
    broadcast2()
    broadcast_both()
