from flexflow.keras.layers import Dense, Input, Maximum, Minimum
import flexflow.keras.optimizers
import flexflow.core as ff

import numpy as np

def elementwise_max():
  input0 = Input(shape=(16*2,), dtype="float32")
  input1 = Input(shape=(10*1,), dtype="float32")

  x0 = Dense(20, activation='relu')(input0) # B,20
  x1 = Dense(20, activation='relu')(input1) # B,20

  f0 = Maximum()([x0, x1]) # B,20

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


def elementwise_min():
  input0 = Input(shape=(16*2,), dtype="float32")
  input1 = Input(shape=(10*1,), dtype="float32")

  x0 = Dense(20, activation='relu')(input0) # B,20
  x1 = Dense(20, activation='relu')(input1) # B,20

  f0 = Minimum()([x0, x1]) # B,20

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

def get_configs():
  import argparse,json
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-config-file",
    help="The path to a JSON file with the configs. If omitted, a sample model and configs will be used instead.",
    type=str,
    default=None,
  )
  args = parser.parse_args()
  if args.config_file is not None:
    with open(args.config_file) as f:
      return json.load(f)
  else:
    return None

if __name__ == '__main__':
    configs = get_configs()
    ff.init_flexflow_runtime(configs)
    elementwise_max()
    elementwise_min()
