import flexflow.keras as keras
from flexflow.keras.layers import Dense, Input, Reshape
from flexflow.keras.backend.internal import gather
import flexflow.keras.optimizers
import flexflow.core as ff
import numpy as np


def regularizer_example():
    input0 = Input(shape=(10,), dtype="float32")

    reg = keras.regularizers.L2(0.001)
    x0 = Dense(16, activation='relu', kernel_regularizer=reg)(input0)
    out = Dense(1)(x0)

    model = flexflow.keras.models.Model(input0, out)

    opt = flexflow.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
      optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
    model.fit(
        x=np.random.randn(300, 10).astype(np.float32),
        y=np.random.randn(300, 1).astype(np.float32),
        epochs=2
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
  args, unknown = parser.parse_known_args()
  if args.config_file is not None:
    with open(args.config_file) as f:
      return json.load(f)
  else:
    return None

if __name__ == '__main__':
    configs = get_configs()
    ff.init_flexflow_runtime(configs)
    regularizer_example()
