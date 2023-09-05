# Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from flexflow.keras.layers import Dense, Input
from flexflow.keras.backend.internal import rsqrt
import flexflow.keras.optimizers
import flexflow.core as ff
import numpy as np

def test_rsqrt():
  inp1 = Input(shape=(32,), dtype="float32")
  inp2 = Input(shape=(20,), dtype="float32")

  x = Dense(20, activation='relu')(inp1)
  out = rsqrt(x + inp2)

  model = flexflow.keras.models.Model([inp1, inp2], out)

  opt = flexflow.keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
  print(model.summary())
  model.fit(
    x = [np.random.randn(300, 32).astype(np.float32),
         np.ones((300, 20)).astype(np.float32)],
    y = np.random.randn(300, 20).astype(np.float32),
    epochs = 2
  )


if __name__ == "__main__":
    configs = ff.get_configs()
    ff.init_flexflow_runtime(configs)
    test_rsqrt()
