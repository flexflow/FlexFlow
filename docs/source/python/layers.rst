**********
Layers API
**********

Layers are the basic building blocks of neural networks in FlexFlow. The inputs of a layer consists of a tensor or a list of tensors and some state variables,
and the outputs of a layer is a tensor or a list of tensors. See https://github.com/flexflow/FlexFlow/examples/python/native/ops for an example for every layer

.. automodule:: flexflow.core.flexflow_cffi
   :noindex:

Conv2D
======
.. autoclass:: FFModel()
   :noindex:
   :members: conv2d
   
Pool2D
======
.. autoclass:: FFModel()
   :noindex:
   :members: pool2d
   
Dense
======
.. autoclass:: FFModel()
   :noindex:
   :members: dense
   
Embedding
=========
.. autoclass:: FFModel()
   :noindex:
   :members: embedding
   
Transpose
=========
.. autoclass:: FFModel()
   :noindex:
   :members: transpose
   
Reverse
=======
.. autoclass:: FFModel()
   :noindex:
   :members: reverse
   
Concatenate
===========
.. autoclass:: FFModel()
   :noindex:
   :members: concat
   
Split
======
.. autoclass:: FFModel()
   :noindex:
   :members: split
   
Reshape
=======
.. autoclass:: FFModel()
   :noindex:
   :members: reshape

Flat
======
.. autoclass:: FFModel()
   :noindex:
   :members: flat
   
BatchNorm
=========
.. autoclass:: FFModel()
   :noindex:
   :members: batch_norm
   
BatchMatMul
===========
.. autoclass:: FFModel()
   :noindex:
   :members: batch_matmul
   
Add
======
.. autoclass:: FFModel()
   :noindex:
   :members: add
   
Subtract
========
.. autoclass:: FFModel()
   :noindex:
   :members: subtract
   
Multiply
========
.. autoclass:: FFModel()
   :noindex:
   :members: multiply
   
Divide
======
.. autoclass:: FFModel()
   :noindex:
   :members: divide
   
Exponential
===========
.. autoclass:: FFModel()
   :noindex:
   :members: exp
   
ReLU
====
.. autoclass:: FFModel()
   :noindex:
   :members: relu
   
ELU
====
.. autoclass:: FFModel()
   :noindex:
   :members: elu
   
Sigmoid
=======
.. autoclass:: FFModel()
   :noindex:
   :members: sigmoid
   
Tanh
====
.. autoclass:: FFModel()
   :noindex:
   :members: tanh
   
Softmax
=======
.. autoclass:: FFModel()
   :noindex:
   :members: softmax
   
Dropout
=======
.. autoclass:: FFModel()
   :noindex:
   :members: dropout
   
MultiheadAttention
==================
.. autoclass:: FFModel()
   :noindex:
   :members: multihead_attention