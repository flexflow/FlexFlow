from flexflow.core import *

from argparse import ArgumentParser

import sys
import numpy as np

def parse_args():
    print(sys.argv)
    parser = ArgumentParser()

# BERT-large
    parser.add_argument('--seq-length', default=512, type=int)
    parser.add_argument('--num-heads', default=16, type=int)
    parser.add_argument('--hidden-size', default=1024, type=int)
    parser.add_argument('--num_layers', default=24, type=int)
    parser.add_argument('--iterations', default=10, type=int)

    args, unknown = parser.parse_known_args()

    return args

def mha(model, q, k, v, batch_size, seq_length, hidden_size, n_heads, kdim, vdim, act=ActiMode.AC_MODE_GELU):
    q = model.dense(q, hidden_size)
    k = model.dense(k, hidden_size)
    v = model.dense(v, hidden_size)

    q = model.reshape(q, shape=(batch_size, seq_length, n_heads, kdim))
    k = model.reshape(k, shape=(batch_size, seq_length, n_heads, kdim))
    v = model.reshape(v, shape=(batch_size, seq_length, n_heads, vdim))
    q = model.transpose(q, perm=(0, 2, 1, 3))
    k = model.transpose(k, perm=(0, 2, 3, 1))
    v = model.transpose(v, perm=(0, 2, 1, 3))
    logits = model.batch_matmul(q, k, a_seq_length_dim=2,b_seq_length_dim=3)
    #logits = model.softmax(logits)
    output = model.batch_matmul(logits, v, a_seq_length_dim=3,b_seq_length_dim=2)
    output = model.transpose(output, perm=(0, 2, 1, 3))
    output = model.reshape(output, shape=(batch_size, seq_length, hidden_size))
    output = model.dense(output, hidden_size, act)
#    output = model.dense(output, hidden_size)
    return output

def create_bert_layer(model, input, batch_size, seq_length, hidden_size, n_heads, kdim, vdim, act=ActiMode.AC_MODE_GELU):
    t = input
    # MHA
#    t = model.multihead_attention(
#            t, t, t,
#            hidden_size, n_heads, kdim, vdim)
    t = mha(model, t, t, t, batch_size, seq_length, hidden_size, n_heads, kdim, vdim)
    # t = model.dense(input, hidden_size, act)
    t = model.dense(t, hidden_size, act)
    # t = model.dropout(t)
    t = model.add(t, input)

    # Intermediate
    intermediate_out = model.dense(t, hidden_size, act)

    # Output
    t = model.dense(intermediate_out, hidden_size, act)
    # t = model.dropout(t)
    t = model.add(t, intermediate_out)

    return t

def top_level_task():
    args = parse_args()

    ffconfig = FFConfig()
    netconfig = NetConfig()

    print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)"
          %(ffconfig.batch_size, ffconfig.workers_per_node, ffconfig.num_nodes))

    ffmodel = FFModel(ffconfig)

    batch_size = ffconfig.batch_size
    seq_length = args.seq_length
    hidden_size = args.hidden_size
    num_heads = args.num_heads
    num_layers = args.num_layers
    kdim = hidden_size // num_heads
    vdim = hidden_size // num_heads

    print('Model config:')
    print(f"  seq_length: {seq_length}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  kdim: {kdim}")
    print(f"  vdim: {vdim}")

    dims_input = [batch_size, seq_length, hidden_size]
    input_tensor = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)
    np_input_tensor = np.zeros(dims_input, dtype=np.float32)
    input_tensor.set_tensor(ffmodel, np_input_tensor)
    #input_tensor.attach_numpy_array(ffconfig, np_input_tensor)
    #input_tensor.detach_numpy_array(ffconfig)

    # build the model
    t = input_tensor
    for _ in range(num_layers):
        t = create_bert_layer(ffmodel, t, batch_size, seq_length, hidden_size, num_heads, kdim, vdim)

    # t now contains entire model. Add single-neuron output
    t = ffmodel.dense(t, 1)

    optimizer = SGDOptimizer(ffmodel, 1e-3)
    ffmodel.optimizer = optimizer
    ffmodel.compile(loss_type=LossType.LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE, metrics=[MetricsType.METRICS_ACCURACY], comp_mode=CompMode.INFERENCE)
    ffmodel.init_layers()
    ts_start = ffconfig.get_current_time()

    iterations = args.iterations
    for it in range(iterations):
#        print(f" ITERATION: {it}")
        ffconfig.begin_trace(111)
        ffmodel.forward(seq_length=it)
        ffconfig.end_trace(111)
    ts_end = ffconfig.get_current_time()
    print(f" Time taken to run forward pass: {(ts_end - ts_start)/iterations}")

if __name__ == "__main__":
    print("BERT Proxy")
    top_level_task()
