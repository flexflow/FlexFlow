from flexflow.core import *
from argparse import ArgumentParser

def parse_args():
  parser = ArgumentParser()
  parser.add_argument('--seq-length', default=256, type=int)
  parser.add_argument('--num-heads', default=16, type=int)
  parser.add_argument('--hidden-size', default=512, type=int)
  args, unknown = parser.parse_known_args()
  return args

def attention():
  args = parse_args()
  ffconfig = FFConfig()
  print("Python API: batch_size(%d) GPUs/node(%d) nodes(%d)" %(ffconfig.batch_size, ffconfig.workers_per_node, ffconfig.num_nodes))
  ffmodel = FFModel(ffconfig)
  batch_size = ffconfig.batch_size
  dims_input = [batch_size, args.seq_length, args.hidden_size]
  input = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)
  q = ffmodel.dense(input, args.hidden_size)
  k = ffmodel.dense(input, args.hidden_size)
  v = ffmodel.dense(input, args.hidden_size)
  
  q = ffmodel.reshape(q, shape=(batch_size, args.seq_length, args.num_heads, args.hidden_size // args.num_heads))
  k = ffmodel.reshape(k, shape=(batch_size, args.seq_length, args.num_heads, args.hidden_size // args.num_heads))
  v = ffmodel.reshape(v, shape=(batch_size, args.seq_length, args.num_heads, args.hidden_size // args.num_heads))
  q = ffmodel.transpose(q, perm=(0, 2, 1, 3))
  k = ffmodel.transpose(k, perm=(0, 2, 3, 1))
  v = ffmodel.transpose(v, perm=(0, 2, 1, 3))
  logits = ffmodel.batch_matmul(q, k)
  #logits = ffmodel.softmax(logits)
  output = ffmodel.batch_matmul(logits, v)
  output = ffmodel.transpose(output, perm=(0, 2, 1, 3))
  output = ffmodel.reshape(output, shape=(batch_size, args.seq_length, args.hidden_size))
  output = ffmodel.dense(output, args.hidden_size, ActiMode.AC_MODE_RELU)
  output = ffmodel.dense(output, args.hidden_size)
  ffoptimizer = SGDOptimizer(ffmodel)
  ffmodel.set_sgd_optimizer(ffoptimizer)
  ffmodel.compile(loss_type=LossType.LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE, metrics=[MetricsType.METRICS_MEAN_SQUARED_ERROR], comp_mode=CompMode.INFERENCE)
  label_tensor = ffmodel.get_label_tensor()

  # Full inputs/label
  dims = [batch_size * 10, args.seq_length, args.hidden_size]
  np_input = np.zeros(dims, dtype=np.float32)
  np_label = np.zeros(dims, dtype=np.float32)
  full_input = ffmodel.create_tensor(dims, DataType.DT_FLOAT)
  full_label = ffmodel.create_tensor(dims, DataType.DT_FLOAT)
  full_input.attach_numpy_array(ffconfig, np_input)
  full_label.attach_numpy_array(ffconfig, np_label)

  dl_input = SingleDataLoader(ffmodel, input, full_input, batch_size * 10, DataType.DT_FLOAT)
  dl_label = SingleDataLoader(ffmodel, label_tensor, full_label, batch_size * 10, DataType.DT_FLOAT)
  
  ffmodel.init_layers()
  epochs = ffconfig.epochs

  dl_input.next_batch(ffmodel)
  dl_label.next_batch(ffmodel)

  ts_start = ffconfig.get_current_time()
  for epoch in range(0, epochs):
    ffmodel.reset_metrics()
    iterations = num_samples // batch_size
    for iter in range(0, iterations):
      ffconfig.begin_trace(111)
      ffmodel.forward()
      ffmodel.zero_gradients()
      ffmodel.backward()
      ffmodel.update()
      ffconfig.end_trace(111)
  ts_end = ffconfig.get_current_time()
  run_time = 1e-6 * (ts_end - ts_start)
  print("EPOCHS %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, num_samples * epochs / run_time));

if __name__ == "__main__":
  print("Attention")
  attention()
