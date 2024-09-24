import pandas as pd
from math import ceil
from random import shuffle, uniform
import json, pickle, requests, os, argparse

class TraceBuilder(object):

  # trace_type: either "conv" or "code"
  def __init__(self, import_times=True, import_prompts=True):
    self.req_times = None
    self.imported_req_times = False
    self.prompt_data = None
    self.imported_prompt_data = False
    if import_times:
      self.import_trace_timestamps()
    if import_prompts:
      self.import_prompt_data()

  def import_trace_timestamps(self, trace_type="conv"):
    if not self.imported_req_times:
      # Import Microsoft LLM 1 hour trace
      df_trace = pd.read_csv("https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_"+trace_type+".csv", parse_dates=["TIMESTAMP"])
      req_times = (pd.to_datetime(df_trace["TIMESTAMP"]).astype(int)//1000) # Timestamps are in microseconds
      req_times = req_times - req_times.min()
      self.req_times = req_times.tolist()
      self.imported_req_times = True
  
  def import_prompt_data(self, shuffle_=True):
    if not self.imported_prompt_data:
      sharegpt_filename = "sharegpt_opt_text_completion_length.pkl"
      sharegpt_filepath = f"./{sharegpt_filename}"
      if os.path.exists(sharegpt_filepath):
        os.remove("sharegpt_opt_text_completion_length.pkl")
      sharegpt_url = f"https://github.com/sosp-ae-39/sosp-ae-astra/raw/main/datasets/{sharegpt_filename}"
      response = requests.get(sharegpt_url)
      with open(sharegpt_filename, "wb") as file:
        file.write(response.content)
      with open(sharegpt_filepath, 'rb') as f:
        data2 = pickle.load(f)
      os.remove("sharegpt_opt_text_completion_length.pkl")

      prompt_lengths = [pair[0] for pair in data2 if pair[0] <= 2048 and pair[0] >= 4 and pair[1] >= 4 and pair[1] <= 2048 and pair[0]+pair[1] <= 2048]
      generation_lengths = [pair[1] for pair in data2 if pair[0] <= 2048 and pair[0] >= 4 and pair[1] >= 4 and pair[1] <= 2048 and pair[0]+pair[1] <= 2048]

      for pair in data2:
        assert(len(pair) == 2)

      prompt_lengths = [pair[0] for pair in data2 if pair[0] <= 2048 and pair[0] >= 4 and pair[1] >= 4 and pair[1] <= 2048 and pair[0]+pair[1] <= 2048]
      generation_lengths = [pair[1] for pair in data2 if pair[0] <= 2048 and pair[0] >= 4 and pair[1] >= 4 and pair[1] <= 2048 and pair[0]+pair[1] <= 2048]
      num_pairs = len(prompt_lengths)
      assert(num_pairs == len(generation_lengths))
      print("Number of conversation pairs: ", num_pairs)

      print(f"Prompt lengths: min={min(prompt_lengths)}, max={max(prompt_lengths)}, avg={sum(prompt_lengths)/len(prompt_lengths)}")
      print(f"Generation lengths: min={min(generation_lengths)}, max={max(generation_lengths)}, avg={sum(generation_lengths)/len(generation_lengths)}")
      total_lengths = [prompt_lengths[i] + generation_lengths[i] for i in range(len(prompt_lengths))]
      print(f"Total lengths: min={min(total_lengths)}, max={max(total_lengths)}, avg={sum(total_lengths)/len(total_lengths)}")

      self.prompt_data = [{"human": prompt_lengths[i], "gpt": generation_lengths[i]} for i in range(num_pairs)]
        
      if shuffle_:
        shuffle(self.prompt_data)
      self.imported_prompt_data = True

  # Delta is in seconds
  # Rate is in req per second
  def generate_trace(self, target_arrival_rate=10, debug_verbose=False):
    self.import_trace_timestamps()
    self.import_prompt_data()

    microsec = 1000000
    avg_arrival_rate = len(self.req_times) / (self.req_times[-1]/float(microsec)) # Request per second. Computed that way to enforce working with numbers of reasonable orders of magnitude
    if debug_verbose:
      print("Avg arrival rate of original trace (req/s): ", avg_arrival_rate)
    scale_factor = float(target_arrival_rate) / avg_arrival_rate
    if debug_verbose:
      print("Scale factor to obtain target arrival rate: ", scale_factor)

    # Buckets are 1 second timeframes
    nb_buckets = ceil(self.req_times[-1] / microsec)
    buckets = []
    j = 0
    k = 0
    for i in range(nb_buckets):
      bucket_size = 0
      while(j < len(self.req_times) and self.req_times[j] >= i*microsec and self.req_times[j] < (i+1)*microsec):
        bucket_size += 1
        j += 1
      bucket_size = bucket_size*scale_factor
      prob = bucket_size - int(bucket_size)
      bucket_size = int(bucket_size) + int(uniform(0, 1) <= prob)
      
      # If used all of the prompt data, loop back at the beggining and reuse some prompts
      if k+bucket_size > len(self.prompt_data):
        bucket = self.prompt_data[k:] + self.prompt_data[:(k+bucket_size)%len(self.prompt_data)]
      else:
        bucket = self.prompt_data[k:k+bucket_size]
      k = (k+bucket_size) % len(self.prompt_data)
      buckets.append(bucket)

    if debug_verbose:
      print("Avg arrival rate obtained (req/s): ", sum([len(b) for b in buckets])/len(buckets))
    return buckets

def generate_and_save_trace(arrival_rate, output_file):
  builder = TraceBuilder()
  trace = builder.generate_trace(target_arrival_rate=arrival_rate, debug_verbose=True)
  with open(output_file, 'w+') as f:
    json.dump(trace, f, indent=2)

if __name__ == '__main__':
  # Set up the argument parser
  parser = argparse.ArgumentParser(description='Generate and save a trace.')
  parser.add_argument('--arrival-rate', type=float, default=10.0, help='The target arrival rate for the trace.')
  parser.add_argument('--output-file', type=str, default='sharegpt.json', help='The path to the output file to save the trace.')

  # Parse the command-line arguments
  args = parser.parse_args()

  # Call the function with the user-provided arrival rate
  generate_and_save_trace(args.arrival_rate, args.output_file)
