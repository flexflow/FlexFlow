import datasets
from datasets import load_dataset
import pandas as pd
from math import ceil
from random import shuffle, uniform
import json

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
      # Import ShareGPT data
      datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True # Workaround for memory permissions (see https://github.com/huggingface/datasets/issues/1785)
      data_files = {"json": "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"}
      json_file = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", data_files=data_files)["json"]

      self.prompt_data = []
      for entry in json_file:
        conv = entry["conversations"]
        i = 0
        while i < len(conv):
          if conv[i]["from"] == "human" and i+1 < len(conv) and conv[i+1]["from"] == "gpt":
            prompt = conv[i]["value"]
            #generated = conv[i+1]["value"]
            if len(prompt) > 0: ################# TODO: Clean prompts (sequence length bounds, context, etc...)
              self.prompt_data.append(prompt)
            i += 2
          else:
            i += 1
      if shuffle_:
        shuffle(self.prompt_data)
      self.imported_prompt_data = True

  # Delta is in seconds
  # Rate is in req per second
  def generate_trace(self, target_arrival_rate=10):
    self.import_trace_timestamps()
    self.import_prompt_data()

    microsec = 1000000
    avg_arrival_rate = len(self.req_times) / (self.req_times[-1]/float(microsec)) # Request per second. Computed that way to enforce working with numbers of reasonable orders of magnitude
    scale_factor = float(target_arrival_rate) / avg_arrival_rate

    nb_buckets = ceil(self.req_times[-1] / (delta*microsec))
    buckets = []
    j = 0
    k = 0
    for i in range(nb_buckets):
      bucket_size = 0
      while(j < len(self.req_times) and self.req_times[j] >= delta*i*microsec and self.req_times[j] < delta*(i+1)*microsec):
        bucket_size += 1
        j += 1
      bucket_size = bucket_size*scale_factor
      prob = bucket_size - int(bucket_size)
      bucket_size = int(bucket_size) + int(uniform(0, 1) <= prob)
      
      # If used all of the prompt data, loop back at the beggining and reuse some prompts
      if k+bucket_size > len(self.prompt_data):
        bucket = self.prompt_data[k:] + self.prompt_data[:(k+bucket_size)%len(prompt_data)]
      else:
        bucket = self.prompt_data[k:k+bucket_size]
      k = (k+bucket_size) % len(self.prompt_data)
      buckets.append(bucket)
    return buckets

if __name__ == '__main__':
  builder = TraceBuilder()
  trace = builder.generate_trace(target_arrival_rate=10)
  # Save to a file
  with open('trace_data.json', 'w', encoding='utf-8') as f:
    json.dump(trace, f, ensure_ascii=False, indent=2)