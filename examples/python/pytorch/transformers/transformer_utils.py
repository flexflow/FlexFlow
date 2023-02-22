import os

import numpy as np
import torch
import argparse
import random
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, 'distilgpt2'),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, 'openaipgt'),
    "bert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer, 'distilbert-base-uncased'),
}

BATCH_SIZE = 32
MAX_LENGTH = 0
class TransformerArgs():
    """ parse argumments
    """

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-m", "--model", dest="model",
                                 required=True, metavar="", help="select pre-trained Transformer model")
        self.parser.add_argument("-t", "--task", dest="task",
                                 required=True, metavar="", help="task classification/generation")
        self.parser.add_argument("-td", "--tfdataset", dest="tfdataset", required=False,
                                 metavar="", help="transformer dataset for training/validation/testing")
        self.parser.add_argument("-rd", "--rawdataset", dest="rawdataset", required=False,
                                 metavar="", help="raw text dataset for training/validation/testing")

    def parse_args(self):
        args, unknown = self.parser.parse_known_args()
        config, model, tokenizer, pre_trained_name = MODEL_CLASSES[args.model]
        assert (args.tfdataset != None | args.rawdataset != None)
        dataset = args.tfdataset if args.tfdataset != None else args.rawdataset
        return dataset, config, model, tokenizer, pre_trained_name


def set_seed(num):
    torch.manual_seed(num)
    np.random.seed(num)
    torch.backends.cudnn.deterministic = True


def get_transformer_dataloaders(tokenizer, max_input_length):
    dataset = load_dataset('imdb')
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    encode_train_dataset = train_dataset.map(lambda examples: tokenizer(
        examples['text'], truncation=True, padding=True, max_length=max_input_length), batched=True)
    encode_test_dataset = test_dataset.map(lambda examples: tokenizer(
        examples['text'], truncation=True, padding=True, max_length=max_input_length), batched=True)

    # generate torch dataloader
    encode_train_dataset.set_format(type='torch', columns=[
                                    'input_ids', 'attention_mask', 'label'])
    encode_test_dataset.set_format(type='torch', columns=[
                                   'input_ids', 'attention_mask', 'label'])

    train_loader = torch.utils.data.DataLoader(
        encode_train_dataset, batch_size=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        encode_test_dataset, batch_size=4, shuffle=True)

    return train_loader, test_loader


class NSPDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)
    
# def group_texts(examples):
#     # Concatenate all texts.
#     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#         # customize this part to your needs.
#     total_length = (total_length // block_size) * block_size
#     # Split by chunks of max_len.
#     result = {
#         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
#         for k, t in concatenated_examples.items()
#     }
#     result["labels"] = result["input_ids"].copy()
#     return result

def process_nsp_dataset(dataset, tokenizer, max_input_length, column_name):
    splited_dataset = dataset[column_name]
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    next_sentence = [
        item for text in splited_dataset['text'] for item in text.split('.') if item != '']

    # generate next sentence A&B
    sentence_a = []
    sentence_b = []
    label = []

    for text in splited_dataset['text']:
        sentences = [
            sentence for sentence in text.split('.') if sentence != ''
        ]
        #
        if (len(sentences) > 1):
            start = random.randint(0, len(sentences) - 2)
            if random.random() >= 0.5:
                sentence_a.append(sentences[start])
                sentence_b.append(sentences[start + 1])
                label.append(0)
            else:
                random_index = random.randint(0, len(next_sentence) - 1)
                sentence_a.append(sentences[start])
                sentence_b.append(next_sentence[random_index])
                label.append(1)

    encode_input = tokenizer(sentence_a, sentence_b, return_tensors='pt',
                             max_length=max_input_length, truncation=True, padding='max_length')

    encode_input['labels'] = torch.LongTensor([label]).T
    
    encode_dataset = NSPDataset(encode_input)

    return torch.utils.data.DataLoader(
        encode_dataset, batch_size=4, shuffle=True)


def get_nsp_data_loader(tokenizer, max_input_length):
    MAX_LENGTH = max_input_length
    dataset = load_dataset('imdb')
    return process_nsp_dataset(dataset, tokenizer, max_input_length, 'train'), process_nsp_dataset(dataset, tokenizer, max_input_length, 'test')




def cal_training_time():
    return


def cal_inference_time():
    return


def precision_diff():
    return
