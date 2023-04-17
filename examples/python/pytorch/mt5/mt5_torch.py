"""
Based on:
https://towardsdatascience.com/how-to-train-an-mt5-model-for-translation-with-simple-transformers-30ba5fa66c5f
https://shivanandroy.com/fine-tune-t5-transformer-with-pytorch/
"""

import os

import numpy as np
#import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import MT5ForConditionalGeneration, T5Tokenizer

BASE_DIR = "examples/python/pytorch/mt5"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


class DataPreparer():
    """
    This class prepares the data -- :meth:`data_to_tsv` should only be called
    once, and the data can be directly loaded from the .tsv files thereafter.
    """
    @staticmethod
    def prepare_data(data_path):
        """
        Returns: train_df, eval_df
            train_df (pd.DataFrame): Training dataframe.
            eval_df (pd.DataFrame): Evaluation dataframe.
        """
        sinhala_train_filename = os.path.join(data_path, "train.trg")
        with open(sinhala_train_filename, "r", encoding="utf-8")as f:
            sinhala_text = f.readlines()
            sinhala_text = [text.strip("\n") for text in sinhala_text]
        english_train_filename = os.path.join(data_path, "train.src")
        with open(english_train_filename, "r") as f:
            english_text = f.readlines()
            english_text = [text.strip("\n") for text in english_text]

        data = []
        for sinhala, english in zip(sinhala_text, english_text):
            data.append(["translate sinhala to english", sinhala, english])
            data.append(["translate english to sinhala", english, sinhala])
        train_df = pd.DataFrame(
            data, columns=["prefix", "input_text", "target_text"]
        )

        sinhala_test_filename = os.path.join(data_path, "test.trg")
        with open(sinhala_test_filename, "r", encoding="utf-8") as f:
            sinhala_text = f.readlines()
            sinhala_text = [text.strip("\n") for text in sinhala_text]
        english_test_filename = os.path.join(data_path, "test.src")
        with open(english_test_filename, "r") as f:
            english_text = f.readlines()
            english_text = [text.strip("\n") for text in english_text]

        data = []
        for sinhala, english in zip(sinhala_text, english_text):
            data.append(["translate sinhala to english", sinhala, english])
            data.append(["translate english to sinhala", english, sinhala])
        eval_df = pd.DataFrame(
            data, columns=["prefix", "input_text", "target_text"]
        )

        return train_df, eval_df

    @staticmethod
    def data_to_tsv():
        """Saves the training data and evaluation data to .tsv files."""
        train_df, eval_df = DataPreparer.prepare_data(
            os.path.join(DATA_DIR, "eng-sin")
        )
        train_df.to_csv(os.path.join(DATA_DIR, "train.tsv"), sep="\t")
        eval_df.to_csv(os.path.join(DATA_DIR, "eval.tsv"), sep="\t")


class SinhaleseDataset(Dataset):
    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text,
        target_text,
    ):
        """
        Args:
            dataframe (pd.DataFrame): Input dataframe.
            tokenizer (transformers.tokenizer): Transformers tokenizer.
            source_len (int): Max length of source text.
            target_len (int): Max length of target text.
            source_text (str): Column name of source text.
            target_text (str): Column name of target text.
        """
        self.df = dataframe
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.target_len = target_len
        self.source_text = self.df[source_text]
        self.target_text = self.df[target_text]

    def __len__(self):
        """Returns the length of the dataframe."""
        return len(self.target_text)

    def __getitem__(self, index):
        """Returns the input IDs, target IDs, and attention masks for the
        given index."""
        src_text = str(self.source_text[index])
        tar_text = str(self.target_text[index])
        src_text = " ".join(src_text.split())
        tar_text = " ".join(tar_text.split())

        src = self.tokenizer.batch_encode_plus(
            [src_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tar = self.tokenizer.batch_encode_plus(
            [tar_text],
            max_length=self.target_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = src["input_ids"].squeeze()
        source_mask = src["attention_mask"].squeeze()
        target_ids = tar["input_ids"].squeeze()
        target_mask = tar["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for i, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]
        if i % 10 == 0:
            print(f"Epoch={epoch}\tbatch={i} \tloss={loss:.3f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            y = data["target_ids"].to(device, dtype=torch.long)
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            preds = [
                tokenizer.decode(
                    g, skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ) for g in generated_ids
            ]
            target = [
                tokenizer.decode(
                    t, skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                for t in y
            ]
            if i % 10 == 0:
                print(f"Epoch={epoch}\tbatch={i}")
            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


def get_dataframes():
    train_df = pd.read_csv(
        os.path.join(DATA_DIR, "train.tsv"), sep="\t",
    ).astype(str)
    eval_df = pd.read_csv(
        os.path.join(DATA_DIR, "eval.tsv"), sep="\t",
    ).astype(str)
    train_df["prefix"] = ""
    eval_df["prefix"] = ""
    return train_df, eval_df


def set_seed(model_params):
    torch.manual_seed(model_params["SEED"])
    np.random.seed(model_params["SEED"])
    torch.backends.cudnn.deterministic = True


def get_datasets(tokenizer, model_params):
    train_df, eval_df = get_dataframes()
    source_text = "input_text"
    target_text = "target_text"
    train_set = SinhaleseDataset(
        train_df,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    eval_set = SinhaleseDataset(
        eval_df,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    return train_set, eval_set


def get_dataloaders(tokenizer, model_params):
    train_set, eval_set = get_datasets(tokenizer, model_params)
    # Use the full dataset as one batch if the given batch size is `None`
    train_batch_size = model_params["TRAIN_BATCH_SIZE"]
    if train_batch_size is None:
        train_batch_size = len(train_set)
    eval_batch_size = model_params["EVAL_BATCH_SIZE"]
    if eval_batch_size is None:
        eval_batch_size = len(eval_set)
    train_params = {
        "batch_size": train_batch_size,
        "shuffle": True,
        "num_workers": 0,
    }
    eval_params = {
        "batch_size": eval_batch_size,
        "shuffle": False,
        "num_workers": 0,
    }
    train_loader = DataLoader(train_set, **train_params)
    eval_loader = DataLoader(eval_set, **eval_params)
    return train_loader, eval_loader


def TorchMT5Trainer(
    model_params,
    device,
    output_dir=OUTPUT_DIR,
):
    set_seed(model_params)

    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    model = MT5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    print("Reading data...")
    train_loader, eval_loader = get_dataloaders(tokenizer, model_params)
    optimizer = torch.optim.SGD(
        params=model.parameters(), lr=model_params["LEARNING_RATE"],
    )

    print("Training...")
    for epoch in range(1, model_params["TRAIN_EPOCHS"] + 1):
        train(epoch, tokenizer, model, device, train_loader, optimizer)

    print("Evaluating...")
    predictions, actuals = eval(0, tokenizer, model, device, eval_loader)
    output_df = pd.DataFrame({"Predictions": predictions, "Actuals": actuals})
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_df.to_csv(output_dir)


if __name__ == "__main__":
    if not os.path.exists(os.path.join(DATA_DIR, "train.tsv")) or \
            not os.path.exists(os.path.join(DATA_DIR, "eval.tsv")):
        DataPreparer.data_to_tsv()

    model_params = {
        "SEED": 42,
        "MODEL": "google/mt5-small",
        "TRAIN_BATCH_SIZE": 32,
        "EVAL_BATCH_SIZE": 32,
        "TRAIN_EPOCHS": 2,
        "MAX_SOURCE_TEXT_LENGTH": 48,
        "MAX_TARGET_TEXT_LENGTH": 48,
        "LEARNING_RATE": 1e-4,
    }
    device = torch.device('cpu')
    TorchMT5Trainer(model_params, device)
