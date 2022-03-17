# -*- coding: utf-8 -*-


# *** Basic Imports *** #
import os
import json

import numpy as np

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import tqdm

# Custom Imports
from models.metrics import test_metrics, stack_scores

# *** Basic Configurations *** #

# Set random values
seed_val = 786
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_val)

torch.backends.cudnn.benchmark = True  # SpeedUP: Turn on cudNN benchmarking


# *** Main Trainer Configurations *** #
class Trainer(object):

    def __init__(
            self,
            language,
            in_folder,
            context_size,
            n_dim,
            n_epochs,
            warmup_epochs,
            val_interval,
            n_batch,
            n_drop,
            learner,
            n_layers,
            optimizer,
            learn_rate,
            loss,
            patience,
            out_folder,
            n_workers,
            pin_memory,
            dev_run
    ):

        self.language = language

        self.in_folder = in_folder
        self.context_size = context_size
        self.n_dim = n_dim
        self.n_epochs = n_epochs
        self.warmup_epochs = warmup_epochs  # minimum number of epochs to run
        self.init_epoch = 0  # initially is 1
        self.val_interval = val_interval  # how many times to evaluate model in 1 epoch
        self.n_batch = n_batch
        self.n_drop = n_drop
        self.word_idx = json.loads(open("./word_idx.json").read())
        self.vocab_size = len(self.word_idx)
        self.learn_rate = learn_rate
        self.learner = learner
        self.n_layers = n_layers
        self.optimizer = optimizer
        self.loss = loss
        self.patience = patience
        self.train_time = None
        self.train_rows = None
        self.valid_rows = None
        self.test_rows = None
        self.n_workers = n_workers  # SpeedUP: parallel processing
        self.pin_memory = pin_memory  # SpeedUP: Pin memory for fast data loading
        self.dev_run = dev_run
        self.out_folder = out_folder
        self.out_dir = None  # it will be set by each fold

    def encoder(self, item):
        return [self.word_idx[word] if word in self.word_idx else self.word_idx['<idf>'] for word in item.split()]

    def prepare_datasets(self, datasets, max_samples=None):

        if max_samples is not None:
            datasets["test"] = datasets["test"].select(range(max_samples))

        def tokenize_function(examples):
            # we pad each example by context_size-1 to help learn context from index 0
            PAD = [self.word_idx['<pad>']] * (self.context_size - 1)
            encoded_examples = [
                PAD + self.encoder(item) for item in examples["text"]]
            return {"encoded_examples": encoded_examples}

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            batch_size=self.n_batch,
            num_proc=self.n_workers,
            remove_columns=["text"],
            load_from_cache_file=True,
        )

        # Main data processing function that will concatenate all texts from our dataset and
        # generate sequences with sliding window with max_seq_length.
        def group_texts(examples):
            context_size_with_pred = self.context_size + 1
            # Concatenate all texts.
            concatenated_examples = {
                k: sum(examples[k], []) for k in examples.keys()}

            result = {}
            for k, t in concatenated_examples.items():
                temp = [t[i:i + context_size_with_pred]
                        for i in range(0, len(t) - context_size_with_pred)]
                result[k] = temp

            return result

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=self.n_workers,
            load_from_cache_file=True,
        )

        return tokenized_datasets["test"]

    def test_model(self, key, data_files, model):

        print("Fold: {} Model: {} > Folder:{}".format(
            key, self.learner, self.in_folder))
        self.out_dir = os.path.join(self.out_folder, os.path.join(
            f"Fold{key}", f"{self.learner}"))
        print(f"Model Folder: {self.out_dir}")

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        cache_dir = os.path.join(self.out_folder, f"Fold{key}Cache")
        datasets = load_dataset(
            'text', data_files=data_files, cache_dir=cache_dir)

        print('*** Preparing Data Loader ***')
        test_dataset = self.prepare_datasets(datasets, max_samples=None)

        # DataLoader will only pass one batch size of examples to collector function
        def collector(example):
            seq = []
            pred = []
            for item in example:
                item = item['encoded_examples']
                seq.append(item[:-1])
                pred.append(item[-1])
            return torch.as_tensor(seq), torch.as_tensor(pred)

        test_loader = DataLoader(
            test_dataset,
            collate_fn=collector,
            batch_size=self.n_batch,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory
        )
        
        print("*** Model testing ***")

        model.eval()

        scores = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                labels = batch.pop("labels")
                logits = model(batch.pop("input_ids"))
                scores.append(test_metrics(labels.unsqueeze(dim=1), logits, self.vocab_size))

        test_scores = stack_scores(scores)
        return test_scores

        