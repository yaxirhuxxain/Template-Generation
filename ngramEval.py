# -*- coding: utf-8 -*-

import json

# *** Basic Imports *** #
import os

import numpy as np
from datasets import load_dataset
from models.ngram import test_ngram


# *** Basic Configurations *** #

# Set random values
seed_val = 786
np.random.seed(seed_val)


# *** Main Configurations *** #
class Trainer(object):

    def __init__(
            self,
            in_folder,
            out_folder,
            context_size,
            smoothing,
            n_batch,
            n_workers
    ):

        self.in_folder = in_folder
        self.context_size = context_size
        self.learner = f"{context_size}_gram"
        self.smoothing = smoothing
        self.n_batch = n_batch
        self.word_idx = json.loads(open("./word_idx.json").read())
        self.vocab_size = len(self.word_idx)
        self.train_time = None
        self.train_rows = None
        self.valid_rows = None
        self.test_rows = None
        self.n_workers = n_workers  # SpeedUP: parallel processing
        self.out_folder = out_folder
        self.out_dir = None  # it will be set by each fold

    
    def encoder(self, item):
        return [word if word in self.word_idx else '<idf>' for word in item.split()]

    def prepare_datasets(self, datasets, max_samples=None):

        if max_samples is not None:
            datasets["test"] = datasets["test"].select(range(max_samples))

        # Main data processing function that will replace unk tokens
        def group_texts(examples):
            # we pad each example by context_size-1 to help learn context from index 0
            PAD = ['<pad>'] * (self.context_size - 1)

            # pad and replacing unk tokens (Note it will not encode into indexes)
            examples = {"text": [PAD + self.encoder(item) for item in examples["text"]]}

            return examples

        tokenized_datasets = datasets.map(
            group_texts,
            batched=True,
            num_proc=self.n_workers,
            load_from_cache_file=True,
        )

        return tokenized_datasets["test"]

    def eval_model(self, key, data_files, model):

        print("Fold: {} Model: {} > Folder:{}".format(key, self.learner, self.in_folder))
        self.out_dir = os.path.join(self.out_folder, os.path.join(f"Fold{key}",
                                                                    f"{self.learner}_smoothing_{str(self.smoothing)}"))
        print(f"Model Folder: {self.out_dir}")

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        print('*** Preparing data ***')

        cache_dir = os.path.join(self.out_folder, f"Fold{key}Cache")
        datasets = load_dataset('text', data_files=data_files, cache_dir=cache_dir)

        print('*** Preparing Data Loader ***')
        test_dataset = self.prepare_datasets(datasets, max_samples=None)

        print("*** Model testing ***")
        test_scores = test_ngram(model, test_dataset, n_gram=self.context_size, word_idx=self.word_idx)

        return test_scores

