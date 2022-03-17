# -*- coding: utf-8 -*-

from utils import BeamSearchNode
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
import numpy as np
from queue import PriorityQueue
import os
import operator
import json
import warnings



# *** Basic Configurations *** #

warnings.filterwarnings("ignore")

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
        self.idx_word = self.get_idx_word()
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

    def get_idx_word(self):
        return {v: k for k, v in self.word_idx.items()}

    def encoder(self, item):
        return [self.word_idx[word] if word in self.word_idx else self.word_idx['<idf>'] for word in item.split()]

    def prepare_datasets(self, datasets, max_samples=None):

        if max_samples is not None:
            datasets["test"] = datasets["test"].select(range(max_samples))

        def tokenize_function(examples):
            encoded_examples = [self.encoder(item)
                                for item in examples["text"]]
            return {"encoded_examples": encoded_examples}

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            batch_size=self.n_batch,
            num_proc=self.n_workers,
            remove_columns=["text"],
            load_from_cache_file=True,
        )

        return tokenized_datasets["test"]

    def test_model(self, key, data_files, model, mode='greedy', ctx=1, beam_k=1, top_k_pred=1):

        print("Fold: {} Model: {} > Folder:{}".format(
            key, self.learner, self.in_folder))
        self.out_dir = os.path.join(
            self.out_folder, os.path.join(f"Fold{key}", f"{self.learner}"))
        print(f"Model Folder: {self.out_dir}")

        if not os.path.exists(self.out_dir):
            raise "Invalid Location"

        print('*** Preparing data ***')

        cache_dir = os.path.join(self.out_folder, f"Fold{key}Cache")
        datasets = load_dataset(
            'text', data_files=data_files, cache_dir=cache_dir)

        print('\t > Preparing Data Loader ')
        test_dataset = self.prepare_datasets(datasets, max_samples=None)
        self.test_rows = len(test_dataset)
        print(f"\t > Test Examples: {self.test_rows}")

        # DataLoader will only pass one batch size of examples to collector function
        def collector(example):
            return [item['encoded_examples'] for item in example]

        test_loader = DataLoader(
            test_dataset,
            collate_fn=collector,
            batch_size=self.n_batch,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory
        )

        print("*** Model testing ***")

        print(f"\t> Fold: {key}")
        print(f"\t> Learner: {self.learner}")
        print(f"\t> Mode: {mode}")
        print(f"\t> ctx: {ctx}")
        print(f"\t> beam_k: {beam_k}")
        print(f"\t> top_k_pred: {top_k_pred}")
        predictions = self.template_generator(model, test_loader, mode=mode, ctx=ctx, beam_k=beam_k,
                                              top_k_pred=top_k_pred)

        return predictions

    def template_generator(self, model, test_loader, mode='beam', ctx=1, beam_k=1, top_k_pred=1):

        model.eval()
        X_pred = []
        X_true = []

        with torch.no_grad():
            for batch in tqdm(test_loader):

                for row in batch:
                    X_true.append([self.idx_word[idx]
                                  for idx in row])  # True sequence
                    # initialize with seed tokens
                    X_pred.append([self.idx_word[idx] for idx in row[0:ctx]])

                # Finding padding length and preparing seed tokens
                bs = len(batch)
                if ctx > 2:
                    # to ensure constant length for rows (bs, ctx)
                    import tensorflow as tf
                    init_tokens = torch.as_tensor(
                        tf.keras.preprocessing.sequence.pad_sequences(
                            [item[0:ctx] for item in batch],
                            maxlen=ctx, dtype='int32', padding='pre', truncating='pre', value=self.word_idx['<pad>']
                        )
                    )
                else:
                    # getting init words for seed
                    init_tokens = torch.as_tensor(
                        [item[0:ctx] for item in batch])
                pad_len = self.context_size - init_tokens.size(1)  # pad length
                padding_mtx = torch.ones(
                    (bs, pad_len)) * self.word_idx['<pad>']  # padding matrix
                seed = torch.cat((padding_mtx, init_tokens), 1).type(
                    torch.long)  # seed matrix padded

                if mode == 'greedy':  # apply greedy approach

                    batch_pred = None
                    while True:
                        logits = model(seed)
                        sampled_token_idx = logits.topk(k=1, dim=1).indices
                        # adding pred and sliding context
                        seed = torch.cat((seed[:, 1:], sampled_token_idx), 1)

                        if batch_pred is None:
                            batch_pred = sampled_token_idx
                        else:
                            batch_pred = torch.cat(
                                (batch_pred, sampled_token_idx), 1)

                        if batch_pred.size(1) > self.context_size:
                            break

                    break_list = [self.word_idx[";"],
                                  self.word_idx["{"], self.word_idx["}"]]
                    for idx, row in enumerate(batch_pred.detach().numpy()):
                        row_pred = []
                        for i in row:
                            row_pred.append(self.idx_word[i])
                            if i in break_list:
                                break
                        X_pred[idx].extend(row_pred)

                elif mode == 'beam':
                    if beam_k < 2 and top_k_pred > beam_k:
                        raise f"beam size cannot be less than predictions size beam_k={beam_k} top_k_pred={top_k_pred}"

                    # decoding goes sentence by sentence
                    for idx in range(seed.size(0)):

                        # Saves the generated sentences
                        endNodes = []

                        # init node - seed, predToken, prevNode, logProb, length
                        node = BeamSearchNode(seed[idx], None, None, 0, 0)
                        nodes = PriorityQueue()

                        # start the queue
                        nodes.put((-node.eval(), node))
                        qsize = 1

                        # start beam search
                        while True:
                            # give up when decoding takes too long
                            if qsize > 1000:
                                break

                            # fetch the best node
                            score, n = nodes.get()

                            if n.pred in [self.word_idx[';'], self.word_idx['{'],
                                          self.word_idx['}']] and n.prevNode is not None:
                                endNodes.append((score, n))
                                # if we reached maximum # of sentences required
                                if len(endNodes) >= top_k_pred:
                                    break
                                else:
                                    continue

                            # decode for one step using decoder
                            logits = model(n.seed.unsqueeze(dim=0))

                            # PUT HERE REAL BEAM SEARCH OF TOP
                            log_prob, indexes = torch.topk(logits, beam_k)

                            nextNodes = []
                            for i in range(beam_k):
                                decoded_t = indexes[0][i].view(-1)
                                log_p = log_prob[0][i].item()

                                node = BeamSearchNode(
                                    torch.cat((n.seed, decoded_t)
                                              )[-self.context_size:],
                                    decoded_t.item(),
                                    n,
                                    n.logp + log_p,
                                    n.len + 1
                                )
                                score = -node.eval()
                                nextNodes.append((score, node))

                            # put them into queue
                            for i in range(len(nextNodes)):
                                score, nn = nextNodes[i]
                                nodes.put((score, nn))

                            # increase qsize
                            qsize += beam_k

                        # choose n-best paths for back tracing
                        if len(endNodes) == 0:
                            endNodes = [nodes.get() for _ in range(top_k_pred)]

                        utterances = []
                        for score, n in sorted(endNodes, key=operator.itemgetter(0)):
                            utterance = [n.pred]

                            # back trace
                            while n.prevNode is not None:
                                n = n.prevNode
                                utterance.append(n.pred)

                            utterance = utterance[::-1]
                            utterance = list(filter(None, utterance))

                            utterances.append(
                                X_pred[idx] + [self.idx_word[ut] for ut in utterance])

                        X_pred[idx] = utterances

                else:
                    raise "invalid evaluation mode"

        return X_true, X_pred
