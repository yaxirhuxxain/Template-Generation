from collections import defaultdict

import nltk
from nltk.util import ngrams
from tqdm import tqdm
import numpy as np

import torch
from torchmetrics.functional.classification.f_beta import f1, fbeta
from torchmetrics.functional.classification.precision_recall import (precision,recall)

# for sorting the probable word acc. to their probabilities
# returns: void
# arg: dict
def sortProbWordDict(prob_dict):
    for key in prob_dict:
        if len(prob_dict[key]) > 1:
            # only at most top 2 most probable words have been taken
            prob_dict[key] = sorted(prob_dict[key], reverse=True)


# creates the dictionaries required for computing Interpolated Knesser Ney probability
# arg: dict, int
# returns: dict, dict
def createKNDict(ngram_dict, n):
    # for knesser ney probability formula we need to find to important things
    # first is for P(Wn|Wn-1) if find no. of ngrams which ends with Wn and no. of ngrams which starts
    # with Wn-1
    # so we divide the formula into two parts ,first part can be found in constant time
    # and second term is found here

    # for storing count of ngram ending with Wn,key:unigram
    first_dict = {}
    # for storing count of ngram having Wn-1 as its starting part, key: trigram sentence
    sec_dict = {}

    for key in ngram_dict:
        # split the key sentence into tokens
        ngram_token = key.split()
        # since the indexing is from 0 ,so for quadgram we need to create a sentence of three words
        # so start from 0 to 2,so we subtract 1,similarly for trigram from 0 to 1
        n_1gram_sen = ' '.join(ngram_token[: n - 1])

        # n_1gram_sen is the word that  stars in sec_dict[n_1gram_sen] number of times in ngram_dict
        if n_1gram_sen not in sec_dict:
            sec_dict[n_1gram_sen] = 1
        else:
            sec_dict[n_1gram_sen] += 1

        if ngram_token[-1] not in first_dict:
            first_dict[ngram_token[-1]] = 1
        else:
            first_dict[ngram_token[-1]] += 1

    return first_dict, sec_dict


# Finds the Knesser Ney probability for prediction
# arg: dict, dict
# return: void
def computeKnesserNeyProb(n_gram_dict, prob_dict):
    d = 0.75
    # first create the dict for storing the count of Wn-1 followed by Wn and for
    # ngrams preceding Wn-1
    n_gram_fs_dict = {}

    # n-gram starts from 2 so match the last ngram size it should be len(dic) + 1
    last_gram_dict = n_gram_dict[len(n_gram_dict) + 1]

    # building first and second KNdict
    for n_gram in n_gram_dict:
        # print(n_gram)
        first_dict = {}
        sec_dict = {}
        first_dict, sec_dict = createKNDict(n_gram_dict[n_gram], n_gram)
        n_gram_fs_dict[n_gram] = {1: first_dict, 2: sec_dict}

    # now find the probability for the sentences
    for n_gram in last_gram_dict:
        n_gram_token = n_gram.split()
        n_gram_sen = ' '.join(n_gram_token[:-1])
        n_gram_pred = n_gram_token[-1]
        prob = 0.0

        for fs_n_gram in n_gram_fs_dict:

            first_dict = {}
            sec_dict = {}
            first_dict = n_gram_fs_dict[fs_n_gram][1]
            sec_dict = n_gram_fs_dict[fs_n_gram][2]
            ngram_dict = n_gram_dict[fs_n_gram]

            # check if the current ngram is equal to max ngrams given
            # then calculate max gram probs else normal
            if len(n_gram_fs_dict) + 1 == fs_n_gram:
                ngram_dict_before = n_gram_dict[fs_n_gram - 1]
                prob1 = max(ngram_dict[n_gram] - d, 0) / ngram_dict_before[n_gram_sen]
                prob2 = d / ngram_dict_before[n_gram_sen] * (sec_dict[n_gram_sen])
            else:
                prob1 = max(first_dict[n_gram_pred] - d, 0) / len(ngram_dict)
                prob2 = (d / len(ngram_dict)) * (sec_dict[' '.join(n_gram_token[1:fs_n_gram])])

            if fs_n_gram == 2:
                uni_prob = first_dict[n_gram_pred] / len(ngram_dict)
                prob = prob1 + prob2 * (uni_prob)
            else:
                prob = prob1 + prob2 * (prob)

        if n_gram_sen not in prob_dict:
            prob_dict[n_gram_sen] = []
            prob_dict[n_gram_sen].append([prob, n_gram_pred])
        else:
            prob_dict[n_gram_sen].append([prob, n_gram_pred])


def doPrediction(sen, prob_dict):
    if sen in prob_dict:
        return prob_dict[sen]
    else:
        return None


def calculate_accuracy(X_test, y_test, model, k):
    k = max(k, 1)
    score = 0
    total = len(X_test)
    for prevWords, trueWord in zip(X_test, y_test):
        logits = doPrediction(prevWords, model)
        if logits:
            for count, value in enumerate(logits):
                if count >= k: break
                # prob = value[0]
                pred = value[1]
                if pred == trueWord:
                    score += 1
                    break

    return score / total


def train_ngram(train_dataset, n_gram=5, smoothing=True):
    # variable declaration
    prob_dict = defaultdict(list)  # for storing probability of probable words for prediction

    if smoothing:
        n_gram_dict = {}  # for keeping count of sentences (n-grams)
        for i in range(2, n_gram + 1):
            print(f"\t> Building {i}-grams")
            # creating ngram dict
            temp_dict = defaultdict(int)
            for item in tqdm(train_dataset):
                tokens_stream = item['text']
                for t in list(ngrams(tokens_stream, i)):
                    sen = ' '.join(t)
                    temp_dict[sen] += 1

            n_gram_dict[i] = temp_dict

        # Smoothing: compute the Knesser Ney probabilities
        computeKnesserNeyProb(n_gram_dict, prob_dict)

    else:
        max_gram_dict = {}
        count_dict = defaultdict(int)
        for item in tqdm(train_dataset):
            tokens_stream = item['text']
            for t in list(ngrams(tokens_stream, n_gram)):
                sen = ' '.join(t)
                count_dict[sen] += 1

                prev_words, target_word = t[:-1], t[-1]
                if prev_words in max_gram_dict:
                    max_gram_dict[prev_words].append(target_word)
                else:
                    max_gram_dict[prev_words] = [target_word]

        for token, count_of_token in count_dict.items():
            token = token.split()
            prev_words, target_word = token[:-1], token[-1]
            try:
                count_of_context = float(len(max_gram_dict[tuple(prev_words)]))
                prob = count_of_token / count_of_context
            except KeyError:
                prob = 0.0

            prev_words = " ".join(prev_words)
            if prev_words in prob_dict:
                prob_dict[prev_words].append([prob, target_word])
            else:
                prob_dict[prev_words] = [[prob, target_word]]

    # sort the probable words by their probability
    sortProbWordDict(prob_dict)

    return prob_dict


def calculate_PRF(X_test, y_test, model, word_idx):
    
    preds = []
    labels = []
    for prevWords, trueWord in zip(X_test, y_test):
        logits = doPrediction(prevWords, model)
        if logits:
            preds.append(word_idx[logits[0][1]])
        else:
            preds.append(0) # it will appedn <pad> token as pred which is near to imposible in test set
        labels.append(word_idx[trueWord])
    
    preds = torch.as_tensor(preds)
    labels = torch.as_tensor(labels)
    
    scores = {}
    scores['precision'] = precision(preds, labels, average='weighted', num_classes=502)
    scores['recall'] = recall(preds, labels, average='weighted', num_classes=502)
    scores['f1'] = f1(preds, labels, average='weighted', num_classes=502, beta=1)
    scores['fbeta'] = fbeta(preds, labels, average='weighted', num_classes=502, beta=0.5)

    return scores

def test_ngram(model, test_dataset, n_gram=5, word_idx=None):
    X_test = []
    y_test = []
    for item in tqdm(test_dataset):
        tokens_stream = item['text']
        for i in range(n_gram, len(tokens_stream)):
            seq = tokens_stream[i - n_gram: i]
            X_test.append(" ".join(seq[0:-1]))
            y_test.append(seq[-1])

    scores = calculate_PRF(X_test, y_test, model, word_idx)
    for i in [1, 2, 3, 4, 5, 10]:
        scores[f'acc_{i}'] = calculate_accuracy(X_test, y_test, model, i)

    return scores


def predict(seed, model, context_size):
    logits = model[" ".join(seed[-(context_size - 1):])]  # -(context_size-1) simple trick for sliding the context
    if not logits:
        return None, None
    # prob = logits[0][0]
    # pred = logits[0][1]
    return logits[0][0], logits[0][1]


def greedy_decoder(seed, model, context_size, MaxPred):
    answer = seed

    # In n-gram model the pred word is included in context so we pad the seed with context - 2
    PAD = ['<pad>'] * (context_size - 2)
    seed = PAD + [seed]

    while True:
        prob_pred = doPrediction(" ".join(seed[-(context_size - 1):]), model)
        if prob_pred is None:
            return answer

        prob, pred = prob_pred[0][0], prob_pred[0][1]  # greedily taking top 1 predictions
        answer += " " + pred
        seed.append(pred)

        if pred in [";", "{", "}"] or len(answer.split()) > MaxPred:
            break
    return answer


def beam_decoder(seed, model, context_size, MaxPred, k=5):
    answer = seed

    # In n-gram model the pred word is included in context so we pad the seed with context - 2
    PAD = ['<pad>'] * (context_size - 2)
    seed = PAD + [seed]

    answers = []
    prob_pred = doPrediction(" ".join(seed[-(context_size - 1):]), model)
    if prob_pred is None:
        return answer

    # selecting top k predictions
    for i in range(k):
        try:
            answers.append([(prob_pred[i][0], prob_pred[i][1])])  # prob, pred
        except:
            answers.append([(0.0, '<idf>')])  # prob, pred

    count = 0
    while True:
        count += 1
        for _ in range(k):
            seq = answers.pop(0)

            if seq[-1][1] in [";", "{", "}"]:
                answers.append(seq)
                continue

            target_seed = seed + [s[1] for s in seq]
            prob_pred = doPrediction(" ".join(target_seed[-(context_size - 1):]), model)
            if prob_pred is None:
                continue

            # selecting top k predictions
            for i in range(k):
                try:
                    answers.append(seq + [(prob_pred[i][0], prob_pred[i][1])])  # prob, pred
                except:
                    answers.append(seq + [(0.0, '<idf>')])  # prob, pred

        dead_list = [sum([np.log(x[0]) for x in seq]) for seq in answers]  # seq[:, 1]
        try:
            top_k_idx = np.argpartition(dead_list, -k)[-k:]
            answers = [answers[i] for i in top_k_idx]  # answers[top_k_idx]
        except:
            pass
        if all([s[-1][1] in [";", "{", "}"] or len(s) > MaxPred for s in answers]):
            break

    dead_list = [sum([np.log(x[0]) for x in seq]) for seq in answers]
    # TODO: Return k best
    best_answer = answers[np.argmax(dead_list)]
    for token in best_answer:
        answer += " " + token[1]
    return answer


def template_generator(model, test_dataset, context_size=5, method='greedy', topK=1):
    X_true = []
    X_pred = []

    for item in tqdm(test_dataset):
        tokens_stream = item['text']

        test_item = " ".join(tokens_stream)
        seed  = test_item[0]
        if method == "greedy":
            ts = greedy_decoder(seed, model, context_size, context_size + 10)
            pred_item = " ".join(seed+ts)
        elif method == "beam":
            ts = beam_decoder(seed, model, context_size, context_size + 10, k=topK)
            pred_item = " ".join(seed+ts)

        else:
            raise "invalid method please choose 'greedy' or 'beam'"
        
        X_true.append(test_item)
        X_pred.append(pred_item)
        
    
    return X_true, X_pred
