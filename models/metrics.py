
import nltk
import torch
import torchmetrics
from torchmetrics.functional.classification import accuracy
from torchmetrics.functional.classification.f_beta import f1, fbeta
from torchmetrics.functional.classification.precision_recall import (precision,
                                                                     recall)



def custom_bleu(references, hypotheses, weights, sf):  # target, preds
    return nltk.translate.bleu_score.sentence_bleu([references], hypotheses, weights=weights, smoothing_function=sf)


def custom_rouge(preds, targets):
    return torchmetrics.functional.rouge_score(preds=preds, targets=targets, rouge_keys=("rougeL"))['rougeL_precision']


def mrr(preds, target, k=1):
    total = target.size(0)
    _, pred = preds.topk(k, 1, True, True)

    hits = torch.nonzero(pred == target)
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks) / total
    return mrr


def test_metrics(labels, logits):
    scores = {}
    vocab_size = logits.size(-1)
    for i in [1, 2, 3, 4, 5, 10]:
        scores[f'acc_{i}'] = accuracy(logits, labels, average='weighted', num_classes=vocab_size, top_k=i)
        scores[f'mrr_{i}'] = mrr(logits, labels, k=i)
        scores['precision'] = precision(logits, labels, average='weighted', num_classes=vocab_size, top_k=i)
        scores['recall'] = recall(logits, labels, average='weighted', num_classes=vocab_size, top_k=i)
        scores['f1'] = f1(logits, labels, average='weighted', num_classes=vocab_size, beta=1, top_k=i)
        scores['fbeta'] = fbeta(logits, labels, average='weighted', num_classes=vocab_size, beta=0.5, top_k=i)

    return scores


def stack_scores(scores):
    stacked_list = {}
    for out in scores:
        for key, val in out.items():
            if key in stacked_list:
                stacked_list[key].append(val)
            else:
                stacked_list[key] = [val]

    _template = {}
    for key, val in stacked_list.items():
        _template[key] = torch.stack(val).mean()

    _template = dict(sorted(_template.items()))  # sort by keys
    return _template
