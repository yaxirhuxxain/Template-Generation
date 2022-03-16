# -*- coding: utf-8 -*-

import os
import xlsxwriter
from itertools import groupby


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def sec_to_hms(seconds):
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    periods = [('hours', hours), ('minutes', minutes), ('seconds', seconds)]
    time_string = ', '.join('{} {}'.format(value, name)
                            for name, value in periods
                            if value)
    return time_string


def pretty_dict(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty_dict(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def scores_to_xls(scores, out_file):
    workbook = xlsxwriter.Workbook(out_file)
    worksheet = workbook.add_worksheet("scores")

    # first line will always be header
    for col, value in enumerate(scores.keys()):
        worksheet.write(1, col, value)

    # start insert vlaues from 2nd row
    for col, value in enumerate(scores.values()):
        worksheet.write(2, col, value)

    workbook.close()


def get_folder_size(source):
    total_size = 0
    total_size = os.path.getsize(source)
    for item in os.listdir(source):
        itempath = os.path.join(source, item)
        if os.path.isfile(itempath):
            total_size += os.path.getsize(itempath)
        elif os.path.isdir(itempath):
            total_size += get_folder_size(itempath)
    return float(total_size) / 1048576


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


class BeamSearchNode(object):
    def __init__(self, seed, pred, prevNode, logProb, length):
        '''
        :param seed:
        :param pred:
        :param prevNode:
        :param logProb:
        :param length:
        '''
        self.seed = seed
        self.pred = pred
        self.logp = logProb
        self.prevNode = prevNode
        self.len = length

    def eval(self, alpha=1.0, reward = 0):
        return self.logp / float(self.len - 1 + 1e-6) + alpha * reward

