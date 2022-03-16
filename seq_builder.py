# -*- coding: utf-8 -*-


import os
import re

from utils import all_equal


def sequence_builder(inDir, outDir, projectsList):
    context_size = 10
    try:
        os.makedirs(outDir)
    except:
        pass

    for i, name in enumerate(projectsList):
        print(f"*** Processing {i + 1}/{len(projectsList)}: {name}")
        data = open(os.path.join(inDir, name), "r", encoding="utf-8").read()
        out_file = open(os.path.join(outDir, name), 'w+', encoding="utf-8")
        for file in data.split('\n'):

            # split by [;{}] & filter out empty lines
            code_lines = re.findall('.*?[;{}]', file)
            code_lines = list(filter(None, code_lines))  # remove none entries
            code_lines = [line.strip() for line in code_lines if line.strip()]  # Remove empty lines

            j = 0
            while (True):
                if j >= len(code_lines) - 1:
                    break

                if code_lines[j].startswith("for ") and (code_lines[j].endswith(";")):
                    k = 1
                    try:
                        while not code_lines[j + k].endswith("{"):
                            if k >= 2:
                                break
                            k += 1

                        # +1 becuase range takes one less the max length
                        tokens_stream = " ".join(code_lines[j:j + k + 1]).split()
                        j = j + k
                    except:
                        tokens_stream = code_lines[j].split()
                else:
                    tokens_stream = code_lines[j].split()

                j += 1  # loop increment

                # skip tokens with length less than 2 or having same values or grater than context_size
                if len(tokens_stream) < 2 or all_equal(tokens_stream) or len(tokens_stream) > context_size:
                    continue

                out_file.write(" ".join(tokens_stream) + '\n')


javaDir = './data/java'
javaProjects = [name for name in os.listdir(javaDir)]
javaOutDir = './data/templateLines/java'
sequence_builder(javaDir, javaOutDir, javaProjects)
