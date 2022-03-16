# -*- coding: utf-8 -*-


import os
import re
import numpy as np


def lines_info(inDir, projectsList):
    lines_len = []
    total_lines = 0

    for i, name in enumerate(projectsList):
        print(f"*** Processing {i + 1}/{len(projectsList)}: {name}")
        data = open(os.path.join(inDir, name), "r", encoding="utf-8").read()

        for file in data.split('\n'):
            # split by [;{}] & filter out empty lines
            code_lines = re.findall('.*?[;{}]', file)
            code_lines = list(filter(None, code_lines))  # remove none entries
            code_lines = [line.strip() for line in code_lines if line.strip()]  # Remove empty lines
            total_lines += len(code_lines)

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
                lines_len.append(len(tokens_stream))
                
    print(f"Total Lines: {total_lines}")
    return lines_len


javaDir = './data/java'
javaProjects = [name for name in os.listdir(javaDir)]

lines_len = lines_info(javaDir, javaProjects)

print(f"Min: {np.min(lines_len)}")
print(f"Max: {np.max(lines_len)}")
print(f"Mean: {np.mean(lines_len)}")
print(f"Median: {np.median(lines_len)}")
print(f"STD: {np.std(lines_len)}")
