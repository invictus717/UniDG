# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections


import argparse
import functools
import glob
import pickle
import itertools
import json
import os
import random
import sys
import time
import numpy as np
import tqdm

from domainbed import datasets
from domainbed import algorithms
from domainbed.lib import misc, reporting
from domainbed import model_selection
from domainbed.lib.query import Q
import warnings

def format_mean(data, latex):
    """Given a list of datapoints, return a string describing their mean and
    standard error"""
    if len(data) == 0:
        return None, None, "X"
    mean = 100 * np.mean(list(data))
    err = 100 * np.std(list(data) / np.sqrt(len(data)))
    if latex:
        return mean, err, "{:.2f} $\\pm$ {:.2f}".format(mean, err)
    else:
        return mean, err, "{:.2f} +/- {:.2f}".format(mean, err)

def print_table(table, header_text, row_labels, col_labels, colwidth=10,
    latex=True):
    """Pretty-print a 2D array of data, optionally with row/col labels"""
    print("")

    if latex:
        num_cols = len(table[0])
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * num_cols + "}")
        print("\\toprule")
    else:
        print("--------", header_text)

    for row, label in zip(table, row_labels):
        row.insert(0, label)

    if latex:
        col_labels = ["\\textbf{" + str(col_label).replace("%", "\\%") + "}"
            for col_label in col_labels]
    table.insert(0, col_labels)

    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth, latex=latex)
        if latex and r == 0:
            print("\\midrule")
    if latex:
        print("\\bottomrule")
        print("\\end{tabular}}")
        print("\\end{center}")
def print_comparison_table(table, ad_table, header_text, row_labels, col_labels, colwidth=10,
    latex=True):
    """Pretty-print a 2D array of data, optionally with row/col labels"""
    print("")

    if latex:
        num_cols = len(table[0])
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * num_cols + "}")
        print("\\toprule")
    else:
        print("--------", header_text)

    for row, label in zip(table, row_labels):
        row.insert(0, label)

    if latex:
        col_labels = ["\\textbf{" + str(col_label).replace("%", "\\%") + "}"
            for col_label in col_labels]
    table.insert(0, col_labels)

    compare_table = [table[1][0] + "(Adapted)"]
  
    for i, row in enumerate(table):
        if i == 0:
            continue
        for j, column in  enumerate(row):
            if j == 0:
                continue
            else:
                cmp = str(round(eval(ad_table[i-1][j-1].split()[0]) - eval(table[i][j].split()[0]),3))
                target = '\\reshl' + '{' + str(ad_table[i-1][j-1]) + '}' + '{' + str(cmp) + '}'
                compare_table.append(target)
    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth, latex=latex)
        if latex and r == 0:
            print("\\midrule")
    for r1, row1 in enumerate([compare_table]):
        misc.print_row(row1, colwidth=colwidth+15, latex=latex)
    if latex:
        print("\\bottomrule")
        print("\\end{tabular}}")
        print("\\end{center}")
def print_results_tables_adaption(records, adaptive_records, selection_method, ad_selection_method,latex):
    """Given all records, print a results table for each dataset."""

    # grouped_records = reporting.get_grouped_records(records).map(lambda group:
    # { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    # ).filter(lambda g: g["sweep_acc"] is not None)
    grouped_records = reporting.get_grouped_records(records).map(lambda group:
    { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    )
    # print(grouped_records)
    adaptive_grouped_records = reporting.get_grouped_records(adaptive_records).map(lambda group:
    { **group, "sweep_acc": ad_selection_method.sweep_acc(group["records"]) }
    )

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    avg_tables = []
    for dataset in dataset_names:
        if latex:
            print("\\subsection{{{}}}".format(dataset))
        test_envs = range(datasets.num_environments(dataset))

        table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
        ad_table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
        for i, algorithm in enumerate(alg_names):
            means = []
            ad_means = []
            trial_means = []
            for j, test_env in enumerate(test_envs):
                trial_accs = (grouped_records
                    .filter_equals(
                        "dataset, algorithm, test_env",
                        (dataset, algorithm, test_env)
                    ).select("sweep_acc"))
                mean, err, table[i][j] = format_mean(trial_accs, latex)
                means.append(mean)
            for k, ad_test_env in enumerate(test_envs):
                ad_trial_accs = (adaptive_grouped_records
                    .filter_equals(
                        "dataset, algorithm, test_env",
                        (dataset, algorithm, ad_test_env)
                    ).select("sweep_acc"))
                ad_mean, ad_err, ad_table[i][k] = format_mean(ad_trial_accs, latex)
                ad_means.append(ad_mean)

            trial_averages = (adaptive_grouped_records
                .filter_equals("algorithm, dataset", (algorithm, dataset))
                .group("trial_seed")
                .map(lambda trial_seed, group:
                    group.select("sweep_acc").mean()
                    )
                )
            trial_mean, trial_err, ad_table[i][-1] = format_mean(trial_averages, latex)
            if None in means:
                table[i][-1] = "X"
            else:
                table[i][-1] = "{:.1f}".format(sum(means) / len(means))

        col_labels = [
            "Algorithm", 
            *datasets.get_dataset_class(dataset).ENVIRONMENTS,
            "Avg"
        ]
        header_text = (f"Dataset: {dataset}, "
            f"model selection method: {selection_method.name}")
        print_comparison_table(table, ad_table,header_text, alg_names, list(col_labels),
            colwidth=20, latex=latex)
    return [table[-1][-1], ad_table[-1][-1]]
def print_avg_table(all_records, records,  selection_method, ad_selection_method,latex,dataset):
    dataset_names = dataset

    if latex:
        print()
        print("\\subsection{Averages}")
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])
    base = [all_records[m][0] for m in range(len(dataset))]
    adapt = [ all_records[m][1] for m in range(len(dataset))]
    base_num = [eval(all_records[m][0]) for m in range(len(dataset))]
    adapt_num = [eval(all_records[m][1].split()[0]) for m in range(len(dataset))]
    adapt = ['\\reshl' + '{' + adapt[k] + '}' + '{' + str(round(adapt_num[k]-base_num[k],1)) + '}' for k in range(len(base_num))]
    base.append(str(round(np.mean(np.array(base_num)),1)))
    adapt.append(str(round(np.mean(np.array(adapt_num)),1)))
    adapt[-1] = '\\reshl' + '{' + adapt[-1] + '}' + '{' + str(round(eval(adapt[-1]) - eval(base[-1]),1)) + '}'
    adapt.insert(0,"UniDG")
    # base_results = [, np.mean(np.array([all_records[m][0] for m in range(len(dataset))]))]
    col_labels = ["Algorithm", *dataset_names, "Avg"]
    header_text = f"Averages, model selection method: {selection_method.name}"
    print_table([base,adapt], header_text, alg_names, col_labels, colwidth=40,
        latex=latex)

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--adapt_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--adapt_algorithm", type=str, default="UniDG")
    args = parser.parse_args()
    log_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    results_file = "adapt_{}_{}_results.tex".format(args.adapt_algorithm,log_time) if args.latex else "adapt_{}_{}_results.txt".format(args.adapt_algorithm,log_time)

    if not os.path.exists(args.output_dir):
        os.system('mkdir -p {}'.format(args.output_dir))
    output_path = os.path.join(args.output_dir, results_file)
    output_file = open(output_path,'w')
    sys.stdout = misc.Tee(output_path, "w")

    if args.latex:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\usepackage{adjustbox}")
        print("\\usepackage[table]{xcolor}")
        print('\\definecolor{mygreen}{HTML}{39b54a}')
        print('\\newcommand{\\reshl}[2]{\\textbf{#1} \\fontsize{7.5pt}{1em}\selectfont\color{mygreen}{$\\uparrow$ \\textbf{#2}}}')
        print("\\begin{document}")
        print("\\section{Experimental results}") 

    DATASET_NAME = [
        'VLCS',
        'PACS',
        'OfficeHome',
        'TerraIncognita'
    ]

    '''
    Referring to Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization (NeurIPS 2021 Spotlight)
    
    The selection method are organized as:
    1. For choosing a base model, you should choose the most generalized (argmax(mean(envs_out_acc))) one among all envs.

    2. In terms of adaption algorithm, you should choose the most promising (argmax(test_env_in_acc)) models of the specific test env.
    '''

    selection_method  = model_selection.IIDAccuracySelectionMethod
    selection_method_adaption = model_selection.IIDAccuracySelectionMethod_Adaption
    all_records = []
    for dataset in DATASET_NAME:
        records = reporting.load_records(os.path.join(args.input_dir,dataset))
        adaptive_records = reporting.load_adaptive_records(os.path.join(args.adapt_dir,dataset),algo=args.adapt_algorithm)
        print("% Current records:", len(records))
        print("% Current Adaption records:", len(adaptive_records))
        if args.latex:
            print()
            # print("\\subsection{{ Experimental results with {}}}".format(
            #     selection_method.name)) 
        avg_base, avg_adapt = print_results_tables_adaption(records, adaptive_records ,selection_method=selection_method, ad_selection_method=selection_method_adaption, latex = args.latex)
        all_records.append([avg_base, avg_adapt])
    print_avg_table(all_records, records, selection_method=selection_method, ad_selection_method=selection_method_adaption, latex = args.latex,dataset=DATASET_NAME)
    
    if args.latex:
        print("\\end{document}")

    output_file.close()
