import datetime
import os
import pandas as pd
import pandas as pd
import matplotlib
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
from collections import defaultdict


def find_logs_by_multikeys(path, key, inclstr=[], exclstr=[]):
    assert isinstance(keys, list)
    keys_list = []
    for key in keys:
        keys_list += find_logs(path, key, inclstr=inclstr, exclstr=exclstr)
    return keys_list


def find_logs(path, key, include_str=[], exclude_str=[]):
    # path is the log of log files
    # key use to find the files
    # include strings should be [abc, de] abc and de must be in the name
    # exclude ~~~ not in the name
    log_list = os.listdir(path)
    log_list.sort()
    full_logs = ["%s/%s" % (path, log) for log in log_list if key in log]
    if len(include_str) > 0:
        for instr in include_str:
            full_logs = [fl for fl in full_logs if instr in fl]
    if len(exclude_str) > 0:
        for exstr in exclude_str:
            full_logs = [fl for fl in full_logs if exstr not in fl]
    return full_logs


def find_models_in_logs(exps, exclude_list=[]):
    # find the iterpretable keys of log files
    # input a list of log file names
    # output a dict, key=file name, value=the simplied notation
    # if you want to exclude some key in the returned value, put in exclude_list
    full_dict = defaultdict(list)
    for exp in exps:
        with open(exp + "/args") as f:
            one_args = json.load(f)
            for k, v in one_args.items():
                if v not in full_dict[k]:
                    full_dict[k].append(v)
    keys = []
    for exp in exps:
        distinct_keys = []
        with open(exp + "/args") as f:
            one_args = json.load(f)
            for k, v in one_args.items():
                if k in exclude_list:
                    continue
                if [v] != full_dict[k]: 
                    distinct_keys.append( "%s=%s" % (k, v))
        key = (", ".join(distinct_keys))
        keys.append(key)
    return dict(zip(exps, keys))


def eval_run(df, value_list=[], slice_by=None, order_by=None, loc_type="last", best_by=None):
    # df is the dataframe you want to analysis
    # value_list is the fields you want to return
    # slice_by is in case you have multiple restarts
    # order_by is usually the epoch or step
    # loc_type is how you chose the analsis location
    # best_by is if you want to choose best performance by validation
    if slice_by is not None:
        slices = list(np.unique(df[slice_by]))
        dfs = []
        for isl in slices:
            dfs.append(df[df[slice_by] == isl])
    else:
        dfs = [df]
    ress = []
    for idfs in dfs:
        if order_by is not None:
            idfs =  idfs.sort_values(by=order_by)
        res = dict([(xf, idfs.iloc[-1][xf]) for xf in value_list])
        res.update({"best_loc": -1, "total_epoch": idfs.shape[0]})
        ress.append(res)
    return ress

def format_groupby(res_df, key="model", 
                   mean_std_flds=["train_nll","train_penalty","train_acc","test_acc"],
                   mean_flds=["total_epoch"],
                   count_flds=["best_loc"],
                   mean_std_latex_form=True):
    out_df = res_df.groupby(key).agg(["mean", "std", "count"]).reset_index()
    format_result = []
    for ir in range(out_df.shape[0]):
        count = -1
        one_dict = {"model": out_df.iloc[ir][key][0]}
        for ifd in mean_std_flds:
            meanifd = out_df.iloc[ir][(ifd, "mean")]
            stdifd = out_df.iloc[ir][(ifd, "std")]
            if np.isnan(stdifd):
                format_ifd = ("%.2f" % meanifd)
            else:
                format_ifd = ("$%.2f \pm %.2f$" % (meanifd , stdifd))
            one_dict.update({ifd: format_ifd})
        for ifd in mean_flds:
            meanifd = out_df.iloc[ir][(ifd, "mean")]
            format_ifd = ("%.2f" % meanifd)
            one_dict.update({ifd: format_ifd})
        for ifd in count_flds:
            countifd = out_df.iloc[ir][(ifd, "count")]
            one_dict.update({ifd: countifd})
        format_result.append(one_dict)
    format_df = pd.DataFrame(format_result)
    return format_df
