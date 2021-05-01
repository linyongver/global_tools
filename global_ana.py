import datetime
import pandas as pd
import pandas as pd
import matplotlib
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib as mpl
display_sfx = 0
display_seed = 0
display_weight = 0
eval_type = "min"
loc_type = "last"#, "last"

exclude_list = [
    "seed", 
#     "sfx",
]

# exps= SPCM__stepfree_QkHYHu
exps=  find_logs_by_multikeys(
        [
            "SPCM__steprestrict_GrWega",
        ], inclstr=[], exclstr=[]) +  [
]

full_df = pd.DataFrame()

import json
from collections import defaultdict
from utils import convert_fmt

full_dict = defaultdict(list)
for exp in exps:
    with open("logs/" + exp + "/args") as f:
        one_args = json.load(f)
        for k, v in one_args.items():
            if v not in full_dict[k]:
                full_dict[k].append(v)
result_list = []
for exp in exps:
    distinct_keys = []
    with open("logs/" + exp + "/args") as f:
        one_args = json.load(f)
        for k, v in one_args.items():
            if k in exclude_list:
                continue
            if [v] != full_dict[k]: 
                distinct_keys.append( "%s=%s" % (k, v))
    key = (", ".join(distinct_keys))
    dftr = pd.read_csv("logs/" + exp+"/res.csv")
