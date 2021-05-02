# global_tools
全局import
import sys
sys.path.append('/home/ylindf/projects/tools')

将args转成header
将args转成header，sfx永远在最后
图片: https://uploader.shimo.im/f/HCtSyjVoRZfiaBXY.png
exDict = {"heheda_hhjk": 7897,
          "sfx": "what",
         "bobo_jj": "kjlj",
         "1_2":3}
default_dict = {"1_2":3}
args2header(exDict, default_dict)

将args export到一个路径
图片: https://uploader.shimo.im/f/HJuor4xTZCyRiVnY.png
将cmd export到一个路径
用法save_cmd(logger_path, sys.argv): #
图片: https://uploader.shimo.im/f/oMB6raxdNiYYW7y2.png


Logger
将一个dict结合epoch和batch一起记录下来，output到csv path里面
声明
图片: https://uploader.shimo.im/f/W2JfzNnE2eAQOfKg.png

387         train_csv_logger = LYCSVLogger(os.path.join(logger_path, 'train.csv'),  mode=mode)
388         val_csv_logger =  LYCSVLogger(os.path.join(logger_path, 'val.csv'),  mode=mode)
389         test_csv_logger =  LYCSVLogger(os.path.join(logger_path, 'test.csv'),  mode=mode)
输出
1000         csv_logger.log(epoch, batch_idx, ees.get_log_summary()) # 每次log自动输出

epoch 性质统计
输出需要统计的量，dict形式，结合logger使用，可能需要重载get_summary
图片: https://uploader.shimo.im/f/rKotkTPeHfLDwdCq.png

整理std之类的
format_groupby(res_df, 
               key="model", 
               mean_std_flds=["train_nll","train_penalty","train_acc","test_acc"],
               mean_flds=["total_epoch"],
               count_flds=["best_loc"])
mean_std_flds会算mean+-std
mean_flds会算mean
count_flds只算count
结果整理
import sys
sys.path.append('/home/ylindf/projects/tools')

from global_utils import args2header, save_args, save_cmd, LYCSVLogger
from global_ana import find_logs, find_logs_by_multikeys
from global_ana import find_models_in_logs
from global_ana import eval_run
from global_ana import format_groupby
import pandas as pd
import numpy as np

# find logs by keys
exps = find_logs(path="logs", key="CMNIST0502171347")
# find model in the logs
log_models = find_models_in_logs(exps)
eval_reses = []
all_dfs = []
for exp in exps:
    model = log_models[exp]
    df = pd.read_csv("%s/res.csv" % exp)
    df["model"] = model
    all_dfs.append(df)
    # read the result files
    eval_res = eval_run(df, 
             value_list=["train_nll","train_acc",
                 "train_penalty","test_acc"],
             slice_by="restart",
             order_by="epoch")
    for eval_item in eval_res:
        full_item = {"model": model}
        full_item.update(eval_item)
        eval_reses.append(full_item)
res_df = pd.DataFrame(eval_reses)
res_df["train_penalty"] = res_df["train_penalty"] * 10e6
res_df["train_nll"] = res_df["train_nll"] * 100
res_df["train_acc"] = res_df["train_acc"] * 100
res_df["test_acc"] = res_df["test_acc"] * 100

format_groupby(res_df, 
               key="model", 
               mean_std_flds=["train_acc","test_acc","train_nll","train_penalty"],
               mean_flds=["total_epoch"],
               count_flds=["best_loc"])
图片: https://uploader.shimo.im/f/lH1fvTR8lT31DuD4.png

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
# Apply the default theme
sns.set_theme()
import matplotlib as mpl

mpl.rcParams['figure.dpi']= 100
mpl.rcParams['figure.figsize']= (16,8)

fast = True # not showing std will be much faster

plot_fields = ["train_nll","train_penalty","train_acc","test_acc"]
x_axis = "epoch"
key="model"
fig, axis = plt.subplots(nrows=(len(plot_fields) + 1) // 2, ncols=2)

full_df = pd.concat(all_dfs)
if fast == True:
    full_df = full_df.groupby(["model", "epoch"]).mean().reset_index()
    full_df = full_df.sort_values(by="epoch")
for ipf in range(len(plot_fields)):
    sns.lineplot(
        x=full_df[x_axis], y=full_df[plot_fields[ipf]],
        hue=full_df[key], ax=axis[ipf // 2, ipf % 2])
plt.show()

图片: https://uploader.shimo.im/f/YP6UZUCVAD8zFegK.png

bash
sfx=CMNIST$(date +%m%d%H%M%S)
echo $sfx
for mix_up in 0 1;
do
    python -u main.py   --hidden_dim=390   --l2_regularizer_weight=0.0011 --lr=0.000489 --penalty_anneal_iters=50  --penalty_weight=91257 --mix_up $mix_up    --steps=501  --n_restarts 1 --data_num 10000 --sfx $sfx
done

