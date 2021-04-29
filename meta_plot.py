
exps=[
'SPCM/20210324/11:30_CR:0.999_0.8_0.1_LNs:0.1_resnet50_sepfc2_ofc_BS:32_LR:0.01_WithIRM_steprestrict_CM:0_NINR:5_IRMWht:10000.0_Sfx:SPCM__steprestrict_PwRfRy',
'SPCM/20210324/11:30_CR:0.999_0.8_0.1_LNs:0.1_resnet50_sepfc2_ofc_BS:32_LR:0.01_WithIRM_steprestrict_NINR:5_IRMWht:10000.0_Sfx:SPCM__steprestrict_PwRfRy',

'SPCM/20210324/11:30_CR:0.999_0.8_0.1_LNs:0.1_resnet50_sepfc2_ofc_BS:32_LR:0.01_WithIRM_steprestrict_CM:0_NINR:10_IRMWht:10000.0_Sfx:SPCM__steprestrict_PwRfRy',
'SPCM/20210324/11:31_CR:0.999_0.8_0.1_LNs:0.1_resnet50_sepfc2_ofc_BS:32_LR:0.01_WithIRM_steprestrict_NINR:10_IRMWht:10000.0_Sfx:SPCM__steprestrict_PwRfRy'
]

full_df = pd.DataFrame()

import json
from collections import defaultdict
full_dict = defaultdict(list)
for exp in exps:
    with open("logs/" + exp + "/args") as f:
        one_args = json.load(f)
        for k, v in one_args.items():
            if v not in full_dict[k]:
                full_dict[k].append(v)

for exp in exps:
    distinct_keys = []
    with open("logs/" + exp + "/args") as f:
        one_args = json.load(f)
        for k, v in one_args.items():
            if [v] != full_dict[k]:
                distinct_keys.append( "%s=%s" % (k, v))
    key = (", ".join(distinct_keys))
    dftr = pd.read_csv("logs/" + exp+"/train.csv")
    with open("logs/" + exp + "/args") as f:
        one_args = json.load(f)
    dftr["mean_train_loss"] = (dftr["avg_loss_group:0"] + dftr["avg_loss_group:1"])/2
    try:
        dftr["diff_train_loss"] = (abs(dftr["avg_loss_group:0"] - dftr["avg_loss_group:3"]) + 
                                   abs(dftr["avg_loss_group:0"] - dftr["avg_loss_group:2"]))/2
    except:
        dftr["diff_train_loss"] = abs(dftr["avg_loss_group:0"] - dftr["avg_loss_group:1"])
    dftr["mean_train_acc"] = (dftr["avg_acc_group:0"] + dftr["avg_acc_group:1"])/2

    dfte = pd.read_csv("logs/" + exp+"/test.csv")
    one_df = pd.DataFrame({
        "epoch": dftr["epoch"].tolist(),
        "mean_train_loss": dftr["mean_train_loss"].tolist(),
        "diff_train_loss": dftr["diff_train_loss"].tolist(),
        "penalty": dftr["penalty:-1"].tolist(),
        "test_acc": dfte["avg_acc_group:0"].tolist(),
        "train_acc": dftr["mean_train_acc"].tolist(),
        "model":  [key] * len(dfte["avg_loss_group:0"].tolist())
    })
    full_df = pd.concat([full_df, one_df], ignore_index=True)

full_df = full_df[full_df.epoch<100]
# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()
import matplotlib as mpl

mpl.rcParams['figure.dpi']= 300
mpl.rcParams['figure.figsize']= (16,4)
# Load an example dataset
fig, axis = plt.subplots(nrows=1, ncols=2)

# Create a visualization
sns.lineplot(
    x=full_df["epoch"], y=full_df["mean_train_loss"],
    hue=full_df["model"], ax=axis[0]
)
sns.lineplot(
    x=full_df["epoch"], y=full_df["diff_train_loss"],
    hue=full_df["model"], ax=axis[1]
)

fig, axis = plt.subplots(nrows=1, ncols=2)
sns.lineplot(
    x=full_df["epoch"], y=full_df["train_acc"],
    hue=full_df["model"], ax=axis[0]
)
sns.lineplot(
    x=full_df["epoch"], y=full_df["test_acc"],
    hue=full_df["model"], ax=axis[1]
)
plt.show()

fig, axis = plt.subplots(nrows=1, ncols=1)
sns.lineplot(
    x=full_df["epoch"], y=full_df["penalty"],
    hue=full_df["model"], ax=axis
)
plt.show()
