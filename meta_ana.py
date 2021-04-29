
    # date_range(begt="20201030", endt="20201231")
CelebA_analyze_configure = [
    {"file": "train.csv",
    "groups": ["avg_acc_group:0", "avg_acc_group:1", 
            "avg_acc_group:2", "avg_acc_group:3"]},
    {"file": "val.csv",
    "groups": ["avg_acc_group:0", "avg_acc_group:1", 
            "avg_acc_group:2", "avg_acc_group:3"]},
    {"file": "test.csv",
    "groups": ["avg_acc_group:0", "avg_acc_group:1", 
            "avg_acc_group:2", "avg_acc_group:3"]}
]

CUB_analyze_configure = CelebA_analyze_configure
PACS_analyze_configure = [
    {"file": "train.csv",
    "groups": ["avg_acc_group:0", "avg_acc_group:1", 
            "avg_acc_group:2"]},
    {"file": "val.csv",
    "groups": ["avg_acc_group:0", "avg_acc_group:1", 
            "avg_acc_group:2"]},
    {"file": "test.csv",
    "groups": ["avg_acc_group:0"]}
]
SP_analyze_configure = [
    {"file": "train.csv",
    "groups": ["avg_acc_group:0", "avg_acc_group:1"]},
    {"file": "test.csv",
    "groups": ["avg_acc_group:0"]}
]


def date_range(begt, endt):
    begt = str(begt) 
    endt = str(endt)
    from datetime import datetime
    from datetime import timedelta
    dt_b = datetime.strptime(begt, "%Y%m%d")
    dt_e = datetime.strptime(endt, "%Y%m%d")
    i = dt_b
    date_list = []
    while i <= dt_e:
        date_list.append(datetime.strftime(i, "%Y%m%d"))
        i = i + timedelta(days=1)

    return date_list


def eval_run(logf, configures, etype="min", loc_type="best"):
#     df_val = pd.read_csv("logs/%s/%s" % (logf, configures[1]["file"]))
    df_test = pd.read_csv("logs/%s/%s" % (logf, configures[1]["file"]))
    df_train = pd.read_csv("logs/%s/%s" % (logf, configures[0]["file"]))
#     print(df_test)
#     print(best_loc)
    if etype == "min":
#         df_val["w_group"] = df_val[configures[1]["groups"]].min(axis=1)
        df_train["w_group"] = df_train[configures[0]["groups"]].min(axis=1)
        df_test["w_group"] = df_test[configures[1]["groups"]].min(axis=1)
        df_train["w_loss"] = df_train[["avg_loss_group:0", "avg_loss_group:1"]].min(axis=1)
    
    elif etype == "mean":
#         df_val["w_group"] = df_val[configures[1]["groups"]].mean(axis=1)
        df_train["w_group"] = df_test[configures[0]["groups"]].mean(axis=1)
        df_test["w_group"] = df_test[configures[1]["groups"]].mean(axis=1)
        df_train["w_loss"] = df_train[["avg_loss_group:0", "avg_loss_group:1"]].mean(axis=1)
    if loc_type == "best":
        best_loc = df_test["w_group"].argmax()
#         best_val = df_val["w_group"].iloc[best_loc]
        best_train = df_train["w_group"].iloc[best_loc]
        best_test = df_test["w_group"].iloc[best_loc]
        best_loss = df_train["w_loss"].iloc[best_loc]
    elif loc_type == "last":
        best_loc = df_test.shape[0]-1
#         print(df_val.shape, df_test.shape)
#         print()
#         best_val = df_val["w_group"].iloc[best_loc]
        best_test = df_test["w_group"].iloc[best_loc]
        best_train = df_train["w_group"].iloc[best_loc]
        best_loss = df_train["w_loss"].iloc[best_loc]
    elif "fix" in loc_type:
        cons = loc_type.split("_")
        best_loc = int(cons[1])
#         best_loc = df_test.shape[0]-1
#         print(df_val.shape, df_test.shape)
#         print()
#         best_val = df_val["w_group"].iloc[best_loc]
        best_test = df_test["w_group"].iloc[best_loc]
        best_train = df_train["w_group"].iloc[best_loc]
        best_loss = df_train["w_loss"].iloc[best_loc]
    elif loc_type == "last5":
        best_loc = -5
#         best_val = df_val["w_group"].iloc[-5:].mean()
        best_test = df_test["w_group"].iloc[-5:].mean()
        best_train = df_train["w_group"].iloc[-5:].mean()
        best_loss = df_train["w_loss"].iloc[-5:].mean()
    else:
        raise Exception

    return {
        "model_name": logf.split("sfx")[-1],
        "best_loc": best_loc,
        "best_train": best_train,
        "best_test": best_test,
        "best_loss": best_loss,
        "total_epoch": df_test.shape[0]}


def find_logs(idate_list, include_str, exclude_str, datasets):
    full_logs = [] 
#     include_str = [key] # 
#     exclude_str = []
    for dataset in datasets:
        for idate in idate_list:
            log_dir = dataset + "/" + idate
            # log_dir = "CUB/20201019"
            # log_dir = "PACS/20201021"

            try:
                log_list = os.listdir("logs/" + log_dir)
                log_list.sort()
                log_list
                logs = [log_dir + "/" + log for log in log_list]
                for log in logs:
                    try:
                        path = "logs/" + log + "/test.csv"
                        file = pd.read_csv(path)
                        if file.shape[0] > thres:
                            full_logs.append(log)
                    except:
                        pass
            except:
                pass
    if len(include_str) > 0:
        for instr in include_str:
            full_logs = [fl for fl in full_logs if instr in fl]
    if len(exclude_str) > 0:
        for exstr in exclude_str:
            full_logs = [fl for fl in full_logs if exstr not in fl]
    module_full_logs = full_logs
#     print(module_full_logs)
    return module_full_logs


def find_logs_by_keys(key, datasets, inclstr=[], exclstr=[]):
    m_path = "meta_run/auto/" + key
    import json
    with open(m_path + "/args") as f:
        meta_args = json.load(f)
    try:
        start_time = datetime.datetime.strptime(
                    meta_args["time"], '%m/%d/%Y, %H:%M:%S')
        from datetime import timedelta
        end_time = start_time + timedelta(days=20)
        start_date = datetime.datetime.strftime(
                start_time,'%Y%m%d')
        end_date = datetime.datetime.strftime(
                end_time,'%Y%m%d')
    except:
        start_date = "20201101"
        end_date = datetime.datetime.strftime(
                datetime.datetime.now(),'%Y%m%d')
    
    drange = date_range(begt=start_date, endt=end_date)

    module_full_logs = find_logs(
        idate_list=drange,
        include_str=[key] + inclstr,
        exclude_str=[] + exclstr,
        datasets=datasets)

    return module_full_logs

def ana_by_keys(keys, etype="min", loc_type="best"):
    for one_key in keys:
    #     datasets = ["SPCelebA"]
    #     print(one_key, isinstance(one_key, str))
        if isinstance(one_key, str):
            key = one_key
            name = "Default"
            datasets = [one_key.split("__")[0]]
        else:
            if len(one_key) == 3:
                key, name, datasets = one_key
            elif len(one_key) == 2:
                key, name = one_key
                datasets = [one_key[0].split("__")[0]]
            elif len(one_key) == 1:
                key = one_key[0]
                name = "Default"
                datasets = [one_key[0].split("__")[0]]
            else:
                raise Exception
        print(key, name, datasets)
    #     try:
        if datasets[0] in ["PACS", "VLCS", "office_home"]:
            configures = PACS_analyze_configure
        elif datasets[0] in ["CelebA", "CUB"]:
            configures = CelebA_analyze_configure
        elif datasets[0] in ["SPCelebA", "SPCUB", "SPCM", "CMNIST"]:
            configures = SP_analyze_configure
        else:  
    #         print(datasets[0])
            raise exception
        m_path = "meta_run/auto/" + key
        import json
        with open(m_path + "/args") as f:
            meta_args = json.load(f)
        module_full_logs = find_logs_by_keys(key, datasets=datasets)
        distinct_keys = []
        total_trails = 1
    #     print(meta_args)
        key_trails = []
        for k, v in meta_args.items():
            if k != "n_epoch" and not isinstance(v, str):
                total_trails *= len(v)
            if len(v) > 1 and not isinstance(v, str):
                distinct_keys.append(k)
                key_trails.append(len(v))
        key_trails = np.array(key_trails)
        sort_index = np.argsort(key_trails)
        distinct_keys = list(np.array(distinct_keys)[sort_index])
        total_epochs = sum([int(x) for x in meta_args["n_epoch"]]) * total_trails

#         print(distinct_keys)
        meta_results = []
        finished_trials = 0
        finished_epochs = 0
        for one_run in module_full_logs:
            one_result = {"path": one_run}
            with open("logs/" + one_run + "/args") as f:
                one_args = json.load(f)
#             print(one_args)
            one_args["n_epoch"] = one_args["n_epochs"]
            for dk in distinct_keys:
                one_result.update({dk: one_args[dk]})
            one_result.update(eval_run(one_run, configures=configures, 
                                       etype=etype, loc_type=loc_type))
            meta_results.append(one_result)
            finished_epochs += one_result["total_epoch"]
    #         print(one_result["total_epoch"], meta_args["n_epoch"])
            if one_result["total_epoch"] in ([int(x)for x in meta_args["n_epoch"]]):
                finished_trials += 1
        try:
            now = datetime.datetime.now() 
            time_spent = datetime.datetime.now() - datetime.datetime.strptime(
                meta_args["time"], '%m/%d/%Y, %H:%M:%S')
            expected_time = now - time_spent + time_spent * total_epochs / finished_epochs
            expected_time = expected_time.strftime('%Y-%m-%d %H:%M:%S')
        except:
            expected_time = "NULL"
        full_res_df = pd.DataFrame(meta_results)
        sum_df = full_res_df[distinct_keys + 
                                   [ "best_test", "best_train", 
                                    "best_loss", "best_loc", "total_epoch"]] 
        # sum_df[sum_df.seed == 1234] if x !="seed" if x !="target_domain" PACS__irmsep__RrAjs if x !="target_domain"
        try:
            if display_seeds:
                keys = [x for x in distinct_keys if  x !="target_domain" ]
            else:
                keys = [x for x in distinct_keys if  x !="target_domain"  and  x !="seed"]
            agg_df = sum_df.groupby(keys).agg(["mean", "count", "std"]) # 
            agg_df = agg_df.drop([('best_loc',  'count'), 
                         ('best_loc',   'std'),
                         ('total_epoch',   'std'),
                         ("best_test", "count"),
                         ("best_train", "count"),
                         ("best_loss", "count"),
                        ], axis=1)
            
#             agg_df.drop([('best_loc', 'count')], axis=1)
#             agg_df.drop([('best_loc',   'std')], axis=1)
#             ('total_epoch', 'count'),
#             ('total_epoch',   'std')]
#             print(agg_df.columns)
            print("-" * 5, key,  name, 
                      "%s/%s" %(finished_trials, total_trails), 
                      "%s/%s" % (finished_epochs, total_epochs), 
                      "ETA:%s" % expected_time)
            if display_style:
                df_pre = (agg_df).reset_index()
                for_dict = {"test_mean": df_pre["best_test"]["mean"].round(3) * 100,
                           "test_std": df_pre["best_test"]["std"].round(3) * 100,
                           "train_mean": df_pre["best_train"]["mean"].round(3) * 100,
                           "train_std": df_pre["best_train"]["std"].round(3) * 100,
                           "total_epoch": df_pre["total_epoch"]["mean"],
                           "best_loc": df_pre["best_loc"]["mean"],
                           "trials": df_pre["total_epoch"]["count"],}
                for ikey in keys:
                    for_dict.update({ikey:df_pre[ikey]})
                df_org = pd.DataFrame(for_dict)
                out_list = []
                for row in range(df_org.shape[0]):
                    if not np.isnan(df_org.iloc[row]["std"]):
                        out_list.append("$%.1f" % (df_org.iloc[row]["mean"]) + "\pm" + "%.1f$" % (df_org.iloc[row]["std"]))
                    else:
                        out_list.append("%.1f" % (df_org.iloc[row]["mean"]))
                df_org["perform"] = out_list
                print("-" * 20)
                display(HTML(df_org[keys + ["perform", "trials", "best_loc", "total_epoch"]].to_html()))
            else:
                display(HTML(agg_df.to_html()))
        except:
            print("-" * 5, key,  name, 
                  "%s/%s" %(finished_trials, total_trails), 
                  "%s/%s" % (finished_epochs, total_epochs), 
                  "ETA:%s" % expected_time)
            display(HTML(sum_df.to_html()))
