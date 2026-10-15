'''
Take in the log files from Pytorch Lighning and plot the curves
'''
import os
import numpy as np
import pandas as pd
import altair as alt
import argparse
import json
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from scipy.interpolate import make_interp_spline

## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)
alt.themes.enable("publishTheme")

def combine_dataframes(dataframes: list[dict[str, np.ndarray]]) -> pd.DataFrame:
    '''
    Combine the dataframes into one to be plotted
    '''
    df_L = []
    for name, df in dataframes.items():
        epochs = np.arange(len(df))
        data = df
        if "loss" in name:
            df_L.append(pd.DataFrame({"epoch": epochs, "loss": data, "type": name}))
        elif "acc" in name:
            df_L.append(pd.DataFrame({"epoch": epochs, "accuracy": data, "type": name}))
        elif "f1" in name:
            df_L.append(pd.DataFrame({"epoch": epochs, "f1_score": data, "type": name}))
        else:
            raise ValueError("Dataframe name must contain either 'loss', 'f1_score', or 'accuracy'")
    return pd.concat(df_L)

def find_files(log_dir, ftype, name=""):
    '''
    Find the files in the directory
    '''
    files = []
    exp = []
    for each in log_dir:
        if each.endswith(f".{ftype}"):
            exp.append("default")
            files.append(each)
        else:
            for dir,_,file in os.walk(each):
                for f in file:
                    if f.endswith(f".{ftype}") and (name in f):
                        files.append(os.path.join(dir, f))
                        exp_name = dir.split("/")[-2]
                        exp.append(exp_name)
    return files, exp

def gather_data_from_csv_log(log_dir, params, labels=None):
    '''
    Gather the data from the log files
    '''
    files, exp = find_files(log_dir, "csv")
    res_dict_L = []
    for param in params:
        current_param_df_L = []
        for i, each in enumerate(files):
            df = pd.read_csv(os.path.join(each))
            if param not in df.columns:
                raise ValueError(f"{param} not in the csv file {os.path.join(log_dir, each)}")
            numpy_data = df.loc[df[param].notnull().to_numpy(), [param]].to_numpy().reshape(-1)
            epoch = df.loc[df[param].notnull(), ["epoch"]].to_numpy().reshape(-1)
            if labels:
                name = labels[i]
            else:
                name = exp[i]
            current_param_df_L.append(pd.DataFrame({"exp": name, 
                                                    param: numpy_data, 
                                                    "epoch": epoch}))
        res_dict_L.append((pd.concat(current_param_df_L), param))
    return res_dict_L

def smooth_adjacent_mean(y, window_size=3):
    """
    Smooths an array by taking the mean of each point and its neighbors.
    Default window size = 3 (previous, current, next point).
    """
    # Ensure window size is odd
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
        
    pad = window_size // 2
    y_padded = np.pad(y, (pad, pad), mode='edge')
    
    smooth_y = np.convolve(y_padded, np.ones(window_size)/window_size, mode='valid')
    return smooth_y

def gather_json_roc(log_dir, labels, save_dir, exp_name):
    files, exp = find_files(log_dir, "json", "roc")
    assert len(files) == len(labels), f"Found {len(files)} files and {len(labels)} labels, expect equal number"
    res_dict_L = []
    all_auc = []
    for i, each in enumerate(files):
        with open(each, "r") as f:
            data = json.load(f)
        if labels:
            name = labels[i]
        else:
            name = exp[i]
        auc_L = []
        fpr_L = []
        tpr_L = []
        for each_fold in data.values():
            auc_L.append(each_fold["roc_auc_micro"])
            fpr_L.append(each_fold["fpr_micro"])
            tpr_L.append(each_fold["tpr_micro"])
        # compute the mean and std for auc, fpr, and tpr
        auc_mean = np.mean(auc_L)
        auc_std = np.std(auc_L)
        mean_fpr = np.linspace(0, 1, 100)
        tpr_interp = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_L, tpr_L)]
        for tpr in tpr_interp:
            tpr[0] = 0.0
        # comput std and mean for each fpr point
        tpr_interp = np.array(tpr_interp)
        tpr_mean = tpr_interp.mean(axis=0)
        tpr_std = tpr_interp.std(axis=0)

        tpr_mean_smooth = smooth_adjacent_mean(tpr_mean, window_size=3)

        # Enforce monotonicity
        tpr_mean_smooth = np.maximum.accumulate(tpr_mean_smooth)
        tpr_mean_smooth = np.clip(tpr_mean_smooth, 0, 1)

        # compute the mean and std for auc, fpr, and tpr
        # auc_mean = np.mean(auc_L)
        # auc_std = np.std(auc_L)

        # mean_fpr = np.linspace(0, 1, 100)
        # tpr_interp = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_L, tpr_L)]
        # tpr_interp = np.array(tpr_interp)

        # tpr_mean = np.mean(tpr_interp, axis=0)
        # tpr_std  = np.std(tpr_interp, axis=0)

        # tpr_mean = np.maximum.accumulate(tpr_mean)
        # tpr_mean = savgol_filter(tpr_mean, 51, 3)
        # tpr_std  = savgol_filter(tpr_std, 51, 3)
        # tpr_mean = np.maximum.accumulate(np.clip(tpr_mean, 0, 1))

        # put into a dataframe
        #mean_fpr, tpr_mean
        df = pd.DataFrame({"fpr": mean_fpr, 
                           "tpr": tpr_mean_smooth,
                        #    "tpr_std": tpr_std,
                            "AUC": auc_mean,
                            # "auc_std": auc_std,
                           "Experiment": name,
                           "Feature Extractor": name.split("+")[-1],
                           "Input Image": ("SPAGHETTI Transformed" if "spaghetti" == name.split("+")[0].lower()
                                         else "CycleGAN Transformed" if "cyclegan" == name.split("+")[0].lower()
                                         else "Pink Filter" if "filter" == name.split("+")[0].lower()
                                         else "UTOM" if "utom" == name.split("+")[0].lower()
                                         else "Original")})
        res_dict_L.append(df)
        all_auc.append(auc_L)
    final_df = pd.concat(res_dict_L)
    auc_df = pd.DataFrame({"AUC": [item for sublist in all_auc for item in sublist],
                           "Experiment": [name for name in labels for _ in range(len(all_auc[0]))]})
    # group by the feature extractor
    feature_L = [x.split("+")[-1] for x in labels for _ in range(len(all_auc[0]))]
    auc_df["Feature Extractor"] = feature_L
    # plot box plot using altair
    chart = alt.Chart(auc_df).mark_boxplot().encode(
        x=alt.X("Experiment:N", axis=alt.Axis(labelAngle=-45), sort=labels),
        y="AUC:Q",
        color=alt.Color("Feature Extractor:N", sort=labels)
    ).interactive()
    chart.save(os.path.join(save_dir, f"auc_box_{exp_name}.html"))
    return final_df

def gather_json_pr(log_dir, labels):
    files, exp = find_files(log_dir, "json", "pr")
    res_dict_L = []
    for i, each in enumerate(files):
        with open(each, "r") as f:
            data = json.load(f)
        if labels:
            name = labels[i]
        else:
            name = exp[i]
        ap_L = []
        precision_L = []
        recall_L = []
        for each_fold in data.values():
            ap_L.append(each_fold["ap_micro"])
            precision_L.append(each_fold["precision_micro"])
            recall_L.append(each_fold["recall_micro"])
        # compute the mean and std for auc, fpr, and tpr
        auc_mean = np.mean(ap_L)
        auc_std = np.std(ap_L)
        mean_recall = np.linspace(0, 1, 100)
        precision_interp = [np.interp(mean_recall, recall, precision) for recall, precision in zip(recall_L, precision_L)]
        for precision in precision_interp:
            precision[0] = 0.0
        # comput std and mean for each fpr point
        precision_interp = np.array(precision_interp)
        precision_mean = precision_interp.mean(axis=0)
        precision_std = precision_interp.std(axis=0)
        # put into a dataframe
        df = pd.DataFrame({"recall": mean_recall,
                           "precision": precision_mean,
                           "precision_std": precision_std,
                            "ap": auc_mean,
                            "ap_std": auc_std,
                           "exp": f"{name}, auc={auc_mean:.2f}+-{auc_std:.2f}"})
        res_dict_L.append(df)
    final_df = pd.concat(res_dict_L)
    return final_df

def gather_txt(log_dir, labels=None, ftype="txt"):
    '''
    Gather the data from the log files
    '''
    files, exp = find_files(log_dir, ftype)
    final_df = pd.DataFrame()
    for i, each in enumerate(files):
        if each.endswith(".npy"):
            data = np.load(each)
        else:
            with open(each, "r") as f:
                data = f.readlines()
                # convert to float
                data = [float(x) for x in data]
        if labels:
            exp_name = labels[i]
        else:
            exp_name = exp[i]
        df = pd.DataFrame({"exp": exp_name, "dice_score": data})
        final_df = pd.concat([final_df, df])
    return final_df

def plot_curves_epochs(data, save_dir, name, labels):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for df, param in data:
        chart = alt.Chart(df).mark_line().encode(
            x="epoch:N",
            y=f"{param}:Q",
            color=alt.Color("exp:N", sort=labels)
        ).interactive()
        chart.save(os.path.join(save_dir, f"{name}_{param}.html"))
    print("Plots saved")

def plot_combined_curves_roc(data, save_dir, name, labels, no_error_bar=False):
    '''
    Plot the combined ROC curve
    '''
    ref = pd.DataFrame({"fpr":[0, 1], "tpr":[0,1]})
    if not no_error_bar:
        data["tpr_upper"] = data["tpr"] + data["tpr_std"]
        data["tpr_lower"] = data["tpr"] - data["tpr_std"]
    # plot with Altair, where same exp_feature is the same color, and same exp_model is the same line style
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X("fpr:Q", title="1 - Specificity"),
        y=alt.Y("tpr:Q", title="Sensitivity"),
        color=alt.Color("Feature Extractor:N", sort=labels),
        strokeDash=alt.StrokeDash("Input Image:N", sort=labels)
    )
    if not no_error_bar:
        error_bar = chart.mark_area(opacity=0.3).encode(
            y=alt.Y("tpr_lower:Q", title="Sensitivity"),
            y2=alt.Y("tpr_upper:Q", title="Sensitivity")
        )
    line = alt.Chart(ref).mark_line(strokeDash=[1,1]).encode(
       x=alt.X("fpr:Q", title="1 - Specificity"),
       y=alt.Y("tpr:Q", title="Sensitivity"),
    )
    if no_error_bar:
        final_chat = chart + line
    else:
        final_chat = chart + error_bar + line
    # change final_chart x and y labels
    final_chat = final_chat.interactive()
    final_chat.save(os.path.join(save_dir, f"{name}_roc.html"))
    data.to_csv(os.path.join(save_dir, f"{name}_roc.csv"))
    print("Plots and csv saved")

def plot_combined_curves_pr(data, save_dir, name, labels):
    '''
    Plot the combined ROC curve
    '''
    data["precision_upper"] = data["precision"] + data["precision_std"]
    data["precision_lower"] = data["precision"] - data["precision_std"]
    # plot with Altair, add std deviation
    chart = alt.Chart(data).mark_line().encode(
        x="recall:Q",
        y="precision:Q",
        color=alt.Color("exp:N", sort=labels)
    )
    error_bar = chart.mark_area(opacity=0.3).encode(
        y="precision_lower:Q",
        y2="precision_upper:Q"
    )
    final_chat = chart + error_bar
    final_chat.interactive().save(os.path.join(save_dir, f"{name}_pr.html"))
    data.to_csv(os.path.join(save_dir, f"{name}_pr.csv"))
    print("Plots and csv saved")

def plot_box(data: pd.DataFrame, save_dir, name, labels, if_horizontal=False):
    '''
    plot side by side box plots
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if if_horizontal:
        chart = alt.Chart(data).mark_boxplot().encode(
            y=alt.Y("exp:N", sort=labels),
            x="dice_score:Q",
            color="exp:N"
        ).interactive()
    else:
        chart = alt.Chart(data).mark_boxplot().encode(
            x=alt.X("exp:N", axis=alt.Axis(labelAngle=-45), sort=labels),
            y="dice_score:Q",
            color="exp:N"
        ).interactive()
    chart.save(os.path.join(save_dir, f"{name}_box.html"))
    print("Plots saved")

def plot_bar(data: pd.DataFrame, save_dir, name, labels):
    '''
    plot side by side bar plots
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    bars = alt.Chart(data).mark_bar().encode(
        x=alt.X("exp:N", axis=alt.Axis(labelAngle=-45), sort=labels),
        y=alt.Y("dice_score:Q", aggregate="mean"),
        color="exp:N"
    )
    error_bars = alt.Chart(data).mark_errorbar(extent='stdev').encode(
        x=alt.X("exp:N", axis=alt.Axis(labelAngle=-45), sort=labels),
        y=alt.Y('dice_score:Q', aggregate='mean'),
        y2=alt.Y2('dice_score:Q', aggregate='stdev')
    )
    chart = bars + error_bars
    chart.interactive().save(os.path.join(save_dir, f"{name}_bar.html"))
    print("Plots saved")

def plot_confusion_matrix(pred: list[str], tru: list[str],
                          save: str, name: str) -> None:
    # check if any of the list is empty
    # if so, do not plot as they are not not on the primary device
    if (len(pred)==0) or (len(tru)==0):
        return None
    # confusion matrix
    labels = np.union1d(tru,pred)
    cm = confusion_matrix(tru, pred, normalize='true', labels=labels)
    df = pd.DataFrame(cm, index = labels, columns = labels)
    plt.figure(figsize=(20, 20))
    sns.heatmap(df, vmin=0, vmax=1)#, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix from Inference (Normalized by Rows)')
    plt.savefig(os.path.join(save, f"cm_{name}_inference.png"))
    plt.close()

def plot_ROC_curve(pred_L: list[np.ndarray], true_L: list[int], cls_to_idx: dict[str, int],
                   save: str, name: str) -> None:
    '''
    Plot to multi-class ROC curve
    '''
    idx_to_cls = {v: k for k, v in cls_to_idx.items()}
    n_classes = len(cls_to_idx)
    final_dict = dict()
    for fold in range(len(pred_L)):
        true_bin = label_binarize(true_L[fold], classes=list(cls_to_idx.values()))
        pred = np.array(pred_L[fold])
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr_array, tpr_array, _ = roc_curve(true_bin[:, i], pred[:, i])
            fpr[idx_to_cls[i]] = fpr_array.tolist()
            tpr[idx_to_cls[i]] = tpr_array.tolist()
            roc_auc[idx_to_cls[i]] = auc(fpr[idx_to_cls[i]], tpr[idx_to_cls[i]])

        # Compute micro-average ROC curve and ROC area (aggregate over all classes)
        fpr_array, tpr_array, _ = roc_curve(true_bin.ravel(), pred.ravel())
        fpr["micro"] = fpr_array.tolist()
        tpr["micro"] = tpr_array.tolist()
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # plot with Altair for the individual classes
        fpr_L = []
        tpr_L = []
        cls_L = []
        for x in cls_to_idx.keys():
            fpr_L.extend(fpr[x])
            tpr_L.extend(tpr[x])
            cls_L.extend([x]*len(fpr[x]))
        roc_df = pd.DataFrame({"fpr": fpr_L,
                            "tpr": tpr_L, 
                            "class": cls_L})
        chart = alt.Chart(roc_df).mark_line().encode(
            x="fpr:Q",
            y="tpr:Q",
            color="class:N"
        ).interactive()
        chart.save(os.path.join(save, f"roc_{name}_fold{fold}.html"))
        print("AUC scores: ", roc_auc)

        # save the three dicts as json
        big_dic = {f"fold_{fold}": 
                   {"roc_auc": roc_auc, "fpr": fpr, "tpr": tpr, 
                    "roc_auc_micro": roc_auc["micro"], "fpr_micro": fpr["micro"], "tpr_micro": tpr["micro"], 
                    "exp": name}
                }
        final_dict = final_dict | big_dic
        
    with open(os.path.join(save, f"roc_{name}.json"), 'w') as f:
        json.dump(final_dict, f)

def plot_PR_curve(pred_L: list[np.ndarray], true_L: list[int], cls_to_idx: dict[str, int],
                   save: str, name: str) -> None:
    '''
    Plot to multi-class PR curve
    '''
    idx_to_cls = {v: k for k, v in cls_to_idx.items()}
    n_classes = len(cls_to_idx)
    final_dict = dict()
    for fold in range(len(pred_L)):
        true_bin = label_binarize(true_L[fold], classes=list(cls_to_idx.values()))
        pred = np.array(pred_L[fold])
        # Compute ROC curve and ROC area for each class
        precision = dict()
        recall = dict()
        ap = dict()

        for i in range(n_classes):
            precision_array, recall_array, _ = precision_recall_curve(true_bin[:, i], pred[:, i])
            precision[idx_to_cls[i]] = precision_array.tolist()
            recall[idx_to_cls[i]] = recall_array.tolist()
            ap[idx_to_cls[i]] = average_precision_score(true_bin[:, i], pred[:, i])

        # Compute micro-average ROC curve and ROC area (aggregate over all classes)
        precision_array, recall_array, _ = precision_recall_curve(true_bin.ravel(), pred.ravel())
        precision["micro"] = precision_array.tolist()
        recall["micro"] = recall_array.tolist()
        ap["micro"] = average_precision_score(true_bin.ravel(), pred.ravel())

        # plot with Altair for the individual classes
        precision_L = []
        recall_L = []
        cls_L = []
        for x in cls_to_idx.keys():
            precision_L.extend(precision[x])
            recall_L.extend(recall[x])
            cls_L.extend([x]*len(precision[x]))
        ap_df = pd.DataFrame({"precision": precision_L,
                            "recall": recall_L, 
                            "class": cls_L})
        chart = alt.Chart(ap_df).mark_line().encode(
            x="recall:Q",
            y="precision:Q",
            color="class:N"
        ).interactive()
        chart.save(os.path.join(save, f"pr_{name}_fold{fold}.html"))
        print("AP scores: ", ap)

        # save the three dicts as json
        big_dic = {f"fold_{fold}": 
                   {"ap": ap, "precision": precision, "recall": recall, 
                    "ap_micro": ap["micro"], "precision_micro": precision["micro"], "recall_micro": recall["micro"], 
                    "exp": name}
                }
        final_dict = final_dict | big_dic
        
    with open(os.path.join(save, f"pr_{name}.json"), 'w') as f:
        json.dump(final_dict, f)

def main():
    if args.labels:
        assert len(args.labels) == len(args.log_dir), "Length must be the same for labels and log_dir"
    if args.type == "roc":
        result_df = gather_json_roc(args.log_dir, args.labels, args.save_dir, args.name)
        plot_combined_curves_roc(result_df,  args.save_dir, args.name, args.labels, args.no_error_bar)
    elif args.type == "pr":
        result_df = gather_json_pr(args.log_dir, args.labels)
        plot_combined_curves_pr(result_df,  args.save_dir, args.name, args.labels)
    elif args.type == "epoch":
        result_L = gather_data_from_csv_log(args.log_dir, args.param, args.labels)
        plot_curves_epochs(result_L, args.save_dir, args.name, args.labels)
    elif args.type == "box":
        result = gather_txt(args.log_dir, args.labels, args.file_type)
        plot_box(result, args.save_dir, args.name, args.labels, args.horizontal)
    elif args.type == "bar":
        result = gather_txt(args.log_dir, args.labels)
        plot_bar(result, args.save_dir, args.name, args.labels)
    else:
        raise ValueError("Type not supported")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Plotting curves by combining various PyTorch Lightning metrics.csv files")
    argparser.add_argument("--log_dir", type=str, nargs="+", help="The director(ies) of the log file(s). Can have subdirectories")
    argparser.add_argument("--save_dir", type=str, help="The directory to save the plots")
    argparser.add_argument("--name", type=str, help="The name of the experiment")
    argparser.add_argument("--type", type=str, default="box", help="The type of plot, can be roc, pr, epoch, or box")
    argparser.add_argument("--file_type", type=str, default="txt", help="The type of file")
    argparser.add_argument("--labels", type=str, nargs="+", default=None, help="labels for each directories. If not used, will use the directory name")
    argparser.add_argument("--param", type=str, nargs="+", 
                           help="parameters in the csv file to plot in one plot. Must exist in all csv files")
    argparser.add_argument("--horizontal", action="store_true", default=False, help="plot horizontal box plot")
    argparser.add_argument("--no_error_bar", action="store_true", default=False, help="do not plot error bar for ROC and PR curves")
    args = argparser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main()

