import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
from sklearn.metrics import (confusion_matrix, precision_score,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader


def compute_scores(model: nn.Module, l_dataset: DataLoader):
    all_predictions = list()
    all_labels = list()
    for indexed_elem in l_dataset:
        indexed_in = indexed_elem["inputs"]
        indexed_labels = indexed_elem["outputs"]
        predictions = model(indexed_in)
        predictions = torch.argmax(predictions, -1).view(-1)
        labels = indexed_labels.view(-1)
        valid_indices = labels != 0

        valid_predictions = predictions[valid_indices]
        valid_labels = labels[valid_indices]

        all_predictions.extend(valid_predictions.tolist())
        all_labels.extend(valid_labels.tolist())
    # global precision. Does take class imbalance into account.
    micro_precision_recall_fscore = precision_recall_fscore_support(all_labels, all_predictions,
                                                                    average="micro",
                                                                    zero_division=0)

    # precision per class and arithmetic average of them. Does not take into account class imbalance.
    macro_precision_recall_fscore = precision_recall_fscore_support(
        all_labels, all_predictions, average="macro", zero_division=0)

    per_class_precision = precision_score(all_labels, all_predictions,
                                          average=None,
                                          zero_division=0)

    return {"macro_precision_recall_fscore": macro_precision_recall_fscore,
            "micro_precision_recall_fscore": micro_precision_recall_fscore,
            "per_class_precision": per_class_precision,
            "confusion_matrix": confusion_matrix(all_labels, all_predictions,
                                                 normalize='true')}


def pprint_confusion_matrix(conf_matrix, num_classes):
    df_cm = pd.DataFrame(conf_matrix, range(num_classes), range(num_classes))
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.show()
