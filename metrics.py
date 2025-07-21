import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
import numpy as np


def get_scores(edges_pos, edges_neg, adj_rec, adj_orig):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        # print(e)
        # print(adj_rec[e[0], e[1]])
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:

        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def get_precision_recall_f1(adj_rec, adj_label):
    """
    Calcola precision, recall e F1 score tra una matrice di adiacenza predetta e una etichetta.
    
    Args:
        adj_rec (torch.Tensor): Matrice di adiacenza predetta (logit o probabilità).
        adj_label (torch.Tensor): Matrice di adiacenza etichetta (0 o 1).
        
    Returns:
        precision (float): Precision del modello.
        recall (float): Recall del modello.
        f1 (float): F1 score del modello.
    """
    # Converti i tensori in vettori
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()

    # Calcola metriche
    precision = precision_score(labels_all.cpu().numpy(), preds_all.cpu().numpy())
    recall = recall_score(labels_all.cpu().numpy(), preds_all.cpu().numpy())
    f1 = f1_score(labels_all.cpu().numpy(), preds_all.cpu().numpy())
    
    return precision, recall, f1
