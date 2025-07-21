import torch
import torch.nn.functional as F
from torch.optim import Adam
import scipy.sparse as sp
import numpy as np
import os
import time
import pandas as pd

from input_data import load_data
from preprocessing import *
from metrics import *
import model as m
import pickle
from model import GATE

import csv

import random

from types import SimpleNamespace


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



subjects = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Methods', 'Reinforcement_Learning',
            'Rule_Learning', 'Theory']

seed = 42
    
results = []

for emb_inter in [1024, 512, 256, 128, 64, 32]:
  for emb_intra in [64]:#[32, 16, 8, 4]:
    
    args = {
        'dataset': 'cora',
        'model': 'GATE',
        'hidden_dims': [1433, 256, emb_inter],
        'lambda_': 0.2,
        'use_feature': True,
        'num_epoch': 400,
        'learning_rate': 0.01,
        'patience': 50,
        'dropout': 0.1
    }
    
    args_inter = {   
        'dataset': 'cora',
        'model': 'GATE',
        'hidden_dims': [emb_inter, 32, emb_intra],
        'lambda_': 0.2,
        'use_feature': True,
        'num_epoch': 400,
        'learning_rate': 0.01,
        'patience': 50,
        'dropout': 0.1
    }
    
    
    
    args_global = {
        'dataset': 'cora',
        'model': 'GATE',
        'hidden_dims': [emb_inter+emb_intra, 64, 32],
        'lambda_': 0.2,
        'use_feature': True,
        'num_epoch': 400,
        'learning_rate': 0.01,
        'patience': 50,
        'dropout': 0.1
    }
    
    
    
    args = SimpleNamespace(**args)
    args_inter = SimpleNamespace(**args_inter)
    args_global = SimpleNamespace(**args_global)
    
    
    
    
    ###### TRAIN ########
    
    
    set_seed(seed)
    
    for subject in subjects:
        # Train on CPU (hide GPU) due to memory constraints
        # os.environ['CUDA_VISIBLE_DEVICES'] = ""
        
        with open('../data/adj_matrix/' + subject + '_adj_matrix.pkl', 'rb') as f:
            adj = pickle.load(f)
        
        with open('../data/features/' + subject + 'features.pkl', 'rb') as f:
            features = pickle.load(f)
        
        # with open('../data/Probabilistic_Methodsfeatures.pkl', 'rb') as f:
        #    features = pickle.load(f)
        
        # Store original adjacency matrix (without diagonal entries) for later
        # rimozione degli elementi diagonali dalla matrice di adiacenza per evitare i self loop
        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()
    
        # suddivisione del grafo in set di addestramento, validazione e test
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj, test = 5, val = 10)
        adj = adj_train
    
        # Some preprocessing
        adj_norm = preprocess_graph(adj)
        
        num_nodes = adj.shape[0]
        
        features = sparse_to_tuple(features.tocoo())
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]
        
        # Create Model
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        
        
        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = sparse_to_tuple(adj_label)
        
        
        
        adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                                torch.FloatTensor(adj_norm[1]), 
                                torch.Size(adj_norm[2]))
        adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                                torch.FloatTensor(adj_label[1]), 
                                torch.Size(adj_label[2]))
        features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), 
                                torch.FloatTensor(features[1]), 
                                torch.Size(features[2]))
        
        weight_mask = adj_label.to_dense().view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0)) 
        weight_tensor[weight_mask] = pos_weight
    
        with open('../data/adj_matrix/' + subject + '_adj_matrix_norm.pkl', 'wb') as f:
            pickle.dump(adj_norm, f)
        with open('../data/features/' + subject + 'features_torch.pkl', 'wb') as f:
            pickle.dump(features, f)
        
        # init model and optimizer
        model = GATE(hidden_dims=args.hidden_dims, lambda_=args.lambda_, A=adj_norm)
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
    
    
        best_val_roc = -float('inf')
        best_val_ap = -float('inf')
        early_stop_count = 0  # Conta il numero di epoche senza miglioramento
        patience = 50  # Numero di epoche senza miglioramento prima di fermarsi
    
        
        for epoch in range(args.num_epoch):
            t = time.time()
            A_pred = model(features)
            optimizer.zero_grad()
            loss = log_lik = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
    
            if 'VGAE' in args.model:
                kl_divergence = 0.5 / A_pred.size(0) * (1 + 2 * model.logstd - model.mean ** 2 - torch.exp(model.logstd) ** 2).sum(1).mean()
                loss -= kl_divergence
        
            loss.backward()
            optimizer.step()
            
            train_acc = get_acc(A_pred, adj_label)
            val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred, adj_orig)
        
            # Monitor the best validation performance
            if val_roc > best_val_roc or val_ap > best_val_ap:
                best_val_roc = val_roc
                best_val_ap = val_ap
                # Save the model if it improves
                torch.save(model.state_dict(), '../data/models/' + subject + '_best_model.pth')
                torch.save(A_pred, '../data/models/' + subject + '_best_Apred.pth')
                early_stop_count = 0  # Reset the counter if there's improvement
            else:
                early_stop_count += 1
            
            if early_stop_count >= patience:
                print("Early stopping at epoch", epoch + 1)
                break
            
            # if epoch % 50 == 0:
            #     print(f"Epoch: {epoch + 1:04d}, train_loss: {loss.item():.5f}, train_acc: {train_acc:.5f}, val_roc: {val_roc:.5f}, val_ap: {val_ap:.5f}, time: {time.time() - t:.5f}")
            
        # Test the model after training
        best_model = GATE(hidden_dims=args.hidden_dims, lambda_=args.lambda_, A=adj_norm)
        best_model.load_state_dict(torch.load('../data/models/' + subject + '_best_model.pth'))
        test_roc, test_ap = get_scores(test_edges, test_edges_false, best_model(features), adj_orig)
        print("End of training!", f"test_roc: {test_roc:.5f}, test_ap: {test_ap:.5f}, best_val_roc: {best_val_roc:.5f}, best_val_ap: {best_val_ap:.5f}")
    
    
    
    
    for subject in subjects:
        # Train on CPU (hide GPU) due to memory constraints
        # os.environ['CUDA_VISIBLE_DEVICES'] = ""
        
        with open('../data/adj_matrix/' + subject + '_adj_matrix_norm.pkl', 'rb') as f:
            adj_norm = pickle.load(f)
        with open('../data/features/' + subject + 'features_torch.pkl', 'rb') as f:
            features = pickle.load(f)
        
        loaded_model = GATE(hidden_dims=args.hidden_dims, lambda_=args.lambda_, A=adj_norm)
        loaded_model.load_state_dict(torch.load('../data/models/' + subject + '_best_model.pth'))
        set_seed(seed)
        encoded_features = loaded_model.encode(features)
        A_pred = loaded_model(features)
        
        torch.save(encoded_features, '../data/features/' + subject + '_encoded_features.pth')
    
    encoded_features_list = []
    
    for subject in subjects:
        file_path = f'../data/features/{subject}_encoded_features.pth'
        encoded_features = torch.load(file_path)  
        encoded_features_list.append(encoded_features.to_dense())
    
    all_encoded_features = torch.cat(encoded_features_list, dim=0)
    
    torch.save(all_encoded_features, '../data/features/all_encoded_features.pth')
    
    
    
    
    
    # Train on CPU (hide GPU) due to memory constraints
    # os.environ['CUDA_VISIBLE_DEVICES'] = ""
    set_seed(seed)
    
    with open('../data/adj_matrix/inter_adj_matrix.pkl', 'rb') as f:
        adj = pickle.load(f)
    
    features = torch.load('../data/features/all_encoded_features.pth')
    
    if features.is_sparse:
        # Converti il tensore in denso
        tensor_dense = features.to_dense()
    else:
        tensor_dense = features
    
    # Converti il tensore in formato NumPy
    tensor_np = tensor_dense.detach().numpy()
    
    # Ora crea una matrice sparsa CSC
    features = sp.csc_matrix(tensor_np)
    
    #with open('../data/Probabilistic_Methodsfeatures.pkl', 'rb') as f:
    #    features = pickle.load(f)
    
    # Store original adjacency matrix (without diagonal entries) for later
    # rimozione degli elementi diagonali dalla matrice di adiacenza per evitare i self loop
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    
    # suddivisione del grafo in set di addestramento, validazione e test
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj, test = 5, val = 10)
    adj = adj_train
    
    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    
    num_nodes = adj.shape[0]
    
    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    
    # Create Model
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    
    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                            torch.FloatTensor(adj_norm[1]), 
                            torch.Size(adj_norm[2]))
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                            torch.FloatTensor(adj_label[1]), 
                            torch.Size(adj_label[2]))
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),  
                                        torch.FloatTensor(features[1]),                         
                                        torch.Size(features[2]))
    
    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)) 
    weight_tensor[weight_mask] = pos_weight
    
    with open('../data/adj_matrix/inter_adj_matrix_norm.pkl', 'wb') as f:
        pickle.dump(adj_norm, f)
    with open('../data/features/inter_features_torch.pkl', 'wb') as f:
        pickle.dump(features, f)  
    
    # init model and optimizer
    model = GATE(hidden_dims=args_inter.hidden_dims, lambda_=args_inter.lambda_, A=adj_norm)
    
    optimizer = Adam(model.parameters(), lr=args_inter.learning_rate)
    
    # Early stopping parameters
    best_val_roc = -float('inf')
    best_val_ap = -float('inf')
    early_stop_count = 0
    patience = 50  # Numero massimo di epoche senza miglioramenti
    
    
    
    # Converti i tensori in vettori
    labels_all = adj_label.to_dense().view(-1).long()
    # Conta il numero di 0 e 1 in labels_all e preds_all
    num_labels = torch.bincount(labels_all)
    print(f"Labels: {dict(enumerate(num_labels.tolist()))}")
    
    
    # Train model
    for epoch in range(args_inter.num_epoch):
        t = time.time()
    
        # Forward pass
        A_pred = model(features)
        optimizer.zero_grad()
        loss = log_lik = norm * F.binary_cross_entropy(
            A_pred.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor
        )
        
        if 'VGAE' in args_inter.model:
            kl_divergence = 0.5 / A_pred.size(0) * (
                1 + 2 * model.logstd - model.mean ** 2 - torch.exp(model.logstd) ** 2
            ).sum(1).mean()
            loss -= kl_divergence
    
        # Backpropagation
        loss.backward()
        optimizer.step()
    
        # Metrics
        train_acc = get_acc(A_pred, adj_label)
        val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred, adj_orig)
    
        # Check for improvement
        if val_roc > best_val_roc or val_ap > best_val_ap:
            best_val_roc = val_roc
            best_val_ap = val_ap
            torch.save(model.state_dict(), '../data/models/inter_best_model.pth')
            torch.save(A_pred, '../data/models/inter_best_Apred.pth')
            early_stop_count = 0  # Reset early stopping counter
        else:
            early_stop_count += 1
    
        # Print progress
        # if epoch % 50 == 0:
        #     print(
        #         f"Epoch: {epoch + 1:04d}, train_loss: {loss.item():.5f}, train_acc: {train_acc:.5f}, "
        #         f"val_roc: {val_roc:.5f}, val_ap: {val_ap:.5f}, time: {time.time() - t:.5f}"
        #     )
    
        # Early stopping condition
        if early_stop_count >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    # Load the best model and evaluate on the test set
    best_model = GATE(hidden_dims=args_inter.hidden_dims, lambda_=args_inter.lambda_, A=adj_norm)
    best_model.load_state_dict(torch.load('../data/models/inter_best_model.pth'))
    set_seed(seed)
    encoded_features = best_model.encode(features)
    torch.save(encoded_features, '../data/features/inter_best_model.pth')
    test_roc, test_ap = get_scores(test_edges, test_edges_false, best_model(features), adj_orig)
    print(
        "End of training!",
        f"Best Validation ROC: {best_val_roc:.5f}, AP: {best_val_ap:.5f}",
        f"Test ROC: {test_roc:.5f}, Test AP: {test_ap:.5f}"
    )
    
    acc = get_acc(A_pred, adj_label)
    precision, recall, f1 = get_precision_recall_f1(A_pred, adj_label)
    print(f"Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    
    
    
    
    
    
    
    
    # Train on CPU (hide GPU) due to memory constraints
    # os.environ['CUDA_VISIBLE_DEVICES'] = ""
    
    set_seed(seed)
    
    with open('../data/adj_matrix/global_adj_matrix.pkl', 'rb') as f:
        adj = pickle.load(f)
    
    features_intra = torch.load('../data/features/all_encoded_features.pth')
    
    features_inter = torch.load('../data/features/inter_best_model.pth')
    
    features = torch.cat((features_inter, features_intra), dim=1)
    
    if features.is_sparse:
        # Converti il tensore in denso
        tensor_dense = features.to_dense()
    else:
        tensor_dense = features
    
    # Converti il tensore in formato NumPy
    tensor_np = tensor_dense.detach().numpy()
    
    # Ora crea una matrice sparsa CSC
    features = sp.csc_matrix(tensor_np)
    
    #with open('../data/Probabilistic_Methodsfeatures.pkl', 'rb') as f:
    #    features = pickle.load(f)
    
    # Store original adjacency matrix (without diagonal entries) for later
    # rimozione degli elementi diagonali dalla matrice di adiacenza per evitare i self loop
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    
    # suddivisione del grafo in set di addestramento, validazione e test
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj, test = 5, val = 10)
    adj = adj_train
    
    # print('train_edges: ', train_edges.shape)
    # print('val_edges: ', val_edges.shape)
    # print('val_edges_false: ', len(val_edges_false))
    # print('test_edges: ', val_edges.shape)
    # print('test_edges_false: ', len(val_edges_false))
    
    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    
    num_nodes = adj.shape[0]
    
    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    
    # Create Model
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    
    
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    
    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                            torch.FloatTensor(adj_norm[1]), 
                            torch.Size(adj_norm[2]))
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                            torch.FloatTensor(adj_label[1]), 
                            torch.Size(adj_label[2]))
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),  
                                        torch.FloatTensor(features[1]),                         
                                        torch.Size(features[2]))
    
    
    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)) 
    weight_tensor[weight_mask] = pos_weight
    
    with open('../data/adj_matrix/global_adj_matrix_norm.pkl', 'wb') as f:
        pickle.dump(adj_norm, f)
    with open('../data/features/global_features_torch.pkl', 'wb') as f:
        pickle.dump(features, f)
    
    
    # init model and optimizer
    model = GATE(hidden_dims=args_global.hidden_dims, lambda_=args_global.lambda_, A=adj_norm)
    
    optimizer = Adam(model.parameters(), lr=args_global.learning_rate)
    
    # Early stopping parameters
    best_val_roc = -float('inf')
    best_val_ap = -float('inf')
    early_stop_count = 0
    patience = 50  # Numero massimo di epoche senza miglioramenti
    
    # Converti i tensori in vettori
    labels_all = adj_label.to_dense().view(-1).long()
    # Conta il numero di 0 e 1 in labels_all e preds_all
    num_labels = torch.bincount(labels_all)
    print(f"Labels: {dict(enumerate(num_labels.tolist()))}")
    
    # Train model
    for epoch in range(args_global.num_epoch):
        t = time.time()
    
        # Forward pass
        A_pred = model(features)
        optimizer.zero_grad()
        loss = log_lik = norm * F.binary_cross_entropy(
            A_pred.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor
        )
    
        if 'VGAE' in args_global.model:
            kl_divergence = 0.5 / A_pred.size(0) * (
                1 + 2 * model.logstd - model.mean ** 2 - torch.exp(model.logstd) ** 2
            ).sum(1).mean()
            loss -= kl_divergence
    
        # Backpropagation
        loss.backward()
        optimizer.step()
    
        # Metrics
        train_acc = get_acc(A_pred, adj_label)
        val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred, adj_orig)
    
        # Check for improvement
        if val_roc > best_val_roc or val_ap > best_val_ap:
            best_val_roc = val_roc
            best_val_ap = val_ap
            torch.save(model.state_dict(), '../data/models/global_best_model.pth')
            torch.save(A_pred, '../data/models/global_best_Apred.pth')
            early_stop_count = 0  # Reset early stopping counter
        else:
            early_stop_count += 1
    
        # Print progress
        # if epoch % 50 == 0:
        #     print(
        #         f"Epoch: {epoch + 1:04d}, train_loss: {loss.item():.5f}, train_acc: {train_acc:.5f}, "
        #         f"val_roc: {val_roc:.5f}, val_ap: {val_ap:.5f}, time: {time.time() - t:.5f}"
        #     )
    
        # Early stopping condition
        if early_stop_count >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    # Load the best model and evaluate on the test set
    best_model = GATE(hidden_dims=args_global.hidden_dims, lambda_=args_global.lambda_, A=adj_norm)
    best_model.load_state_dict(torch.load('../data/models/global_best_model.pth'))
    test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred, adj_orig)
    print(
        "End of training!",
        f"Best Validation ROC: {best_val_roc:.5f}, AP: {best_val_ap:.5f}",
        f"Test ROC: {test_roc:.5f}, Test AP: {test_ap:.5f}"
    )
    
    acc = get_acc(A_pred, adj_label)
    precision, recall, f1 = get_precision_recall_f1(A_pred, adj_label)
    print(f"Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    
    
    
    
    
    
    ####### TEST ##########
    
    set_seed(seed)
    
    # with open('../data/adj_matrix/inter_adj_matrix.pkl', 'rb') as f:
    with open('../data/adj_matrix/global_adj_matrix.pkl', 'rb') as f:
        adj_generale = pickle.load(f)
    
    # Store original adjacency matrix (without diagonal entries) for later
    # rimozione degli elementi diagonali dalla matrice di adiacenza per evitare i self loop
    adj_generale_orig = adj_generale
    adj_generale_orig = adj_generale_orig - sp.dia_matrix((adj_generale_orig.diagonal()[np.newaxis, :], [0]), shape=adj_generale_orig.shape)
    adj_generale_orig.eliminate_zeros()
    
    adj_generale_label = adj_generale_orig + sp.eye(adj_generale_orig.shape[0])
    adj_generale_label = sparse_to_tuple(adj_generale_label)
    
    adj_generale_label = torch.sparse.FloatTensor(torch.LongTensor(adj_generale_label[0].T), 
                            torch.FloatTensor(adj_generale_label[1]), 
                            torch.Size(adj_generale_label[2]))
    
    # suddivisione del grafo in set di addestramento, validazione e test
    adj_generale_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_generale, test = 5, val = 10)
    
    
    
    
    
    
    
    
    model_dir = '../data/models/'
    
    node_ranges = []
    l = 0
    for i, subject in enumerate(subjects):
        # Carica la matrice
        file_path = f"{model_dir}{subject}_best_Apred.pth"
        A_pred_single = torch.load(file_path)
        node_ranges.append((l,l+A_pred_single.shape[0]))
        l = l + A_pred_single.shape[0]
    
    
    
    
    # Directory con le matrici salvate
    model_dir = '../data/models/'
    
    # Calcola il numero totale di nodi
    num_total_nodes = max([end for _, end in node_ranges])  # Indice massimo +1
    
    # Inizializza la matrice finale come una matrice sparsa (più efficiente in termini di memoria)
    final_adj_matrix = torch.zeros((num_total_nodes, num_total_nodes))
    
    for i, subject in enumerate(subjects):
        # Carica la matrice
        file_path = f"{model_dir}{subject}_best_Apred.pth"
        A_pred_single = torch.load(file_path)
        A_pred_dense = A_pred.to_dense() if A_pred_single.is_sparse else A_pred_single
        
        # Ottieni il range di nodi per questa matrice
        start, end = node_ranges[i]
        
        # Inserisci la matrice nel blocco corrispondente della matrice finale
        final_adj_matrix[start:end, start:end] = A_pred_dense
    
    print("Shape of the final adjacency matrix:", final_adj_matrix.shape)
    
    # binary_final_matrix = (final_adj_matrix > 0.5).float()
    
    
    
    
    file_path = f'../data/models/inter_best_Apred.pth'
    
    A_pred_inter = torch.load(file_path)
    
    
    file_path = f'../data/models/global_best_Apred.pth'
    
    A_pred_global = torch.load(file_path)
    
    
    tensor = (A_pred_inter + final_adj_matrix) * A_pred_global
    
    
    
    ####### SAVE ########    
    # Nome file dinamico
    filename = f"Results/Results_emb_inter{emb_inter}_emb_intra{emb_intra}.csv"
    
    # Calcolo metriche
    test_roc, test_ap = get_scores(test_edges, test_edges_false, tensor, adj_generale_orig)
    acc = get_acc(tensor, adj_generale_label)
    precision, recall, f1 = get_precision_recall_f1(tensor, adj_generale_label)
    
    # Salva in lista
    results.append({
        'emb_inter': emb_inter,
        'emb_intra': emb_intra,
        'ROC AUC': round(float(test_roc), 3),
        'AP': round(float(test_ap), 3),
        'Accuracy': round(float(acc), 3),
        'Precision': round(float(precision), 3),
        'Recall': round(float(recall), 3),
        'F1': round(float(f1), 3),
    })

    # Crea DataFrame e salva
    df = pd.DataFrame(results)
    df.to_csv("Results/Ablation.csv", index=False)
    
    
    
    
    
