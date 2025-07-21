import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

import args
import args_global

# Funzione di attivazione (sostituendo lambda con una funzione esplicita)
def identity_activation(x):
    return x


class VGAE(nn.Module):
    def __init__(self, adj):
        super(VGAE, self).__init__()
        self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=identity_activation)
        self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=identity_activation)

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred


class VGAE_global(nn.Module):
    def __init__(self, adj):
        super(VGAE_global, self).__init__()
        self.base_gcn = GraphConvSparse(args_global.input_dim, args_global.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(args_global.hidden1_dim, args_global.hidden2_dim, adj, activation=identity_activation)
        self.gcn_logstddev = GraphConvSparse(args_global.hidden1_dim, args_global.hidden2_dim, adj, activation=identity_activation)

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), args_global.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred


class VGAE_extended(nn.Module):
    def __init__(self, adj):
        super(VGAE_extended, self).__init__()
        self.base_gcn = GraphConvSparse(args_extended.input_dim, args_extended.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(args_extended.hidden1_dim, args_extended.hidden2_dim, adj, activation=identity_activation)
        self.gcn_logstddev = GraphConvSparse(args_extended.hidden1_dim, args_extended.hidden2_dim, adj, activation=identity_activation)

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), args_extended.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


class GAE(nn.Module):
    def __init__(self, adj=None):
        super(GAE, self).__init__()
        self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=identity_activation)

    def encode(self, X):
        hidden = self.base_gcn(X)
        z = self.mean = self.gcn_mean(hidden)
        return z

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred


class GAE_global(nn.Module):
    def __init__(self, adj=None):
        super(GAE_global, self).__init__()
        self.base_gcn = GraphConvSparse(args_global.input_dim, args_global.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(args_global.hidden1_dim, args_global.hidden2_dim, adj, activation=identity_activation)

    def encode(self, X):
        hidden = self.base_gcn(X)
        z = self.mean = self.gcn_mean(hidden)
        return z

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred


class GAE_extended(nn.Module):
    def __init__(self, adj=None):
        super(GAE_extended, self).__init__()
        self.base_gcn = GraphConvSparse(args_extended.input_dim, args_extended.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(args_extended.hidden1_dim, args_extended.hidden2_dim, adj, activation=identity_activation)

    def encode(self, X):
        hidden = self.base_gcn(X)
        z = self.mean = self.gcn_mean(hidden)
        return z

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred


# class GraphConv(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(VGAE,self).__init__()
#         self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
#         self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
#         self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

#     def forward(self, X, A):
#         out = A*X*self.w0
#         out = F.relu(out)
#         out = A*X*self.w0
#         return out





# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import args_gate as config  # Importa il file di configurazione

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class GATE(nn.Module):
#     def __init__(self, hidden_dims, lambda_, A):
#         super(GATE, self).__init__()
#         self.lambda_ = lambda_
#         self.n_layers = len(hidden_dims) - 1

#         self.A = A

#         # Pesi per ciascun layer
#         self.W = nn.ParameterList()
#         for i in range(self.n_layers):
#             weight = nn.Parameter(torch.Tensor(hidden_dims[i], hidden_dims[i+1]))
#             nn.init.xavier_uniform_(weight)
#             self.W.append(weight)

#         # Parametri per il meccanismo di attenzione: v0 e v1 per ogni layer
#         self.v0 = nn.ParameterList()
#         self.v1 = nn.ParameterList()
#         for i in range(self.n_layers):
#             a0 = nn.Parameter(torch.Tensor(hidden_dims[i+1], 1))
#             a1 = nn.Parameter(torch.Tensor(hidden_dims[i+1], 1))
#             nn.init.xavier_uniform_(a0)
#             nn.init.xavier_uniform_(a1)
#             self.v0.append(a0)
#             self.v1.append(a1)

#         # Dizionario per salvare (opzionalmente) le matrici di attenzione per ogni layer
#         self.C = {}

#     def forward(self, X):
#         """
#         Args:
#           A: matrice di adiacenza normalizzata in formato sparso (torch.sparse_coo_tensor) di shape (N, N)
#           X: feature dei nodi, tensor denso di shape (N, F_in)
#         Returns:
#           A_pred: matrice di adiacenza ricostruita (tensor denso di shape (N, N))
#         """
#         H = X
#         self.C = {}

#         # Encoder: trasformazione lineare seguita dalla propagazione tramite attenzione
#         for layer in range(self.n_layers):
#             H = torch.matmul(H, self.W[layer])
#             C_layer = self.graph_attention_layer(self.A, H, self.v0[layer], self.v1[layer])
#             self.C[layer] = C_layer
#             H = torch.sparse.mm(C_layer, H)
        
#         self.H = H  # rappresentazione finale dei nodi
        
#         # Decoder: ricostruzione della matrice di adiacenza tramite prodotto interno
#         A_pred = torch.sigmoid(torch.matmul(H, H.t()))
#         return A_pred

#     def graph_attention_layer(self, A, M, v0, v1):
#         """
#         Calcola la matrice di attenzione usando A in formato sparso.
        
#         Args:
#           A: torch.sparse_coo_tensor, matrice di adiacenza normalizzata (N, N)
#           M: rappresentazioni nodali (tensor denso di shape (N, F))
#           v0, v1: parametri di attenzione (tensor di shape (F, 1))
        
#         Returns:
#           att: torch.sparse_coo_tensor, matrice di attenzione normalizzata (N, N)
#         """
#         # Estrai indici e valori dal tensore sparso A
#         indices = A._indices()   # shape [2, E]
#         values = A._values()     # shape [E]

#         # Calcola i termini di attenzione per ciascun nodo
#         f1 = torch.matmul(M, v0).squeeze(1)  # shape (N,)
#         f2 = torch.matmul(M, v1).squeeze(1)  # shape (N,)

#         # Per ogni arco (i, j) in A, calcola il logit: f1[i] + f2[j]
#         edge_logits = f1[indices[0]] + f2[indices[1]]  # shape (E,)
        
#         # Applica il mascheramento: moltiplica i valori di A per i logit
#         masked_logits = values * edge_logits

#         # Applica la funzione sigmoid sui valori (dense)
#         att_edge = torch.sigmoid(masked_logits)

#         # Normalizzazione softmax per riga: per ogni nodo sorgente somma gli exp degli edge
#         exp_edge = torch.exp(att_edge)
#         row_sum = torch.zeros(M.size(0), device=M.device)
#         row_sum = row_sum.scatter_add(0, indices[0], exp_edge)
#         normalized_edge = exp_edge / row_sum[indices[0]]

#         # Ricostruisci la matrice sparsa di attenzione con torch.sparse_coo_tensor
#         att = torch.sparse_coo_tensor(indices, normalized_edge, A.shape)
#         return att



import torch
import torch.nn as nn
import torch.nn.functional as F
import args_gate as config  # Importa il file di configurazione

class GATE(nn.Module):
    def __init__(self, hidden_dims, lambda_, A):
        super(GATE, self).__init__()
        self.lambda_ = lambda_
        self.n_layers = len(hidden_dims) - 1

        self.A = A

        # Pesi per ciascun layer
        self.W = nn.ParameterList()
        for i in range(self.n_layers):
            weight = nn.Parameter(torch.Tensor(hidden_dims[i], hidden_dims[i+1]))
            nn.init.xavier_uniform_(weight)
            self.W.append(weight)

        # Parametri per il meccanismo di attenzione: v0 e v1 per ogni layer
        self.v0 = nn.ParameterList()
        self.v1 = nn.ParameterList()
        for i in range(self.n_layers):
            a0 = nn.Parameter(torch.Tensor(hidden_dims[i+1], 1))
            a1 = nn.Parameter(torch.Tensor(hidden_dims[i+1], 1))
            nn.init.xavier_uniform_(a0)
            nn.init.xavier_uniform_(a1)
            self.v0.append(a0)
            self.v1.append(a1)

        # Dizionario per salvare (opzionalmente) le matrici di attenzione per ogni layer
        self.C = {}

    def encode(self, X):
        """
        Funzione di encoding che calcola le rappresentazioni dei nodi.
        
        Args:
            X: tensor denso di shape (N, F_in), feature dei nodi.
        Returns:
            H: tensor denso di shape (N, F_out), rappresentazione finale dei nodi.
        """
        H = X
        self.C = {}
        # Encoder: trasformazione lineare seguita dalla propagazione tramite attenzione
        for layer in range(self.n_layers):
            H = torch.matmul(H, self.W[layer])
            C_layer = self.graph_attention_layer(self.A, H, self.v0[layer], self.v1[layer])
            self.C[layer] = C_layer
            H = torch.sparse.mm(C_layer, H)
        self.H = H  # rappresentazione finale dei nodi
        return H

    def decode(self, H):
        """
        Funzione di decoding che ricostruisce la matrice di adiacenza
        a partire dalle rappresentazioni dei nodi.
        
        Args:
            H: tensor denso di shape (N, F_out), rappresentazioni dei nodi.
        Returns:
            A_pred: tensor denso di shape (N, N), matrice di adiacenza ricostruita.
        """
        A_pred = torch.sigmoid(torch.matmul(H, H.t()))
        return A_pred

    def forward(self, X):
        """
        Forward pass del modello.
        
        Args:
            X: tensor denso di shape (N, F_in), feature dei nodi.
        Returns:
            A_pred: tensor denso di shape (N, N), matrice di adiacenza ricostruita.
        """
        H = self.encode(X)
        A_pred = self.decode(H)
        return A_pred

    def graph_attention_layer(self, A, M, v0, v1):
        """
        Calcola la matrice di attenzione usando A in formato sparso.
        
        Args:
          A: torch.sparse_coo_tensor, matrice di adiacenza normalizzata (N, N)
          M: rappresentazioni nodali (tensor denso di shape (N, F))
          v0, v1: parametri di attenzione (tensor di shape (F, 1))
        
        Returns:
          att: torch.sparse_coo_tensor, matrice di attenzione normalizzata (N, N)
        """
        # Estrai indici e valori dal tensore sparso A
        indices = A._indices()   # shape [2, E]
        values = A._values()     # shape [E]

        # Calcola i termini di attenzione per ciascun nodo
        f1 = torch.matmul(M, v0).squeeze(1)  # shape (N,)
        f2 = torch.matmul(M, v1).squeeze(1)  # shape (N,)

        # Per ogni arco (i, j) in A, calcola il logit: f1[i] + f2[j]
        edge_logits = f1[indices[0]] + f2[indices[1]]  # shape (E,)
        
        # Applica il mascheramento: moltiplica i valori di A per i logit
        masked_logits = values * edge_logits

        # Applica la funzione sigmoid sui valori (dense)
        att_edge = torch.sigmoid(masked_logits)

        # Normalizzazione softmax per riga: per ogni nodo sorgente somma gli exp degli edge
        exp_edge = torch.exp(att_edge)
        row_sum = torch.zeros(M.size(0), device=M.device)
        row_sum = row_sum.scatter_add(0, indices[0], exp_edge)
        normalized_edge = exp_edge / row_sum[indices[0]]

        # Ricostruisci la matrice sparsa di attenzione con torch.sparse_coo_tensor
        att = torch.sparse_coo_tensor(indices, normalized_edge, A.shape)
        return att



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MLPDecoder(nn.Module):
#     """Decoder corretto per la forma dell'output"""
#     def __init__(self, embed_dim):
#         super().__init__()
#         self.decoder = nn.Sequential(
#             nn.Linear(2 * embed_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )

#     def forward(self, H):
#         # Genera tutte le coppie possibili (i,j)
#         num_nodes = H.size(0)
#         row = torch.arange(num_nodes).repeat_interleave(num_nodes)
#         col = torch.arange(num_nodes).repeat(num_nodes)
        
#         h_pairs = torch.cat([H[row], H[col]], dim=1)
#         logits = self.decoder(h_pairs).squeeze()
        
#         return torch.sigmoid(logits).view(num_nodes, num_nodes)

# class GATE(nn.Module):
#     """Versione corretta con gestione della forma"""
#     def __init__(self, hidden_dims, lambda_, A, dropout_rate=0.5):
#         super(GATE, self).__init__()
#         self.lambda_ = lambda_
#         self.n_layers = len(hidden_dims) - 1
#         self.A = A
#         self.dropout = nn.Dropout(dropout_rate)

#         # Inizializzazione parametri
#         self.W = nn.ParameterList()
#         self.v0 = nn.ParameterList()
#         self.v1 = nn.ParameterList()
        
#         for i in range(self.n_layers):
#             # Layer weights
#             w = nn.Parameter(torch.Tensor(hidden_dims[i], hidden_dims[i+1]))
#             nn.init.xavier_uniform_(w)
#             self.W.append(w)
            
#             # Attention parameters
#             a0 = nn.Parameter(torch.Tensor(hidden_dims[i+1], 1))
#             a1 = nn.Parameter(torch.Tensor(hidden_dims[i+1], 1))
#             nn.init.xavier_uniform_(a0)
#             nn.init.xavier_uniform_(a1)
#             self.v0.append(a0)
#             self.v1.append(a1)

#         self.decoder = MLPDecoder(hidden_dims[-1])

#     def encode(self, X):
#         H = X
#         for layer in range(self.n_layers):
#             H = self.dropout(torch.matmul(H, self.W[layer]))
#             C = self.graph_attention_layer(self.A, H, self.v0[layer], self.v1[layer])
#             H = torch.sparse.mm(C, H)
#         return H
    
#     def decode(self, H):
#         return self.decoder(H)
    
#     def forward(self, X):
#         return self.decode(self.encode(X))
    
#     def graph_attention_layer(self, A, M, v0, v1):
#         indices = A._indices()
#         values = A._values()
        
#         # Compute attention logits
#         f1 = torch.matmul(M, v0).squeeze(1)
#         f2 = torch.matmul(M, v1).squeeze(1)
#         edge_logits = f1[indices[0]] + f2[indices[1]]
        
#         # Softmax normalization
#         exp_edge = torch.exp(edge_logits)
#         row_sum = torch.zeros(M.size(0), device=M.device).scatter_add(0, indices[0], exp_edge)
#         normalized_edge = exp_edge / (row_sum[indices[0]] + 1e-8)
        
#         return torch.sparse_coo_tensor(indices, normalized_edge, A.shape)