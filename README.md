# MAELP


This is the official implementation of the paper: A Modular Autoencoder Ensemble for Link Prediction in Heterogeneous Multilayer Networks


## Abstract

Link prediction is a core problem in network science and graph representation learning. Most existing link prediction approaches work on homogeneous networks characterized by a single type of node, edge, and node feature space. However, many real-world scenarios involve different entities and interaction modes and are better modeled through heterogeneous multilayer networks. These networks consist of layers that may differ in node types, edge semantics, and node feature spaces. In this setting, link prediction becomes much more complex because structural patterns and node features may vary across layers. This paper addresses this challenge by introducing Modular Autoencoder Ensemble for Link Prediction (MAELP). MAELP has three main components, namely: (i) a set of intralayer autoencoders, modeling layer-specific structures; (ii) an interlayer autoencoder, modeling dependencies among layers; and (iii)  a lightweight fusion autoencoder, synthesizing the resulting embeddings into a unified representation. MAELP ensures that each layer encodes its own features before integrating information from different layers. This preserves layer-specific information while discovering cross-layer patterns. The resulting node embeddings are richer, enabling higher-performance link prediction. Extensive experiments on several heterogeneous multilayer networks show consistent improvements over state-of-the-art approaches.


<img width="2529" height="1287" alt="Workflow" src="https://github.com/user-attachments/assets/05de1005-71a3-4c6a-8f9a-aaba13091b4a" />
