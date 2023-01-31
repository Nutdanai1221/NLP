# NLP
## Paper reading assignment
### 1. Graph Transformer Networks (NeurIPS, 2019)
|Problems| Limitation of existing Graph Neural Networks (GNNs) in handling heterogeneous graphs  |
|:------:|:-----|
|Related work| 1. Graph Neural Networks (GNNs) are machine learning models developed to perform various tasks on graphs. There are two approaches, spectral and non-spectral.<br> 2. Node classification with GNNs is used to predict a label for each node in a graph based on its structure and features  |
|Solution| Author porpose a new framework call Graph Transformer Networks(GTNs). The GTNs are designed to transform heterogeneous graphs into meta-path graphs, which are paths connecting different types of edges. The GTNs learn to generate these meta-path graphs and node representations |
|Result|Author compares the performance of different Graph Neural Network (GNN) models for node classification in heterogeneous graphs. The models compared are GCN, GAT, HAN, and the proposed GTN model. The results show that the GNN-based methods perform better than random walk-based network embedding methods.|
