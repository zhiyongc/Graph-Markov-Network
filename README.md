# GraphMarkovNetwork
> Graph Markov Network for Traffic Forecasting with Missing Data

### Introduction
This is the github repo for sharing the code for implementing the Graph Markov Network (GMN) proposed in [1]. The GMN is proposed to solve the traffic forecasting problems while the traffic data has missing values. The Graph Markov Model is designed based on the Graph Markov Process (GMP), which provides a new perspective to model the transition process of the spatial-temporal data. 

The idea of GMN is very simple and easy to be implemented. The structure of GMN is similar to the autoregressive model and recurrent neural networks. The difference is that GMN takes the spatial structure of the data (network-wide traffic states) as a graph and attempts to infer missing values from the values of neighboring nodes in the graph. The following figure demonstrates the GMP structure. The gray-colored nodes in the left demonstrate the nodes with missing values. Vectors on the right side represent the traffic states. The traffic states at time _t_ are numbered to match the graph and the vector. The future state (in red color) can be inferred from their neighbors at previous time steps.





For most details of the Graph Markov Network or the Graph Markov Process, you can easily refer to this \[[post](https://zhiyongcui.com/blog/2020/07/16/graph-markov-network.html)\] or refer to the paper \[[TR Part C](https://www.sciencedirect.com/science/article/pii/S0968090X20305866)\] or \[[arXiv](https://arxiv.org/abs/1912.05457)\].

### Data

### Usage

### Reference
[1] Cui, Zhiyong, Longfei Lin, Ziyuan Pu, and Yinhai Wang. "[Graph markov network for traffic forecasting with missing data.](https://www.sciencedirect.com/science/article/pii/S0968090X20305866)" Transportation Research Part C: Emerging Technologies 117 (2020): 102671. \[[arXiv](https://arxiv.org/abs/1912.05457)\]

### BibTeX
You are welcomed to cite the following paper if the source code is helpful. 
```
  @article{cui2020graph,
    title={Graph markov network for traffic forecasting with missing data},
    author={Cui, Zhiyong and Lin, Longfei and Pu, Ziyuan and Wang, Yinhai},
    journal={Transportation Research Part C: Emerging Technologies},
    volume={117},
    pages={102671},
    year={2020},
    publisher={Elsevier}
  }
```
