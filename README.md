# Welcome!

# Setting up env

In the DistSys Directory, run the install script to download all the required packages.
Next, set your $GOPATH to include the DistSys directory:

1. `cd DistSys`  
2. `bash install.sh`  
3. `export GOPATH=$PWD`  

# simpleBlockChain

In the DistSys Directory run the following:

`go run *.go -i node_id -t total_nodes -d dataset`  

For example,  
`sudo go run *.go -i 0 -t 4 -d creditcard`  

Runs a node with Id 0 in a network of 4 nodes each with a part of the credit card dataset  
Node Ids start from 0 upto (numberOfNodes - 1)
  
  
# Biscotti: machine learning on the blockchain

Biscotti is a fully decentralized peer-to-peer system for multi-party machine learning (ML). Peers participate in the learning process by contributing (possibly private) datasets and coordinating in training a global model of the union of their datasets. Biscotti uses blockchain primitives for coordination between peers and relies on differential privacy and cryptography techniques to provide privacy and security guarantees to peers.

For more details about Biscotti's design, see our [Arxiv paper](https://arxiv.org/abs/1811.09904).

# Dependencies

We use the the go-python library for interfacing between the distributed system code in Go and the ML logic in Python. Unfortunately, Go-python doesn't support Python versions > 2.7.12. Please ensure that your default OS Python version is 2.7.12.



