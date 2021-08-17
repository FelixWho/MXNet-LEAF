# MXNet-LEAF

The federated learning paradigm was recently introduced and is now gaining traction as a means of training across many users' devices. Prior to this repository, there were limited resources to build and test federated learning models in the MXNet environment. MXNet-LEAF changes that.

# Instructions for Use

## Requirements for Talwalkar Lab's LEAF Dataset

    cd MXNet-LEAF
    pip install -r requirements.txt
    pip3 install -r requirements.txt # install requirements for both Python 2 and 3, just in case
    git clone https://github.com/TalwalkarLab/leaf.git

Note, we are not yet done setting up LEAF. Each LEAF dataset has its own instructions, and I recommend visiting https://github.com/TalwalkarLab/leaf to figure out how to set them up. For example, if you are interesting in using the Femnist dataset, just like the instructions ask, go to the Femnist folder and run `./preprocess.sh -s niid --sf 0.05 -k 0 -t sample`.

Additionally worth noting, the setup for Reddit dataset is different from the rest--be mindful!

## Other Requirements

    pip3 install mxnet
    pip3 install gluonnlp

## Workflow

1. Install requirements
2. Clone the LEAF repository
3. Install the dataset of choice using LEAF instructions
4. If using dataset Sent140, run `/embeddings/get_embs.sh` to install GloVe embedding
5. Run `main.py`. Examples included below

    ```
    python3 main.py --dataset FEMNIST --lr 0.1 --nbyz 0 --p 0 --server_pc 0 --aggregation simple --batch_size 3
    python3 main.py --dataset FEMNIST --lr 0.005 --nbyz 0 --p 0 --server_pc 0 --aggregation trim

    python3 main.py --dataset CELEBA --lr 0.001 --nbyz 0 --p 0 --server_pc 0 --aggregation simple --batch_size 3
    python3 main.py --dataset CELEBA --lr 0.001 --nbyz 0 --p 0 --server_pc 0 --aggregation median --batch_size 3

    python3 main.py --dataset SHAKESPEARE --lr 0.1 --nbyz 0 --p 0 --server_pc 0 --aggregation simple

    python3 main.py --dataset SENT140 --lr 0.1 --nbyz 0 --p 0 --server_pc 0 --aggregation simple --batch_size 2

    python3 main.py --dataset REDDIT --lr 7 --nbyz 0 --p 0 --server_pc 0 --aggregation simple --batch_size 3
    ```

Note, LEAF dataset names are to be completely capitalized to avoid confusion. The LEAF dataset names are as follows:

    FEMNIST
    CELEBA
    SENT140
    SHAKESPEARE
    REDDIT
    SYNTHETIC # no plans to implement

# Extra

[mxnet_from_source.md](https://github.com/FelixWho/MXNet-LEAF/blob/master/mxnet_from_source.md) contains instructions for installing MXNet from source without root/sudo permission
