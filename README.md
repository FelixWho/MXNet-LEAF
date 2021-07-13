# Instructions for Use

## Requirements for Talwalkar Lab's LEAF Dataset

    cd FederatedLearningFramework
    pip install -r requirements.txt
    pip3 install -r requirements.txt # install requirements for both Python 2 and 3
    git clone https://github.com/TalwalkarLab/leaf.git

## Other Requirements

    mxnet
    gluonnlp
    ...

## Workflow

1. Install requirements
2. Clone the LEAF repository
3. Install the dataset of choice using LEAF instructions
4. If using dataset in [REDDIT, SENT140], run `/embeddings/get_embs.sh` to install GloVe embedding
5. Run `test_byz_p.py` e.x.

    ```
    python3 test_byz_p.py --dataset FashionMNIST
    ```

Note, LEAF dataset names are assumed to be completely capitalized to avoid confusion. The LEAF dataset names are as follows:

    FEMNIST
    CELEBA
    SENT140
    SHAKESPEARE
    REDDIT # in progress
    SYNTHETIC # no plans to implement

