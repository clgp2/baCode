#script to train models and use them to make predictions
import os
from pathlib import Path

base_path = Path(__file__).parent

src_lang = 'en'
trg_lang = 'ro'
bpe_size = 30000

name= "02-02-bicleaner-preprocessed"

#mkdir when dir does not exist yet 
path_yaml_config= (base_path / f"../models/transformer_{name}.yaml").resolve()
modeldir=(base_path / f"../models/").resolve()
datadir = (base_path / f"../data/02-preprocessed/02-02-bicleaner-preprocessed/").resolve()

config = """
name: "{name}_transformer"

data:
    src: "{source_language}"
    trg: "{target_language}"
    train: "{datadir}/train.bpe"
    dev: "{datadir}/dev.bpe"
    test: "{datadir}/test.bpe"
    level: "bpe"
    lowercase: False
    max_sent_length: 140
    src_vocab: "{datadir}/vocab.txt"
    trg_vocab: "{datadir}/vocab.txt"

testing:
    beam_size: 5
    alpha: 1.0

training:
    #load_model: "/home/bernadeta/models/enro_dcep_transformer_2GPU/best.ckpt"
    random_seed: 42
    optimizer: "adam"
    normalization: "tokens"
    adam_betas: [0.9, 0.999]
    scheduling: "plateau"
    patience: 8
    decrease_factor: 0.7
    loss: "crossentropy"
    learning_rate: 0.0002
    learning_rate_min: 0.00000001
    weight_decay: 0.0
    label_smoothing: 0.1
    batch_size: 4096
    batch_type: "token"
    eval_batch_size: 3600
    eval_batch_type: "token"
    batch_multiplier: 1
    early_stopping_metric: "ppl"
    epochs: 300
    validation_freq: 5000
    logging_freq: 100
    eval_metric: "bleu"
    model_dir: "{modeldir}/transformer_{name}"
    overwrite: True
    shuffle: True
    use_cuda: True
    max_output_length: 140
    print_valid_sents: [0, 1, 2, 3]
    keep_last_ckpts: 5
    save_latest_ckpt: True

model:
    initializer: "xavier"
    bias_initializer: "zeros"
    init_gain: 1.0
    embed_initializer: "xavier"
    embed_init_gain: 1.0
    tied_embeddings: True
    tied_softmax: True
    encoder:
        type: "transformer"
        num_layers: 6
        num_heads: 8
        embeddings:
            embedding_dim: 512
            scale: True
            dropout: 0.
        # typically ff_size = 4 x hidden_size
        hidden_size: 512
        ff_size: 2048
        dropout: 0.1
    decoder:
        type: "transformer"
        num_layers: 6
        num_heads: 8
        embeddings:
            embedding_dim: 512
            scale: True
            dropout: 0.
        # typically ff_size = 4 x hidden_size
        hidden_size: 512
        ff_size: 2048
        dropout: 0.1

""".format(name=name, source_language=src_lang, target_language=trg_lang,
           datadir=datadir, modeldir=modeldir)

with open("test".format(name=name),'w') as f:
    f.write(config)

#would use a shell script to submit to slurm
os.system(f"python -m joeynmt train {path_yaml_config}")