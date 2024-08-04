# python -m torch.distributed.launch --nproc_per_node=2 gcn_imdb_sentence_classification_ddp_train.py

# torchrun --nproc_per_node=2 beta_nlp/src/classification/gcn_imdb_sentence_classification_ddp_train.py

torchrun --standalone --nnodes=1 --nproc-per-node=2 beta_nlp/src/classification/gcn_imdb_sentence_classification_ddp_train.py
