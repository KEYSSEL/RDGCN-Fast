#!/bin/bash


# data_dir="data/DBP15K/zh_en" chemin vers le dossier du datasets
# save="resultats_rdgcn/zh_en" chemin vers le dossier de sauvegarde des résultats

python3 run.py  --log rdgcn_fast_test \
                --data_dir "./data/EN_FR_100K" \
                --rate 0.2 \
                --use_attr \
                --attr_alpha 0.2 \
                --val 0.1\
                --early \
                --epoch 600 \
                --check 3 \
                --update 3 \
                --train_batch_size 500000 \
                --encoder "rdgcn" \
                --hiddens "300,300,300" \
                --decoder "align" \
                --swap \
                --sampling "N" \
                --k "25" \
                --margin "1.0" \
                --alpha "1.0" \
                --feat_drop 0.0 \
                --lr 0.001 \
                --train_dist "manhattan" \
                --test_dist "euclidean" \
                --save "./results_test/EN_FR_100K_rdgcn_fast" \
                --patience 3 \
                --use_fasttext
