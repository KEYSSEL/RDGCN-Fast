#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import time
import argparse
import os
import copy
import random
import math
import gc
import numpy as np
import json
import re
from gensim.models import FastText
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from load_data import *
from models import *
from utils import *
from semi_utils import bootstrapping, boot_update_triple, bootstrapping_gpu

from torch.utils.tensorboard import SummaryWriter
import distutils.version
import sys

# comme LooseVersion manque, on le récupère depuis setuptools
try:
    from setuptools._distutils import version
    distutils.version = version
except ImportError:
    pass

from torch.utils.tensorboard import SummaryWriter
import logging

def initialize_with_glove(id2ins_dict, glove_path, embed_dim=100):
    glove_dict = {}
    print(f"Chargement de GloVe depuis {glove_path}...")
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_dict[word] = vector

    num_entities = max(int(k) for k in id2ins_dict.keys()) + 1
    initial_weights = torch.empty(num_entities, embed_dim)
    torch.nn.init.xavier_normal_(initial_weights)
    
    found_count = 0
    for idx, uri in id2ins_dict.items():
        name = uri.split('/')[-1].split(':')[-1].lower()
        words = re.sub(r'[_:\-\.]', ' ', name).split()
        
        vectors = [glove_dict[w] for w in words if w in glove_dict]
        
        if vectors:
            initial_weights[idx] = torch.from_numpy(np.mean(vectors, axis=0))
            found_count += 1
            
    print(f"✅ GloVe : {found_count}/{num_entities} entités initialisées ({found_count/num_entities:.2%})")
    return initial_weights

def get_fasttext_embeddings(id2ins_dict, path=False, embed_dim=300):
    sentences = []
    id_pattern = re.compile(r'^q\d+$')
    for uri in id2ins_dict.values():
        name = uri.split('/')[-1].split(':')[-1]
        if id_pattern.match(name):
            continue
        words = re.sub(r'[_:\-\.]', ' ', name).lower().split()
        if words:
            sentences.append(words)

    model = FastText(sentences=sentences, vector_size=embed_dim, window=3, min_count=1, epochs=200)

    if path:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        model.save(path)
        print(f"✅ Modèle FastText entraîné et sauvegardé dans: {path}")
        
    num_entities = max(int(k) for k in id2ins_dict.keys()) + 1
    initial_weights = torch.empty(num_entities, embed_dim)
    
    torch.nn.init.xavier_normal_(initial_weights)

    found_count = 0
    for idx, uri in id2ins_dict.items():
        name = uri.split('/')[-1].split(':')[-1].lower()
        words = re.sub(r'[_:\-\.]', ' ', name).split() 
        vectors = [model.wv[w] for w in words if w in model.wv]
        if vectors:
            initial_weights[idx] = torch.from_numpy(np.mean(vectors, axis=0))
            found_count += 1
    initial_weights = F.normalize(initial_weights, p=2, dim=1)
    print(f"🎯 Initialisation terminée : {found_count}/{num_entities} entités couvertes par FastText.")
    return initial_weights

def get_fasttext_attr_embeddings(id2attr_dict, model_path, embed_dim=200):
    """
    Extrait la sémantique des attributs et génère les embeddings initiaux
    en utilisant le modèle FastText déjà entraîné lors de l'initialisation des entités.
    """
    print(f"🌍 [FASTTEXT-ATTR] Extraction sémantique pour {len(id2attr_dict)} attributs...")
    
    # Étape 1 : Nettoyage et préparation
    sentences = []
    attr_words_list = [] 
    
    for idx, uri in id2attr_dict.items():
        name = uri.split('/')[-1].split('#')[-1] 
        
        words = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        words = re.sub(r'[_:\-\.]', ' ', words).lower().split()
        
        attr_words_list.append((idx, words))
        if words:
            sentences.append(words)

    # Étape 2 : Chargement du modèle
    if model_path and os.path.exists(model_path):
        print(f"   -> Chargement du modèle FastText existant depuis {model_path}")
        model = FastText.load(model_path)
        # On pourrait faire un update du vocabulaire ici si nécessaire (model.build_vocab(sentences, update=True) puis model.train(...)),
        # mais on suppose que le modèle entraîné sur les entités est assez riche.
    else:
        print(f"   ==> Entraînement d'un nouveau modèle FastText (ce n'est pas censé arriver si les entités l'ont déjà fait)")
        model = FastText(sentences=sentences, vector_size=embed_dim, window=3, min_count=1, epochs=200)

    # Étape 3 : Création de la matrice d'embeddings
    num_attrs = max(int(k) for k in id2attr_dict.keys()) + 1
    initial_weights = torch.empty(num_attrs, embed_dim)
    torch.nn.init.xavier_normal_(initial_weights)

    found_count = 0
    for idx, words in attr_words_list:
        vectors = [model.wv[w] for w in words if w in model.wv]
        if vectors:
            initial_weights[idx] = torch.from_numpy(np.mean(vectors, axis=0))
            found_count += 1
    initial_weights = F.normalize(initial_weights, p=2, dim=1)
    print(f"🎯 Attributs FastText : {found_count}/{num_attrs} initialisés avec succès.")
    return initial_weights


class Experiment:
    def __init__(self, args):
        self.save = args.save
        self.save_prefix = "%s_%s" % (args.data_dir.split("/")[-1], args.log)
        
        self.hiddens = list(map(int, args.hiddens.split(",")))
        self.heads = list(map(int, args.heads.split(",")))
        
        self.args = args
        self.args.encoder = args.encoder.lower()
        self.args.decoder = args.decoder.lower().split(",")
        self.args.sampling = args.sampling.split(",")        
        self.args.k = list(map(int, args.k.split(",")))
        self.args.margin = [float(x) if "-" not in x else list(map(float, x.split("-"))) for x in args.margin.split(",")]
        self.args.alpha = list(map(float, args.alpha.split(",")))
        assert len(self.args.decoder) >= 1
        assert len(self.args.decoder) == len(self.args.sampling) and \
                len(self.args.sampling) == len(self.args.k) and \
                len(self.args.k) == len(self.args.alpha)

        self.cached_sample = {}
        self.best_result = ()


    def evaluate(self, it, test, ins_emb, is_test, mapping_emb=None, initial_train_idx=None, initial_val_idx=None):
        t_test = time.time()
        top_k = [1, 3, 5, 10]
        test_left = test[:, 0]
        test_right = test[:, 1]
        ill_idx_np = np.array(d.ill_idx)
        
        train_left  = np.array(initial_train_idx)[:, 0]
        train_right = np.array(initial_train_idx)[:, 1]

        if initial_val_idx is not None and len(initial_val_idx) > 0:
            val_left  = np.array(initial_val_idx)[:, 0]
            val_right = np.array(initial_val_idx)[:, 1]
            exclude_left  = np.concatenate([test_left, train_left, val_left])
            exclude_right = np.concatenate([test_right, train_right, val_right])
        else:
            exclude_left = np.concatenate([test_left, train_left])
            exclude_right = np.concatenate([test_right, train_right])

        other_left = np.setdiff1d(ill_idx_np[:, 0], exclude_left)
        other_right = np.setdiff1d(ill_idx_np[:, 1], exclude_right)
        
        full_left = np.concatenate([test_left, other_left])
        full_right = np.concatenate([test_right, other_right])
        
        if mapping_emb is not None:
            logger.info("using mapping")
            left_emb = mapping_emb[full_left]
        else:
            left_emb = ins_emb[full_left]
            
        right_emb = ins_emb[full_right]
        left_tensor = torch.FloatTensor(left_emb).to(device)
        right_tensor = torch.FloatTensor(right_emb).to(device)
        
        left_tensor = F.normalize(left_tensor, p=2, dim=1)
        right_tensor = F.normalize(right_tensor, p=2, dim=1)
    
        CHUNK_SIZE = 512  # nombre de lignes traitées à la fois, ajuster si OOM

        test_len = len(test)
        ranks_l2r = torch.zeros(test_len, device='cpu')
        ranks_r2l = torch.zeros(test_len, device='cpu')
        
        # Avant la boucle L2R — calcul de nearest2 pour tous les right
        if self.args.csls > 0:
            nearest2_list = []
            for start in range(0, test_len, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, test_len)
                chunk_right = right_tensor[start:end]
                if self.args.test_dist in ['cosine', 'inner']:
                    d_tmp = torch.mm(chunk_right, left_tensor.t())
                else:
                    d_tmp = -torch.cdist(chunk_right, left_tensor, p=2)
                val2, _ = torch.topk(d_tmp, self.args.csls + 1, dim=1)
                nearest2_list.append(val2[:, 1:].mean(dim=1))
                del d_tmp
            nearest2_global = torch.cat(nearest2_list).unsqueeze(1) 

        # --- L2R : pour chaque chunk de left, on calcule sa ligne de distance ---
        for start in range(0, test_len, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, test_len)
            chunk_left = left_tensor[start:end]           

            if self.args.test_dist in ['cosine', 'inner']:
                chunk_dist = torch.mm(chunk_left, right_tensor.t())
            elif self.args.test_dist == 'euclidean':
                chunk_dist = -torch.cdist(chunk_left, right_tensor, p=2)
            elif self.args.test_dist == 'manhattan':
                chunk_dist = -torch.cdist(chunk_left, right_tensor, p=1)
            else:
                chunk_dist = torch.mm(chunk_left, right_tensor.t())

            # CSLS sur le chunk
            if self.args.csls > 0:
                k_csls = self.args.csls
                val1, _ = torch.topk(chunk_dist, k_csls + 1, dim=1)
                nearest1 = val1[:, 1:].mean(dim=1, keepdim=True)
                chunk_dist = 2 * chunk_dist - nearest1 - nearest2_global[start:end]

            # Rang de la bonne entité 
            targets = torch.arange(start, end, device=chunk_dist.device).unsqueeze(1)
            _, sorted_chunk = torch.sort(chunk_dist, dim=1, descending=True)
            chunk_ranks = (sorted_chunk == targets).nonzero(as_tuple=True)[1].float() + 1.0
            ranks_l2r[start:end] = chunk_ranks.cpu()
            del chunk_dist, sorted_chunk

        # --- R2L : pour chaque chunk de right ---
        for start in range(0, test_len, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, test_len)
            chunk_right = right_tensor[start:end]

            if self.args.test_dist in ['cosine', 'inner']:
                chunk_dist = torch.mm(chunk_right, left_tensor.t())
            elif self.args.test_dist == 'euclidean':
                chunk_dist = -torch.cdist(chunk_right, left_tensor, p=2)
            elif self.args.test_dist == 'manhattan':
                chunk_dist = -torch.cdist(chunk_right, left_tensor, p=1)
            else:
                chunk_dist = torch.mm(chunk_right, left_tensor.t())

            targets = torch.arange(start, end, device=chunk_dist.device).unsqueeze(1)
            _, sorted_chunk = torch.sort(chunk_dist, dim=1, descending=True)
            chunk_ranks = (sorted_chunk == targets).nonzero(as_tuple=True)[1].float() + 1.0
            ranks_r2l[start:end] = chunk_ranks.cpu()
            del chunk_dist, sorted_chunk

        torch.cuda.empty_cache()

        mean_l2r = ranks_l2r.mean().item()
        mrr_l2r = (1.0 / ranks_l2r).mean().item()
        mean_r2l = ranks_r2l.mean().item()
        mrr_r2l = (1.0 / ranks_r2l).mean().item()

        acc_l2r = np.array([np.round((ranks_l2r <= k).float().mean().item(), 4) for k in top_k])
        acc_r2l = np.array([np.round((ranks_r2l <= k).float().mean().item(), 4) for k in top_k])
        
        logger.info("l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s ".format(top_k, acc_l2r.tolist(), mean_l2r, mrr_l2r, time.time() - t_test))
        logger.info("r2l: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s \n".format(top_k, acc_r2l.tolist(), mean_r2l, mrr_r2l, time.time() - t_test))
        if is_test:
            for i, k in enumerate(top_k):
                writer.add_scalar("l2r_HitsAt{}".format(k), acc_l2r[i], it)
                writer.add_scalar("r2l_HitsAt{}".format(k), acc_r2l[i], it)
            writer.add_scalar("l2r_MeanRank", mean_l2r, it)
            writer.add_scalar("l2r_MeanReciprocalRank", mrr_l2r, it)
            writer.add_scalar("r2l_MeanRank", mean_r2l, it)
            writer.add_scalar("r2l_MeanReciprocalRank", mrr_r2l, it)
        return (acc_l2r, mean_l2r, mrr_l2r, acc_r2l, mean_r2l, mrr_r2l)


    def init_emb(self):
        e_scale, r_scale = 1, 1
        if not self.args.encoder:
            if self.args.decoder == ["rotate"]:
                r_scale = r_scale / 2
            elif self.args.decoder == ["hake"]:
                r_scale = (r_scale / 2) * 3
            elif self.args.decoder == ["transh"]:
                r_scale = r_scale * 2
            elif self.args.decoder == ["transr"]:
                r_scale = self.hiddens[0] + 1

        # --- INITIALISATION RDGCN (une possibilité serait d'utiliser les Vecteurs pré-entraînés GCN) ---
        if self.args.encoder.lower() == "rdgcn" and not getattr(self.args, 'use_glove', False) and not getattr(self.args, 'use_fasttext', False) and self.args.pre != "":
            all_files = os.listdir(self.args.pre)
                
            ins_candidates = [f for f in all_files if f.endswith('ins.npy')]
            rel_candidates = [f for f in all_files if f.endswith('rel.npy')]
            
            if not ins_candidates or not rel_candidates:
                raise FileNotFoundError(f"Aucun fichier se terminant par 'ins.npy' ou 'rel.npy' trouvé dans {self.args.pre}")
            
            ins_file = sorted(ins_candidates)[-1]
            rel_file = sorted(rel_candidates)[-1]
            
            ins_path = os.path.join(self.args.pre, ins_file)
            rel_path = os.path.join(self.args.pre, rel_file)
            print("💡 [RDGCN Mode] Chargement des embeddings pré-entraînés GCN...")
            ins_emb_np = np.load(ins_path)
            rel_emb_np = np.load(rel_path)
            ins_tensor = torch.FloatTensor(ins_emb_np)
            rel_tensor = torch.FloatTensor(rel_emb_np)

            # Padding relations
            target_rel_size = d.rel_num + 5
            current_rel_size = rel_tensor.size(0)
            if target_rel_size > current_rel_size:
                padding_tensor = torch.zeros(target_rel_size - current_rel_size, rel_tensor.size(1))
                torch.nn.init.xavier_normal_(padding_tensor)
                rel_tensor = torch.cat([rel_tensor, padding_tensor], dim=0)

            self.ins_embeddings = torch.nn.Embedding.from_pretrained(ins_tensor, freeze=False).to(device)
            self.rel_embeddings = torch.nn.Embedding.from_pretrained(rel_tensor, freeze=False).to(device)
            
        # --- OPTION FASTEXT (N DIMENSIONS) ---
        elif getattr(self.args, 'use_fasttext', False):
            print(f"🌍 [FASTTEXT] Initialisation sémantique (dim: {self.hiddens[0]})")
            ft_model_path = os.path.join(self.save, "uri_fasttext.model") if self.save else False
            ins_weights = get_fasttext_embeddings(d.id2ins_dict, ft_model_path, embed_dim=self.hiddens[0])
            
            self.ins_embeddings = torch.nn.Embedding.from_pretrained(ins_weights, freeze=False).to(device)
            self.rel_embeddings = torch.nn.Embedding(d.rel_num + 5, int(self.hiddens[0] * r_scale)).to(device)
            nn.init.xavier_normal_(self.rel_embeddings.weight)


        # --- OPTION GLOVE (N DIMENSIONS) ---
        elif getattr(self.args, 'use_glove', False):
            print(f"🌍 [GloVe Mode] Initialisation sémantique (dim: {self.hiddens[0]}) depuis {self.args.glove_path}")
            
            ins_weights = initialize_with_glove(d.id2ins_dict, self.args.glove_path, embed_dim=self.hiddens[0])
            
            self.ins_embeddings = torch.nn.Embedding.from_pretrained(ins_weights, freeze=False).to(device)
            self.rel_embeddings = torch.nn.Embedding(d.rel_num + 5, int(self.hiddens[0] * r_scale)).to(device)
            nn.init.xavier_normal_(self.rel_embeddings.weight)

        # --- INITIALISATION NORMALE (XAVIER / ALÉATOIRE) ---
        else:
            self.ins_embeddings = torch.nn.Embedding(d.ins_num, self.hiddens[0] * e_scale).to(device)
            self.rel_embeddings = torch.nn.Embedding(d.rel_num + 5, int(self.hiddens[0] * r_scale)).to(device)

        if self.args.decoder == ["rotate"] or self.args.decoder == ["hake"]:
            ins_range = (self.args.margin[0] + 2.0) / float(self.hiddens[0] * e_scale)
            nn.init.uniform_(tensor=self.ins_embeddings.weight, a=-ins_range, b=ins_range)
            rel_range = (self.args.margin[0] + 2.0) / float(self.hiddens[0] * r_scale)
            nn.init.uniform_(tensor=self.rel_embeddings.weight, a=-rel_range, b=rel_range)
            if self.args.decoder == ["hake"]:
                r_dim = int(self.hiddens[0] / 2)
                nn.init.ones_(tensor=self.rel_embeddings.weight[:, r_dim : 2*r_dim])
                nn.init.zeros_(tensor=self.rel_embeddings.weight[:, 2*r_dim : 3*r_dim]) 
        elif self.args.encoder.lower() != "rdgcn" and not getattr(self.args, 'use_glove', False):
            nn.init.xavier_normal_(self.ins_embeddings.weight)
            nn.init.xavier_normal_(self.rel_embeddings.weight)

        # --- OPTION ATTRIBUTS (Couche apprenable dynamique) ---
        if getattr(self.args, 'use_attr', False):
            if d.attr_num > 0:
                print(f"🌟 [Attr Mode] Création d'une couche d'attributs apprenable ({d.attr_num} attributs)...")
                if getattr(self.args, 'use_fasttext', False) and hasattr(d, 'id2attr_dict'):
                    ft_model_path = os.path.join(self.save, "uri_fasttext.model") if self.save else False
                    attr_weights = get_fasttext_attr_embeddings(d.id2attr_dict, ft_model_path, embed_dim=self.hiddens[0])
                    self.attr_embeddings = torch.nn.Embedding.from_pretrained(attr_weights, freeze=False).to(device)
                else:
                    self.attr_embeddings = torch.nn.Embedding(d.attr_num, self.hiddens[0]).to(device)
                    nn.init.xavier_normal_(self.attr_embeddings.weight)
                
                # Création d'une matrice creuse pour fusionner les attributs rapidement à chaque époque
                indices = []
                values = []
                enriched_count = 0
                for i in range(d.ins_num):
                    if i in d.ent2attrs and len(d.ent2attrs[i]) > 0:
                        attrs = d.ent2attrs[i]
                        weight = 1.0 / len(attrs)
                        for attr in attrs:
                            indices.append([i, attr])
                            values.append(weight)
                        enriched_count += 1
                
                if indices:
                    indices_t = torch.LongTensor(indices).t().to(device)
                    values_t = torch.FloatTensor(values).to(device)
                    self.attr_sparse_mat = torch.sparse_coo_tensor(indices_t, values_t, (d.ins_num, d.attr_num)).to(device)
                else:
                    self.attr_sparse_mat = None
                    
                print(f"✅ Attributs configurés pour {enriched_count}/{d.ins_num} entités.")
            else:
                print("⚠️ [Attr Mode] Ignoré : Aucun attribut trouvé dans ce jeu de données.")

        if any(x in self.args.decoder for x in ["alignea", "mtranse_align", "transedge"]):
            self.ins_embeddings.weight.data = F.normalize(self.ins_embeddings.weight, p=2, dim=1)
            self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight, p=2, dim=1)

        self.enh_ins_emb = self.ins_embeddings.weight.cpu().detach().numpy()
        self.mapping_ins_emb = None

    # Fonction pour combiner Structure & Attributs
    def get_input_embeddings(self):
        ins_emb = self.ins_embeddings.weight
        if getattr(self.args, 'use_attr', False) and hasattr(self, 'attr_sparse_mat') and self.attr_sparse_mat is not None:
            # On multiplie la matrice creuse par les vecteurs d'attributs pour les attribuer aux bonnes entités
            attr_features = torch.sparse.mm(self.attr_sparse_mat, self.attr_embeddings.weight)
            attr_features = F.normalize(attr_features, p=2, dim=1)
            
            alpha = getattr(self.args, 'attr_alpha', 0.2) 
            x_input = ins_emb + (alpha * attr_features)
            
            # On renormalise le tout pour que la couche d'attention reste stable
            return F.normalize(x_input, p=2, dim=1)
        return ins_emb


    def train_and_eval(self):
        self.init_emb()
        top_k = [1, 3, 5, 10]
        graph_encoder = None
        if self.args.encoder:
            graph_encoder = Encoder(self.args.encoder, self.hiddens, self.heads+[1], activation=F.elu, feat_drop=self.args.feat_drop, attn_drop=self.args.attn_drop, negative_slope=0.2, bias=False, rel_num=d.rel_num).to(device)
            logger.info(graph_encoder)
        knowledge_decoder = []
        for idx, decoder_name in enumerate(self.args.decoder):
            knowledge_decoder.append(Decoder(decoder_name, params={
                "e_num": d.ins_num,
                "r_num": d.rel_num,
                "dim": self.hiddens[-1],
                "feat_drop": self.args.feat_drop,
                "train_dist": self.args.train_dist,
                "sampling": self.args.sampling[idx],
                "k": self.args.k[idx],
                "margin": self.args.margin[idx],
                "alpha": self.args.alpha[idx],
                "boot": self.args.bootstrap,
            }).to(device))
        logger.info(knowledge_decoder)

        base_params = [p for p in [self.ins_embeddings.weight, self.rel_embeddings.weight] if p.requires_grad]
        
        if getattr(self.args, 'use_attr', False) and hasattr(self, 'attr_embeddings'):
            base_params.append(self.attr_embeddings.weight)
            
        decoder_params = [p for k_d in knowledge_decoder for p in k_d.parameters()]        
        encoder_base_params = list(graph_encoder.parameters()) if self.args.encoder else []

        if self.args.encoder.lower() == "rdgcn":
            param_groups = []
            if len(base_params) > 0:
                param_groups.append({'params': base_params, 'lr': self.args.lr})
            param_groups.append({'params': decoder_params, 'lr': self.args.lr})
            if len(encoder_base_params) > 0:
                param_groups.append({'params': encoder_base_params, 'lr': self.args.lr})
            
            opt = optim.Adam(param_groups, weight_decay=self.args.wd)

        else:
            params = base_params + decoder_params + encoder_base_params
            opt = optim.Adagrad(params, lr=self.args.lr, weight_decay=self.args.wd)
            
        # -------------------------------------------------

        if self.args.dr:
            scheduler = optim.lr_scheduler.ExponentialLR(opt, self.args.dr)
            
        logger.info("="*50)
        logger.info(f"DÉTAILS DES PARAMÈTRES ({self.args.encoder.upper()})")
        logger.info("-"*50)

        categories = {
            "Base (Embeddings)": base_params,
            "Decoder": decoder_params,
            "Encoder": encoder_base_params
        }

        for cat_name, p_list in categories.items():
            if len(p_list) > 0:
                logger.info(f"--- {cat_name} ---")
                for i, p in enumerate(p_list):
                    mem_mb = (p.numel() * 4) / (1024**2)
                    logger.info(f"  [{i}] Shape: {str(list(p.shape)):<15} | Elements: {p.numel():<10} | Device: {p.device} | Mem: {mem_mb:.2f}MB")
        all_params_for_log = base_params + decoder_params + encoder_base_params
        logger.info("-"*50)
        logger.info(f"Utilisation du périphérique : {device}")
        total_elements = sum(p.numel() for p in all_params_for_log)
        total_mem = (total_elements * 4) / (1024**2)
        logger.info(f"TOTAL: {len(all_params_for_log)} tenseurs | {total_elements:,} paramètres | ~{total_mem:.2f} MB en RAM/VRAM")
        logger.info("="*50)

        logger.info(opt)
        # -------------------------------------------------            
        
        # Train
        logger.info("Start training...")
        best_emb = None
        best_mapping = None
        best_rel_emb = None
        best_ins_emb_weights = None
        best_attr_emb_weights = None 
        self.initial_train_idx = list(d.ill_train_idx)
        self.initial_val_idx   = list(d.ill_val_idx) if d.ill_val_idx is not None and len(d.ill_val_idx) > 0 else []
        for it in range(0, self.args.epoch):

            for idx, k_d in enumerate(knowledge_decoder):
                if (k_d.name == "align" and len(d.ill_train_idx) == 0):
                    continue
                t_ = time.time()
                if k_d.print_name.startswith("["):  
                    loss = self.train_1_epoch(it, opt, None, k_d, d.ins_G_edges_idx, d.triple_idx, d.ill_train_idx, [d.kg1_ins_ids, d.kg2_ins_ids], d.boot_triple_idx, d.boot_pair_dix, self.ins_embeddings.weight, self.rel_embeddings.weight)
                else:               
                    loss = self.train_1_epoch(it, opt, graph_encoder, k_d, d.ins_G_edges_idx, d.triple_idx, d.ill_train_idx, [d.kg1_ins_ids, d.kg2_ins_ids], d.boot_triple_idx, d.boot_pair_dix, self.ins_embeddings.weight, self.rel_embeddings.weight)
                if hasattr(k_d, "mapping"):
                    self.mapping_ins_emb = k_d.mapping(self.ins_embeddings.weight).cpu().detach().numpy()
                loss_name = "loss_" + k_d.print_name.replace("[", "_").replace("]", "_")
                writer.add_scalar(loss_name, loss, it)
                logger.info("epoch: %d\t%s: %.8f\ttime: %ds" % (it, loss_name, loss, int(time.time()-t_)) )

            if self.args.dr:
                scheduler.step()

            if self.args.bootstrap and it >= self.args.start_bp and (it + 1) % self.args.update == 0:
                with torch.no_grad():
                    if graph_encoder and graph_encoder.name == "naea":
                        beta = self.args.margin[-1]
                        emb = beta * self.enh_ins_emb + (1 - beta) * self.get_input_embeddings().cpu().detach().numpy()
                    else:
                        emb = self.enh_ins_emb
                     
                    d.labeled_alignment, A, B = bootstrapping(ref_sim_mat=sim(emb[d.ill_test_idx[:, 0]], emb[d.ill_test_idx[:, 1]], metric=self.args.test_dist, normalize=True, csls_k=0), 
                                                        ref_ent1=d.ill_test_idx[:, 0].tolist(), 
                                                        ref_ent2=d.ill_test_idx[:, 1].tolist(), 
                                                        labeled_alignment=d.labeled_alignment, th= self.args.threshold,       
                                                        top_k=10, is_edit=False)
          
                    
                if d.labeled_alignment:
                    d.boot_triple_idx = boot_update_triple(A, B, d.triple_idx)
                    d.boot_pair_dix = [(A[i], B[i])for i in range(len(A))]
                    if self.args.encoder.lower() in ["naea", "rdgcn"]:
                        d.ins_G_edges_idx, d.ins_G_values_idx, d.r_ij_idx = d.gen_sparse_graph_from_triples(d.triple_idx + d.boot_triple_idx, d.ins_num, with_r=True)
                    
                    if self.args.encoder.lower() == "rdgcn":
                        d.dual_edges_idx, d.dual_edges_weight = d.gen_dual_graph(d.triple_idx + d.boot_triple_idx, d.rel_num)
                    logger.info("Bootstrapping: + " + str(len(A)) + " ills, " + str(len(d.boot_triple_idx)) + " triples.")

            # Evaluate
            if (it + 1) % self.args.check == 0:
                logger.info("Start validating...")
                with torch.no_grad():
                    if graph_encoder and graph_encoder.name == "naea":
                        beta = self.args.margin[-1]
                        emb = beta * self.enh_ins_emb + (1 - beta) * self.get_input_embeddings().cpu().detach().numpy()
                    else:
                        emb = self.enh_ins_emb
                    if len(d.ill_val_idx) > 0:
                        print("-------JEU DE VALIDATION--------")
                        result = self.evaluate(it, d.ill_val_idx, emb, False, self.mapping_ins_emb, self.initial_train_idx, self.initial_val_idx)
                        print("-------JEU DE TEST--------")
                        self.evaluate(it, d.ill_test_idx, emb, True, self.mapping_ins_emb, self.initial_train_idx, self.initial_val_idx)
                    else:
                        result = self.evaluate(it, d.ill_test_idx, emb, True, self.mapping_ins_emb, self.initial_train_idx, self.initial_val_idx)

                # Early Stop
                if not hasattr(self, 'patience_counter'):
                    self.patience_counter = 0
                if not hasattr(self, 'previous_score'):
                    self.previous_score = 0.0
                
                if self.args.early and len(self.best_result) != 0:
                    current_score = result[0][0]
                    
                    if current_score > self.previous_score + 5*1e-4:  
                        self.patience_counter = 0
                        logger.info(f"--- [Patience] Remontée détectée ({current_score:.4f} > {self.previous_score:.4f}) -> Compteur remis à 0 ---")
                    else:
                        self.patience_counter += 1
                        logger.info(f"--- [Patience] Baisse ou stagnation détectée ({self.patience_counter}/{self.args.patience}) ---")
                        
                    self.previous_score = current_score

                    if current_score >= self.best_result[0][0]:  
                        self.best_result = result
                        best_emb = copy.deepcopy(emb)
                        best_rel_emb = copy.deepcopy(self.rel_embeddings.weight.cpu().detach().numpy())
                        best_ins_emb_weights = copy.deepcopy(self.ins_embeddings.weight.cpu().detach().numpy())
                        best_attr_emb_weights = copy.deepcopy(self.attr_embeddings.weight.cpu().detach().numpy()) if hasattr(self, 'attr_embeddings') else None
                        
                        if self.mapping_ins_emb is not None:
                            best_mapping = copy.deepcopy(self.mapping_ins_emb)
                        if not os.path.exists(self.args.save):
                            os.makedirs(self.args.save)
                        if graph_encoder:
                            torch.save(graph_encoder.state_dict(), self.args.save + "/best_model.ckpt")
                        logger.info(f"--- [Record] Nouveau record global Hits@1 ({current_score:.4f}) - Modèle sauvegardé ---")

                    if self.patience_counter >= self.args.patience:
                        logger.info(f"--- [STOP] Early stopping déclenché après {self.args.patience} baisses locales consécutives ---")
                        break
                else:
                    self.best_result = result
                    self.previous_score = result[0][0]
                    
        # Restaurer le meilleur état complet
        if self.args.early and graph_encoder and os.path.exists(self.args.save + "/best_model.ckpt"):
            logger.info("Chargement des meilleurs poids pour la sauvegarde finale...")
            graph_encoder.load_state_dict(torch.load(self.args.save + "/best_model.ckpt"))
        # Restaurer les embeddings du meilleur epoch
        if best_ins_emb_weights is not None:
            self.ins_embeddings.weight.data = torch.FloatTensor(best_ins_emb_weights).to(device)
        if best_rel_emb is not None:
            self.rel_embeddings.weight.data = torch.FloatTensor(best_rel_emb).to(device)
        if hasattr(self, 'attr_embeddings') and  best_attr_emb_weights is not None:
            self.attr_embeddings.weight.data = torch.FloatTensor(best_attr_emb_weights).to(device)

        # Recalcul des embeddings enrichis avec le meilleur modèle
        if graph_encoder:
            graph_encoder.eval()
            with torch.no_grad():
                edges = torch.LongTensor(d.ins_G_edges_idx).to(device)
                current_input_emb = self.get_input_embeddings()
                
                if graph_encoder.name == "rdgcn":
                    dual_edges_idx = torch.LongTensor(d.dual_edges_idx).to(device)
                    dual_edges_weight = torch.FloatTensor(d.dual_edges_weight).to(device)
                    r_ij_idx = torch.LongTensor(d.r_ij_idx).to(device)
                    enh_emb = graph_encoder.forward(
                        edges, current_input_emb,
                        r=None,
                        dual_edges=[dual_edges_idx, dual_edges_weight] ,
                        r_ij_idx=r_ij_idx)
                else:
                    enh_emb = graph_encoder(edges, current_input_emb)
                self.enh_ins_emb = enh_emb.cpu().detach().numpy()
        elif best_emb is not None:
            self.enh_ins_emb = best_emb

        # Sauvegarde des embeddings
        if self.save != "":
            if not os.path.exists(self.save):
                os.makedirs(self.save)
            time_str = (self.save_prefix + "_"+ time.strftime("%Y%m%d-%H%M", time.gmtime()))
            np.save(self.save + "/%s_enh_ins.npy" % time_str, self.enh_ins_emb)
            np.save(self.save + "/%s_ins.npy" % time_str, self.ins_embeddings.weight.cpu().detach().numpy())
            np.save(self.save + "/%s_rel.npy" % time_str, self.rel_embeddings.weight.cpu().detach().numpy())
            logger.info("Embeddings saved!")

        # Alignements Top-10 sur le test
        if self.save != "":
            logger.info("Sauvegarde des alignements finaux...")
            final_emb = self.enh_ins_emb
            test_left  = final_emb[d.ill_test_idx[:, 0]]
            test_right = final_emb[d.ill_test_idx[:, 1]]
            sim_mat    = sim(test_left, test_right, metric=self.args.test_dist, normalize=True)
            top_k_indices = np.argsort(sim_mat, axis=1)[:, -10:][:, ::-1]

            output_path = os.path.join(self.save, "final_alignments_top10.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("Entite_left\tEntite_right\tRang\tScore\tVrai\tVerification\n")
                for row_idx in range(len(d.ill_test_idx)):
                    id_l      = d.ill_test_idx[row_idx, 0]
                    name_l    = d.id2ins_dict[id_l]
                    true_id_r = d.ill_test_idx[row_idx, 1]
                    true_name_r = d.id2ins_dict[true_id_r]
                    for rank in range(10):
                        col_idx  = top_k_indices[row_idx, rank]
                        id_r     = d.ill_test_idx[col_idx, 1]
                        name_r   = d.id2ins_dict[id_r]
                        score    = sim_mat[row_idx, col_idx]
                        verif    = "vrai" if name_r == true_name_r else "faux"
                        f.write(f"{name_l}\t{name_r}\t{rank+1}\t{score:.4f}\t{true_name_r}\t{verif}\n")
            logger.info(f"Alignements Top-10 sauvegardés : {output_path}")

        # Config JSON
        if self.save != "":
            config_path = os.path.join(self.save, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(vars(self.args), f, indent=4)
            logger.info(f"Configuration sauvegardée : {config_path}")
        
        # Évaluation finale
        if len(d.ill_test_idx) > 0:
            logger.info("=" * 50)
            logger.info("🏆 ÉVALUATION FINALE OFFICIELLE SUR LE JEU DE TEST")
            logger.info("=" * 50)
            with torch.no_grad():
                res = self.evaluate(self.args.epoch, d.ill_test_idx, self.enh_ins_emb, True, self.mapping_ins_emb, self.initial_train_idx, self.initial_val_idx)
            logger.info("=" * 50)

            if self.save != "":
                metrics_path = os.path.join(self.save, "métriques.txt")
                with open(metrics_path, "w", encoding="utf-8") as f:
                    f.write("=== RÉSULTATS FINAUX (L2R) ===\n")
                    f.write(f"Hits@1 = précision = rappel = F1-score  : {res[0][0]:.4f}\n")
                    f.write(f"Hits@3  : {res[0][1]:.4f}\n")
                    f.write(f"Hits@10 : {res[0][3]:.4f}\n")
                    f.write(f"MR      : {res[1]:.2f}\n")
                    f.write(f"MRR     : {res[2]:.4f}\n")
                logger.info(f"Métriques sauvegardées : {metrics_path}")
    

    def train_1_epoch(self, it, opt, encoder, decoder, edges, triples, ills, ids, boot_triples, boot_pairs, ins_emb, rel_emb):
        if encoder:
            encoder.train()
        decoder.train()
        losses = []
        if "pos_"+decoder.print_name not in self.cached_sample or it % self.args.update == 0:
            if decoder.name in ["align", "mtranse_align", "n_r_align"]:
                if decoder.boot:
                    self.cached_sample["pos_"+decoder.print_name] = ills.tolist() + boot_pairs
                else:
                    self.cached_sample["pos_"+decoder.print_name] = ills.tolist()
                self.cached_sample["pos_"+decoder.print_name] = np.array(self.cached_sample["pos_"+decoder.print_name])
            else:
                if decoder.boot:
                    self.cached_sample["pos_"+decoder.print_name] = triples + boot_triples
                else:
                    self.cached_sample["pos_"+decoder.print_name] = triples
            np.random.shuffle(self.cached_sample["pos_"+decoder.print_name])
            
        train = self.cached_sample["pos_"+decoder.print_name]
        if self.args.train_batch_size == -1:
            train_batch_size = len(train)
        else:
            train_batch_size = self.args.train_batch_size
        for i in range(0, len(train), train_batch_size):
            pos_batch = train[i:i+train_batch_size]

            if (decoder.print_name+str(i) not in self.cached_sample or it % self.args.update == 0) and decoder.sampling_method:
                self.cached_sample[decoder.print_name+str(i)] = decoder.sampling_method(pos_batch, triples, ills, ids, decoder.k, params={
                    "emb": self.enh_ins_emb,
                    "metric": self.args.test_dist,
                })
            
            if decoder.sampling_method:
                neg_batch = self.cached_sample[decoder.print_name+str(i)]
    
            opt.zero_grad()
            if decoder.sampling_method:
                neg = torch.LongTensor(neg_batch).to(device)
                if neg.size(0) > len(pos_batch) * decoder.k:
                    pos = torch.LongTensor(pos_batch).repeat(decoder.k * 2, 1).to(device)
                elif hasattr(decoder.func, "loss") and decoder.name not in ["rotate", "hake", "conve", "mmea", "n_transe"]:
                    pos = torch.LongTensor(pos_batch).to(device)
                else:
                    pos = torch.LongTensor(pos_batch).repeat(decoder.k, 1).to(device)
            else:
                pos = torch.LongTensor(pos_batch).to(device)

            # On récupère ici les embeddings combinés (Structure + Attributs) 
            current_input_emb = self.get_input_embeddings()

            if encoder:
                use_edges = torch.LongTensor(edges).to(device)
                
                if encoder.name == "rdgcn":
                    dual_edges_idx = torch.LongTensor(d.dual_edges_idx).to(device)
                    dual_edges_weight = torch.FloatTensor(d.dual_edges_weight).to(device)
                    r_ij_idx = torch.LongTensor(d.r_ij_idx).to(device)
                    
                    # On passe les embeddings enrichis d'attributs à RDGCN
                    enh_emb = encoder.forward(use_edges, current_input_emb, r=rel_emb, dual_edges=[dual_edges_idx, dual_edges_weight], r_ij_idx=r_ij_idx)
                else:
                    enh_emb = encoder.forward(use_edges, current_input_emb, rel_emb[d.r_ij_idx] if encoder.name=="naea" else None)
            else:
                enh_emb = current_input_emb
            
            self.enh_ins_emb = enh_emb[0].cpu().detach().numpy() if encoder and encoder.name == "naea" else enh_emb.cpu().detach().numpy()
            
            if decoder.name == "n_r_align":
                rel_emb = current_input_emb

            if decoder.sampling_method:
                pos_score = decoder.forward(enh_emb, rel_emb, pos)
                neg_score = decoder.forward(enh_emb, rel_emb, neg)
                target = torch.ones(neg_score.size()).to(device)

                loss = decoder.loss(pos_score, neg_score, target) * decoder.alpha
            else:
                loss = decoder.forward(enh_emb, rel_emb, pos) * decoder.alpha
            
            loss.backward()
            
            opt.step()
            losses.append(loss.item())
        
        return np.mean(losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/DBP15K/zh_en", required=False, help="input dataset file directory, ('data/DBP15K/zh_en', 'data/DWY100K/dbp_wd')")
    parser.add_argument("--rate", type=float, default=0.3, help="training set rate")
    parser.add_argument("--val", type=float, default=0.0, help="valid set rate")
    parser.add_argument("--save", default="", help="the output dictionary of the model and embedding")
    parser.add_argument("--pre", default="", help="pre-train embedding dir")
    parser.add_argument("--cuda", action="store_true", default=True, help="whether to use cuda or not")
    parser.add_argument("--log", type=str, default="tensorboard_log", nargs="?", help="where to save the log")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    parser.add_argument("--epoch", type=int, default=1000, help="number of epochs to train")
    parser.add_argument("--check", type=int, default=5, help="check point")
    parser.add_argument("--update", type=int, default=5, help="number of epoch for updating negtive samples")
    parser.add_argument("--train_batch_size", type=int, default=-1, help="train batch_size (-1 means all in)")
    parser.add_argument("--early", action="store_true", default=False, help="whether to use early stop") 
    parser.add_argument("--share", action="store_true", default=False, help="whether to share ill emb")
    parser.add_argument("--swap", action="store_true", default=False, help="whether to swap ill in triple")
    parser.add_argument('--patience', type=int, default=3, help='Nombre de checks avant early stop')

    parser.add_argument('--use_fasttext', action="store_true", default=False, help='Utiliser Fasttext pour l\'initialisation')
    parser.add_argument('--use_glove', action="store_true", default=False, help='Utiliser GloVe pour l\'initialisation')
    parser.add_argument('--glove_path', type=str, default='data/glove.6B/glove.6B.100d.txt', help='Chemin du fichier GloVe')
    parser.add_argument('--use_attr', action="store_true", default=False, help='Utiliser les attributs pour enrichir l\'initialisation')
    parser.add_argument('--attr_alpha', type=float, default=0.1, help='Poids accordé aux attributs lors de la fusion dynamique (ex: 0.1)')

    parser.add_argument("--bootstrap", action="store_true", default=False, help="whether to use bootstrap")
    parser.add_argument("--start_bp", type=int, default=9, help="epoch of starting bootstrapping")
    parser.add_argument("--threshold", type=float, default=0.75, help="threshold of bootstrap alignment")

    parser.add_argument("--encoder", type=str, default="GCN-Align", nargs="?", help="which encoder to use: . max = 1")
    parser.add_argument("--hiddens", type=str, default="100,100,100", help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma")
    parser.add_argument("--heads", type=str, default="1,1", help="heads in each gat layer, splitted with comma")
    parser.add_argument("--attn_drop", type=float, default=0, help="dropout rate for gat layers")

    parser.add_argument("--decoder", type=str, default="Align", nargs="?", help="which decoder to use: . min = 1")
    parser.add_argument("--sampling", type=str, default="N", help="negtive sampling method for each decoder")
    parser.add_argument("--k", type=str, default="25", help="negtive sampling number for each decoder")
    parser.add_argument("--margin", type=str, default="1", help="margin for each margin based ranking loss (or params for other loss function)")
    parser.add_argument("--alpha", type=str, default="1", help="weight for each margin based ranking loss")
    parser.add_argument("--feat_drop", type=float, default=0, help="dropout rate for layers")

    parser.add_argument("--lr", type=float, default=0.005, help="initial learning rate")
    parser.add_argument("--wd", type=float, default=0, help="weight decay (L2 loss on parameters)")
    parser.add_argument("--dr", type=float, default=0, help="decay rate of lr")    

    parser.add_argument("--train_dist", type=str, default="euclidean", help="distance function used in train (inner, cosine, euclidean, manhattan)")
    parser.add_argument("--test_dist", type=str, default="euclidean", help="distance function used in test (inner, cosine, euclidean, manhattan)")

    parser.add_argument("--csls", type=int, default=0, help="whether to use csls in test (0 means not using)")
    parser.add_argument("--rerank", action="store_true", default=False, help="whether to use rerank in test")

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    writer = SummaryWriter("_runs/%s_%s" % (args.data_dir.split("/")[-1], args.log))
    logger.info(args)

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Load Data
    d = AlignmentData(data_dir=args.data_dir, rate=args.rate, share=args.share, swap=args.swap, val=args.val, with_r=args.encoder.lower()in ["naea", "rdgcn"])
    logger.info(d)

    experiment = Experiment(args=args)

    t_total = time.time()
    experiment.train_and_eval()
    logger.info("optimization finished!")
    logger.info("total time elapsed: {:.4f} s".format(time.time() - t_total))