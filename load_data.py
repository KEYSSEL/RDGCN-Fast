#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import torch
import re
from collections import defaultdict


class AlignmentData:

    def __init__(self, data_dir="data/DBP15K/ja_en", rate=0.3, share=False, swap=False, val=0.0, with_r=False):
        t_ = time.time()

        self.rate = rate
        self.val = val
        self.ins2id_dict, self.id2ins_dict, [self.kg1_ins_ids, self.kg2_ins_ids] = self.load_dict(data_dir + "/ent_ids_", file_num=2)
        self.rel2id_dict, self.id2rel_dict, [self.kg1_rel_ids, self.kg2_rel_ids] = self.load_dict(data_dir + "/rel_ids_", file_num=2)
        self.ins_num = max(self.id2ins_dict.keys()) + 1 if self.id2ins_dict else 0
        self.rel_num = max(self.id2rel_dict.keys()) + 1 if self.id2rel_dict else 0
        self.triple_idx = self.load_triples(data_dir + "/triples_", file_num=2)
        self.ent2attrs, self.attr_num, self.id2attr_dict = self.load_attributes_from_triples(data_dir + "/attr_triples_", file_num=2)
        #self.ent2attrs, self.attr_num, self.id2attr_dict = self.load_attributes( data_dir=data_dir + "/ent_attrs_",  dict_dir=data_dir + "/attr_ids_",  file_num=2)

        
        self.ill_idx = self.load_triples(data_dir + "/all_pairs.txt", file_num=1)
        np.random.shuffle(self.ill_idx)

        # --- DÉCOUPAGE 100% DÉTERMINISTE ---
        self.ill_idx = sorted(self.ill_idx)
        
        rng = np.random.default_rng(seed=23)
        rng.shuffle(self.ill_idx)
        # -----------------------------------


        self.ill_train_idx, self.ill_val_idx, self.ill_test_idx = np.array(self.ill_idx[:int(len(self.ill_idx) // 1 * rate)], dtype=np.int32), np.array(self.ill_idx[int(len(self.ill_idx) // 1 * rate) : int(len(self.ill_idx) // 1 * (rate+val))], dtype=np.int32), np.array(self.ill_idx[int(len(self.ill_idx) // 1 * (rate+val)):], dtype=np.int32)

        self.ins_G_edges_idx, self.ins_G_values_idx, self.r_ij_idx = self.gen_sparse_graph_from_triples(self.triple_idx, self.ins_num, with_r)

        # --- AJOUT RDGCN : Création de la variable pour le graphe dual ---
        if with_r:
            self.dual_edges_idx, self.dual_edges_weight = self.gen_dual_graph(self.triple_idx, self.rel_num)
        else:
            self.dual_edges_idx, self.dual_edges_weight = None, None
        # -----------------------------------------------------------------
        
        assert (share != swap or (share == False and swap == False))
        if share:
            self.triple_idx = self.share(self.triple_idx, self.ill_train_idx)   # 1 -> 2:base
            self.kg1_ins_ids = (self.kg1_ins_ids - set(self.ill_train_idx[:, 0])) | set(self.ill_train_idx[:, 1])
            self.ill_train_idx = []
        if swap:
            self.triple_idx = self.swap(self.triple_idx, self.ill_train_idx)
        self.labeled_alignment = set()
        self.boot_triple_idx = []
        self.boot_pair_dix = []

        self.init_time = time.time() - t_

    def load_triples(self, data_dir, file_num=2):
        if file_num == 2:
            file_names = [data_dir + str(i) for i in range(1, 3)]
        else:
            file_names = [data_dir]
        triple = []
        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as f:
                data = f.read().strip().split("\n")
                data = [tuple(map(int, i.split("\t"))) for i in data]
                triple += data
        np.random.shuffle(triple)
        return triple

    def load_dict(self, data_dir, file_num=2):
        if file_num == 2:
            file_names = [data_dir + str(i) for i in range(1, 3)]
        else:
            file_names = [data_dir]
        what2id, id2what, ids = {}, {}, []
        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as f:
                data = f.read().strip().split("\n")
                data = [i.split("\t") for i in data]
                what2id = {**what2id, **dict([[i[1], int(i[0])] for i in data])}
                id2what = {**id2what, **dict([[int(i[0]), i[1]] for i in data])}
                ids.append(set([int(i[0]) for i in data]))
        return what2id, id2what, ids

    
    def load_attributes(self, data_dir, dict_dir, file_num=2):
        
        if file_num == 2:
            file_names = [data_dir + str(i) for i in range(1, 3)]
            dict_names = [dict_dir + str(i) for i in range(1, 3)]
        else:
            file_names = [data_dir]
            dict_names = [dict_dir]
        
        for file_name in file_names:
            print(f"[load_attributes] Attempting to load attributes from file: {file_name}")


        # 1. Chargement du dictionnaire des attributs (id2attr_dict)
        id2attr_dict = {}
        for dict_name in dict_names:
            try:
                with open(dict_name, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            attr_id = int(parts[0])
                            attr_uri = parts[1]
                            id2attr_dict[attr_id] = attr_uri
            except FileNotFoundError:
                pass 

        attr_num = len(id2attr_dict)
        if attr_num > 0:
            print(f"[load_attributes] {attr_num} unique attributes loaded from dictionaries.")

        # 2. Chargement des associations entités -> attributs (ent2attrs)
        ent2attrs = {}
        for file_name in file_names:
            try:
                with open(file_name, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) > 1:
                            ent_id = int(parts[0])
                            attrs = [int(x) for x in parts[1:]]
                            
                            if ent_id not in ent2attrs:
                                ent2attrs[ent_id] = []
                            ent2attrs[ent_id].extend(attrs)
            except FileNotFoundError:
                pass 
                
        return ent2attrs, attr_num, id2attr_dict

    def load_attributes_from_triples(self, data_dir, file_num=2):
        
        if file_num == 2:
            file_names = [data_dir + str(i) for i in range(1, 3)]
        else:
            file_names = [data_dir]
 
        attr2id   = {}   # attr_uri  -> int id
        ent2attrs = {}   # ent_id    -> [attr_id, ...]
 
        for file_name in file_names:
            try:
                with open(file_name, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip("\r\n")
                        if not line:
                            continue
                        
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            ent_uri  = parts[0].strip().strip('<>')
                            attr_raw_val = parts[2].strip()
                            is_textual = False
                            
                            if "@" in attr_raw_val:
                                is_textual = True
                            elif "^^" in attr_raw_val:
                                datatype = attr_raw_val.split("^^")[1].lower()
                                if "string" in datatype or "langstring" in datatype:
                                    is_textual = True
                                else:
                                    # Types comme integer, date, float, etc. -> REJETÉS
                                    is_textual = False
                            else:
                                # On vérifie si c'est un nombre caché.
                                clean_test_val = attr_raw_val.strip('"\'')
                                try:
                                    float(clean_test_val)
                                    is_textual = False
                                except ValueError:
                                    is_textual = True 
                            
                            # Si ce n'est pas textuel, on ignore complètement cet attribut
                            if not is_textual:
                                continue

                            if "^^" in attr_raw_val:
                                attr_val = attr_raw_val.split("^^")[0]
                            elif "@" in attr_raw_val:
                                attr_val = attr_raw_val.split("@")[0]
                            else:
                                attr_val = attr_raw_val
                                
                            attr_val = attr_val.strip('"\'')
                            
                            if not attr_val:
                                continue

                            if ent_uri not in self.ins2id_dict:
                                continue
     
                            ent_id = self.ins2id_dict[ent_uri]
     
                            if attr_val not in attr2id:
                                attr2id[attr_val] = len(attr2id)
                            attr_id = attr2id[attr_val]
     
                            if ent_id not in ent2attrs:
                                ent2attrs[ent_id] = []
                            if attr_id not in ent2attrs[ent_id]:
                                ent2attrs[ent_id].append(attr_id)
 
            except FileNotFoundError:
                print(f"⚠️ [load_attributes_from_triples] Fichier introuvable : {file_name}")
 
        attr_num = len(attr2id)
        if attr_num > 0:
            print(f"✅ [load_attributes_from_triples] {attr_num} attributs textuelles uniques trouvés, "
                  f"{len(ent2attrs)} entités enrichies.")
        else:
            print("⚠️ [load_attributes_from_triples] Aucun attribut trouvé. Vérifie le format de tes fichiers !")
            
        id2attr_dict = {v: k for k, v in attr2id.items()}
        return ent2attrs, attr_num, id2attr_dict

    def gen_sparse_graph_from_triples(self, triples, ins_num, with_r=False):
            edge_dict = {}
            for (h, r, t) in triples:
                if h != t:
                    if (h, t) not in edge_dict:
                        edge_dict[(h, t)] = []
                        edge_dict[(t, h)] = []
                    edge_dict[(h, t)].append(r)
                    edge_dict[(t, h)].append(-r)
                    
            if with_r:
                edges = [[h, t] for (h, t) in edge_dict for r in edge_dict[(h, t)]]
                values = [1 for (h, t) in edge_dict for r in edge_dict[(h, t)]]
                r_ij = [abs(r) for (h, t) in edge_dict for r in edge_dict[(h, t)]]
                
                edges = np.array(edges, dtype=np.int32)
                values = np.array(values, dtype=np.float32)
                r_ij = np.array(r_ij, dtype=np.int32)
                
                # On utilise 'self.rel_num' comme ID de relation spécial
                self_loop_r = np.full(ins_num, self.rel_num, dtype=np.int32) 
                self_edges = np.stack([np.arange(ins_num), np.arange(ins_num)], axis=1)
                
                edges = np.concatenate([edges, self_edges], axis=0)
                values = np.concatenate([values, np.ones(ins_num, dtype=np.float32)], axis=0)
                r_ij = np.concatenate([r_ij, self_loop_r], axis=0)
                # ---------------------------------------------------------------
                
                return edges, values, r_ij
                
            else:
                edges = [[h, t] for (h, t) in edge_dict]
                values = [1 for (h, t) in edge_dict]
                
            # add self-loop pour le cas sans relations
            edges += [[e, e] for e in range(ins_num)]
            values += [1 for e in range(ins_num)]
            edges = np.array(edges, dtype=np.int32)
            values = np.array(values, dtype=np.float32)
            return edges, values, None

    def gen_dual_graph(self, triples, rel_num):
        
        # 1. On mappe chaque relation à l'ensemble de ses entités (Head et Tail séparés ou joints)
        rel_to_ents = defaultdict(set)
        for h, r, t in triples:
            rel_to_ents[r].add(h)
            rel_to_ents[r].add(t)
        
        # 2. On identifie quelles relations partagent quelles entités (pour accélérer le calcul)
        ent_to_rels = defaultdict(set)
        for r, ents in rel_to_ents.items():
            for e in ents:
                ent_to_rels[e].add(r)
                
        dual_edges_weighted = {} # (r1, r2) -> weight

        # 3. Calcul de Jaccard uniquement pour les relations qui partagent au moins une entité
        for ent, rels in ent_to_rels.items():
            rel_list = list(rels)
            for i in range(len(rel_list)):
                for j in range(i, len(rel_list)):
                    r1, r2 = rel_list[i], rel_list[j]
                    if (r1, r2) not in dual_edges_weighted:
                        # Formule de Jaccard : intersection / union
                        inter = len(rel_to_ents[r1] & rel_to_ents[r2])
                        union = len(rel_to_ents[r1] | rel_to_ents[r2])
                        weight = inter / union if union > 0 else 0
                        
                        dual_edges_weighted[(r1, r2)] = weight
                        dual_edges_weighted[(r2, r1)] = weight

        # 4. On s'assure que les self-loops existent avec un poids de 1.0
        for r in range(rel_num):
            dual_edges_weighted[(r, r)] = 1.0
            
        edges = []
        weights = []
        for (u, v), w in dual_edges_weighted.items():
            edges.append([u, v])
            weights.append(w)
            
        np_edges = np.array(edges, dtype=np.int32)
        np_weights = np.array(weights, dtype=np.float32)
        
        return np_edges, np_weights
    
    def share(self, triples, ill):
        from_1_to_2_dict = dict(ill)
        new_triples = []
        for (h, r, t) in triples:
            if h in from_1_to_2_dict:
                h = from_1_to_2_dict[h]
            if t in from_1_to_2_dict:
                t = from_1_to_2_dict[t]
            new_triples.append((h, r, t))
        new_triples = list(set(new_triples))
        return new_triples
    
    def swap(self, triples, ill):
        from_1_to_2_dict = dict(ill)
        from_2_to_1_dict = dict(ill[:, ::-1])
        new_triples = []
        for (h, r, t) in triples:
            new_triples.append((h, r, t))
            if h in from_1_to_2_dict:
                new_triples.append((from_1_to_2_dict[h], r, t))
            if t in from_1_to_2_dict:
                new_triples.append((h, r, from_1_to_2_dict[t]))
            if h in from_2_to_1_dict:
                new_triples.append((from_2_to_1_dict[h], r, t))
            if t in from_2_to_1_dict:
                new_triples.append((h, r, from_2_to_1_dict[t]))
        new_triples = list(set(new_triples))
        return new_triples

    def __repr__(self):
        return self.__class__.__name__ + " dataset summary:" + \
            "\n\tins_num: " + str(self.ins_num) + \
            "\n\trel_num: " + str(self.rel_num) + \
            "\n\ttriple_idx: " + str(len(self.triple_idx)) + \
            "\n\trate: " + str(self.rate) + "\tval: " + str(self.val) + \
            "\n\till_idx(train/test/val): " + str(len(self.ill_idx)) + " = " + str(len(self.ill_train_idx)) + " + " + str(len(self.ill_test_idx)) + " + " + str(len(self.ill_val_idx)) + \
            "\n\tins_G_edges_idx: " + str(len(self.ins_G_edges_idx)) + \
            "\n\t----------------------------- init_time: " + str(round(self.init_time, 3)) + "s"


if __name__ == '__main__':
    
    # TEST

    d = AlignmentData(share=False, swap=False)
    print(d)
    d = AlignmentData(share=True, swap=False)
    print(d)
    d = AlignmentData(share=False, swap=True)
    print(d)
    