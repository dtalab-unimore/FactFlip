import torch
import time

import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm

from utils import output_metrics
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

import torch.nn.functional as F

def max_pos(l):
    pos = 0
    max_value = -float('inf')
    for i, el in enumerate(l):
        if el > max_value:
            pos = i
            max_value = el

    return pos

class Trainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def train(self, epoch, model, loss_fn, optimizer, train_loader):
        epoch_start_time = time.time()
        model.train()
        tr_loss = 0

        for batch in tqdm(train_loader, desc='Iteration'):
            batch = tuple(t.to(self.device) if not isinstance(t, list) and not isinstance(t, str) else t for t in batch)
            #batch = tuple(t if not isinstance(t, list) and not isinstance(t, str) else t for t in batch)
            claim, evidence, ids_sent1, segs_sent1, att_mask_sent1, labels = batch

            out = model(ids_sent1, segs_sent1, att_mask_sent1)
            if isinstance(labels, list):
                labels = torch.tensor(np.array(labels)) #.to(self.device)
            loss = loss_fn(out, labels.float())

            tr_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        timing = time.time() - epoch_start_time
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Timing: {timing}, Epoch: {epoch + 1}, training loss: {tr_loss}, current learning rate {cur_lr}")

    def val(self, model, val_loader, return_preds=False):
        model.eval()

        loss_fn = nn.CrossEntropyLoss()

        val_loss = 0
        val_preds = []
        val_labels = []
        outs = []
        for batch in tqdm(val_loader):
            batch = tuple(t.to(self.device) if not isinstance(t, list) and not isinstance(t, str) else t for t in batch)
            claim, evidence, ids_sent1, segs_sent1, att_mask_sent1, labels = batch

            with torch.no_grad():
                out = model(ids_sent1, segs_sent1, att_mask_sent1)
                if out.shape[-1] > 3:
                    # hardcoded initial ids for "support", "refute", "not enough information" for Qwen tokenizer
                    if labels.shape[-1] != 2:
                        out.data = out.data[:, torch.tensor([1824, 83177, 537])] #refute: 83177, contrast: 12872, negate: 71219
                    else:
                        out.data = out.data[:, torch.tensor([1824, 83177])]

                preds = torch.max(out.data, 1)[1].cpu().numpy().tolist()
                print(preds, end=" ")

                loss = loss_fn(out, labels.float())
                val_loss += loss.item()

                outs.extend([out.data.cpu().numpy().tolist()[i] for i in range(len(out)) if torch.max(out.data, 1)[1].cpu().numpy().tolist()[i] == torch.max(labels,1)[1].cpu().numpy().tolist()[i]])

                labels = labels.cpu().numpy().tolist()

                val_labels.extend(labels)
                if len(labels[0]) != 2:
                    for pred in preds:
                        if pred == 0:
                            val_preds.append([1,0,0])
                        elif pred == 1:
                            val_preds.append([0,1,0])
                        else:
                            val_preds.append([0,0,1])
                else:
                    val_preds.extend([[1,0] if pred == 0 else [0,1] for pred in preds])

        print(f"val loss: {val_loss}")

        val_acc, val_prec, val_recall, val_f1 = output_metrics(val_labels, val_preds)

        if return_preds:
            return val_acc, val_prec, val_recall, val_f1, val_preds, val_labels
        return val_acc, val_prec, val_recall, val_f1

class AntonymTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def val(self, model, train_loader, **kwargs):
        concept_vectors = {}

        model.eval() # we put it to eval mode because we do not update the gradient
        with torch.no_grad():
            w = model.linear_layer.weight

            for batch in tqdm(train_loader, desc='Training...'):
                batch = tuple(t.to(self.device) if not isinstance(t, list) else t for t in batch)
                sent1, sent2, ids_sent1, segs_sent1, att_mask_sent1, ids_sent2, segs_sent2, att_mask_sent2 = batch
                out, out_sent1, out_sent2 = model.compute_concept_vector(ids_sent1, segs_sent1, att_mask_sent1, ids_sent2, segs_sent2, att_mask_sent2)

                out_normalized = out

                for i in range(len(w)):
                    w_normalized = w[i]
                    cos_sim = torch.matmul(out_normalized, w_normalized.unsqueeze(-1)).squeeze(-1)
                    for j in range(len(out)):
                        if (sent1[j], sent2[j]) not in concept_vectors.keys():
                            concept_vectors[(sent1[j], sent2[j])] = {}
                        if i == 0:
                            concept_vectors[(sent1[j], sent2[j])]["support"] = cos_sim[j].item()
                        elif i == 1:
                            concept_vectors[(sent1[j], sent2[j])]["refute"] = cos_sim[j].item()
                        else:
                            concept_vectors[(sent1[j], sent2[j])]["nei"] = cos_sim[j].item()

        return concept_vectors

    def val_potency(self, model, train_loader, num_labels=2, **kwargs):
        concept_vectors = {}

        model.eval()  # we put it to eval mode because we do not update the gradient
        with torch.no_grad():
            try:
                # roberta
                w = model.linear_layer.weight
            except:
                # qwen
                w = model.plm.lm_head.weight
                if num_labels != 2:
                    w = w[torch.tensor([1824, 83177, 537])] # for qwen2.5
                else:
                    w = w[torch.tensor([1824, 83177])] # for qwen2.5

            for batch in tqdm(train_loader, desc='Training...'):
                batch = tuple(t.to(self.device) if not isinstance(t, list) else t for t in batch)
                #batch = tuple(t if not isinstance(t, list) else t for t in batch)
                # words are provided in pairs following the antonym setup
                # in this case, words are not evaluated wrt each other, so we get the out_sent1 and out_sent2 to evaluate them separately
                # roberta
                sent1, sent2, ids_sent1, segs_sent1, att_mask_sent1, ids_sent2, segs_sent2, att_mask_sent2 = batch
                out, out_sent1, out_sent2 = model.compute_concept_vector(ids_sent1, segs_sent1, att_mask_sent1,
                                                                         ids_sent2, segs_sent2, att_mask_sent2)
                out_sent1 = out_sent1.to(w.device)
                out_sent2 = out_sent2.to(w.device)
                for i in range(len(w)):
                    cos_sim1 = torch.matmul(out_sent1, w[i].unsqueeze(-1)).squeeze(-1)
                    cos_sim2 = torch.matmul(out_sent2, w[i].unsqueeze(-1)).squeeze(-1)
                    for j in range(len(out_sent1)):
                        if sent1[j] not in concept_vectors.keys():
                            concept_vectors[sent1[j]] = {}
                        if sent2[j] not in concept_vectors.keys():
                            concept_vectors[sent2[j]] = {}

                        if i == 0:
                            concept_vectors[sent1[j]]["support"] = cos_sim1[j].item()
                            concept_vectors[sent2[j]]["support"] = cos_sim2[j].item()
                        elif i == 1:
                            concept_vectors[sent1[j]]["refute"] = cos_sim1[j].item()
                            concept_vectors[sent2[j]]["refute"] = cos_sim2[j].item()
                        else:
                            concept_vectors[sent1[j]]["nei"] = cos_sim1[j].item()
                            concept_vectors[sent2[j]]["nei"] = cos_sim2[j].item()

        return concept_vectors
