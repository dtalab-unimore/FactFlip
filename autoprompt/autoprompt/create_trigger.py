import time
import argparse
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import transformers
from transformers import AutoConfig, AutoModel, AutoModelWithLMHead, AutoTokenizer
from tqdm import tqdm
import pandas as pd
from copy import deepcopy
import os

import autoprompt.utils as utils
from autoprompt.model import RobertaModel, LlamaModel
from autoprompt.utils import get_device

from autoprompt.data_processor import AVTCProcessor, FeverProcessor, FeverSymmetricProcessor, SciFactProcessor, VitamincProcessor, \
  dataset, collate_fn, collate_fn_antonym, collate_fn_trigger, FM2Processor, PolitiHopProcessor, HoverProcessor, FactEvalProcessor, AdvLLMProcessor, AntonymsProcessor

from functools import lru_cache
import pickle

logger = logging.getLogger(__name__)



class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient


class PredictWrapper:
    """
    PyTorch transformers model wrapper. Handles necc. preprocessing of inputs for triggers
    experiments.
    """
    def __init__(self, model):
        self.fc_model = model

    def run(self, ids_sent, segs_sent, attn_mask):
        return self.fc_model(ids_sent, segs_sent, attn_mask)

    def __call__(self, ids_sent, segs_sent, attn_mask, trigger_mask=None, trigger_ids=None, labels=None, verbose=False): #model_inputs, trigger_ids):
        # Copy dict so pop operations don't have unwanted side-effects
        """model_inputs = model_inputs.copy()
        trigger_mask = model_inputs.pop('trigger_mask')
        predict_mask = model_inputs.pop('predict_mask')"""
        if trigger_mask is not None:
            ids_sent = replace_trigger_tokens(ids_sent, trigger_ids, trigger_mask)

        predict_logits = self.fc_model(ids_sent, segs_sent, attn_mask)
        if predict_logits.shape[-1] > 3:  # llama
            if labels is not None and labels.shape[-1] != 2:
                predict_logits.data = predict_logits.data[:, torch.tensor([1824, 83177, 537])]
            else:
                predict_logits.data = predict_logits.data[:, torch.tensor([1824, 83177])]
        return predict_logits

class TopKList:
    def __init__(self, k, sort_by_similarity=False, opposite=False):
        self.data = []
        self.data_candidates = []
        self.k = k
        self.sort_by_similarity = sort_by_similarity
        self.opposite = opposite

    def add(self, value, j, word, similarity, num_tokens):
        if isinstance(value[j], torch.Tensor) and value[j].ndim == 0:
            scalar = value[j].item()
        else:
            scalar = value[j]

        if isinstance(similarity, torch.Tensor) and similarity.ndim == 0:
            similarity = similarity.item()

        self.data_candidates.append((scalar, j, word, num_tokens, similarity))

        if not self.sort_by_similarity:
            self.data_candidates = sorted(self.data_candidates, key=lambda x: x[0], reverse=True)
        else:
            self.data_candidates = sorted(self.data_candidates, key=lambda x: x[-1], reverse=True)

        self.data_candidates = self.get_unique_words(self.data_candidates)

        if not self.opposite:
            self.data_candidates = self.data_candidates[:self.k]
        else:
            self.data_candidates = self.data_candidates[-self.k:]

    def get_unique_words(self, lst):
        seen = set()
        unique = []
        for cand in lst:
            key = cand[2]
            if key not in seen:
                seen.add(key)
                unique.append(cand)

        return unique

    def merge(self):
        if len(self.data) == 0:
            self.data = self.data_candidates
        else:
            self.data.extend(self.data_candidates)
            if not self.sort_by_similarity:
                self.data = sorted(self.data, key=lambda x: x[0], reverse=True)
            else:
                self.data = sorted(self.data, key=lambda x: x[-1], reverse=True)

            self.data = self.get_unique_words(self.data)

            if not self.opposite:
                self.data = self.data[:self.k]
            else:
                self.data = self.data[-self.k:]

        self.data_candidates = []

    def get(self):
        return self.data

    def save(self, dir_path):
        with open(os.path.join(dir_path, 'data.pkl'), 'wb') as f:
            pickle.dump(self.data, f)

    def __str__(self):
        return '\n'.join(f"{i+1}: Score={score:.4f}, Index={j}, Word='{word}'"
                         for i, (score, j, word) in enumerate(self.data))

    def __repr__(self):
        return f"TopKList(k={self.k}, data={self.data})"


class AccuracyFn:
    """
    Computing the accuracy when a label is mapped to multiple tokens is difficult in the current
    framework, since the data generator only gives us the token ids. To get around this we
    compare the target logp to the logp of all labels. If target logp is greater than all (but)
    one of the label logps we know we are accurate.
    """
    def __init__(self, tokenizer, label_map, device, tokenize_labels=False):
        self._all_label_ids = []
        self._pred_to_label = []
        for label, label_tokens in label_map.items():
            self._all_label_ids.append(utils.encode_label(tokenizer, label_tokens, tokenize_labels).to(device))
            self._pred_to_label.append(label)

    def __call__(self, predict_logits, gold_label_ids):
        # Get total log-probability for the true label
        gold_logp = get_loss(predict_logits, gold_label_ids)

        # Get total log-probability for all labels
        bsz = predict_logits.size(0)
        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        _, predictions = all_label_logp.max(dim=-1)
        predictions = [self._pred_to_label[x] for x in predictions.tolist()]

        # Add up the number of entries where loss is greater than or equal to gold_logp.
        ge_count = all_label_logp.le(gold_logp.unsqueeze(-1)).sum(-1)
        correct = ge_count.le(1)  # less than in case of num. prec. issues

        return correct.float()

    # TODO: @rloganiv - This is hacky. Replace with something sensible.
    def predict(self, predict_logits):
        bsz = predict_logits.size(0)
        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        _, predictions = all_label_logp.max(dim=-1)
        predictions = [self._pred_to_label[x] for x in predictions.tolist()]
        return predictions

def load_model(model_path, config):
    config = vars(config)
    if "qwen" in config["model_name"].lower():
        fc_model = LlamaModel(config)
    else:
        fc_model = RobertaModel(config)

    @lru_cache()
    def load_from_state_dict(fc_model, model_path):
        device = utils.get_device()
        state_dict = torch.load(model_path, map_location=device)
        fc_model.load_state_dict(state_dict)
        fc_model.to(device)
        fc_model.eval()
        return fc_model

    return load_from_state_dict(fc_model, model_path)

def load_pretrained(model_path, backbone, config):
    """
    Loads pretrained HuggingFace config/model/tokenizer, as well as performs required
    initialization steps to facilitate working with triggers.
    """
    model = load_model(model_path, config)
    tokenizer = AutoTokenizer.from_pretrained(backbone, add_prefix_space=False)
    utils.add_task_specific_tokens(tokenizer)
    return model, tokenizer

def get_samples_by_position(data, positions):
    new_data = [data[position] for position in positions]
    return new_data

def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_embeddings(model):
    """Returns the wordpiece embedding module."""
    #base_model = getattr(model, config.model_type)
    embeddings = model.plm.embeddings.word_embeddings
    return embeddings


def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   vocabulary, # mapping from word to tokens
                   opposite=False,
                   num_candidates=1,
                   memory=None):
    """Returns the top candidate replacements."""
    word_scores = None
    with torch.no_grad():
        while word_scores is None:
            for i, (word, tokens) in enumerate(vocabulary[len(averaged_grad)].items()):
                #if memory is not None and i in memory:
                #    continue
                emb_tokens = embedding_matrix[tokens].squeeze(0)
                result = torch.sum(emb_tokens * -averaged_grad, dim=1, keepdim=False)
                result = torch.mean(result.unsqueeze(-1), dim=0)

                if word_scores is None:
                    word_scores = result
                else:
                    word_scores = torch.cat((word_scores, result))

            if word_scores is None:
                memory = []

        """gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )"""

        """if not increase_loss:
            gradient_dot_embedding_matrix *= -1"""
        len_memory = 0 if memory is None else len(memory)
        #assert len(word_scores) == (len(vocabulary[len(averaged_grad)]) - len_memory), f"{len(word_scores)} is not equal to {len(vocabulary[len(averaged_grad)])-len_memory}"

        num_candidates = min(len(word_scores), num_candidates)
        similarity, top_k_ids = word_scores.topk(num_candidates, largest = not opposite)

    return top_k_ids, similarity


def replace_trigger_tokens(ids_sent, trigger_ids, trigger_mask):
    """Replaces the trigger tokens in input_ids."""
    """out = model_inputs.copy()
    input_ids = model_inputs['input_ids']"""
    trigger_ids = trigger_ids.repeat(trigger_mask.size(0), 1)
    trigger_mask = trigger_mask.to(torch.bool)
    ids_sent = ids_sent.masked_scatter(trigger_mask, trigger_ids)
    """tok = AutoTokenizer.from_pretrained("roberta-base")
    print(tok.decode(ids_sent[0]))
    print(a)"""
    return ids_sent


def get_loss(predict_logits, label_ids):
    predict_logp = F.log_softmax(predict_logits, dim=-1)
    target_logp = predict_logp.gather(-1, label_ids)
    target_logp = target_logp - 1e32 * label_ids.eq(0)  # Apply mask
    target_logp = torch.logsumexp(target_logp, dim=-1)
    return -target_logp


def isupper(idx, tokenizer):
    """
    Determines whether a token (e.g., word piece) begins with a capital letter.
    """
    _isupper = False
    # We only want to check tokens that begin words. Since byte-pair encoding
    # captures a prefix space, we need to check that the decoded token begins
    # with a space, and has a capitalized second character.
    if isinstance(tokenizer, transformers.GPT2Tokenizer):
        decoded = tokenizer.decode([idx])
        if decoded[0] == ' ' and decoded[1].isupper():
            _isupper = True
    # For all other tokenization schemes, we can just check the first character
    # is capitalized.
    elif tokenizer.decode([idx])[0].isupper():
            _isupper = True
    return _isupper

def get_data(config, positions=None, return_data=False, add_trigger_tokens=False):
    if config["dataset"] == "avtc":
        processor = AVTCProcessor(config)
        num_classes = 3

        path_train = "./data/avtc/train.json"
        path_dev = "./data/avtc/dev.json"
        path_test = "./data/avtc/test.json"

    elif config["dataset"] == "fever":
        processor = FeverProcessor(config)
        num_classes = 3

        path_train = "./data/fever/train.jsonl"
        path_dev = "./data/fever/paper_dev.jsonl"
        path_test = "./data/fever/paper_test.jsonl"

    elif config["dataset"] == "feversymmetric":
        processor = FeverSymmetricProcessor(config)
        num_classes = 3

        path_train = "./data/fever/train.jsonl"
        path_dev = "./data/fever/paper_dev.jsonl"
        path_test = "./data/feversymmetric/test.jsonl"

    elif config["dataset"] == "scifact":
        processor = SciFactProcessor(config)
        num_classes = 3

        path_train = "./data/scifact/claims_train.jsonl"
        path_dev = "./data/scifact/claims_dev.jsonl"
        path_test = "./data/scifact/claims_test.jsonl"

    elif config["dataset"] == "vitaminc":
        processor = VitamincProcessor(config)
        num_classes = 3

        path_train = "./data/vitaminc/train.jsonl"
        path_dev = "./data/vitaminc/dev.jsonl"
        path_test = "./data/vitaminc/test.jsonl"

    elif config["dataset"] == "fm2":
        processor = FM2Processor(config)
        num_classes = 2

        path_train = "./data/fm2/train.jsonl"
        path_dev = "./data/fm2/dev.jsonl"
        path_test = "./data/fm2/test.jsonl"

    elif config["dataset"] == "politihop":
        processor = PolitiHopProcessor(config)
        num_classes = 2

        path_train = "./data/politihop/train.tsv"
        path_dev = "./data/politihop/dev.tsv"
        path_test = "./data/politihop/test.tsv"

    elif config["dataset"] == "hover":
        processor = HoverProcessor(config)
        num_classes = 2

        path_train = "./data/hover/train.json"
        path_dev = "./data/hover/dev.json"
        path_test = "./data/hover/test.json"

    else:
        raise ValueError(
            f"{config['dataset']} is not a valid database name (choose between 'avtc', 'fever', 'feversymmetric', 'scifact', 'vitaminc')")

    config["num_classes"] = num_classes
    if config["test_only"]:
        if config["dataset"] == "scifact":
            data_test = processor.read_input_files(path_dev, name="dev", add_space=add_trigger_tokens, add_trigger_tokens=add_trigger_tokens)
        else:
            data_test = processor.read_input_files(path_test, name="dev", add_space=add_trigger_tokens, add_trigger_tokens=add_trigger_tokens)

        test_set = dataset(data_test)
        test_dataloader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
    else:
        data_train = processor.read_input_files(path_train, name="train", add_space=add_trigger_tokens, add_trigger_tokens=add_trigger_tokens)
        data_dev = processor.read_input_files(path_dev, name="dev", add_space=add_trigger_tokens, add_trigger_tokens=add_trigger_tokens)
        data_test = processor.read_input_files(path_test, name="test", add_space=add_trigger_tokens, add_trigger_tokens=add_trigger_tokens)
        if config["dataset"] == "scifact":
            # scifact test set is blind, so we use 20% of train as dev, and the dev as test
            tmp = data_dev
            data_dev = data_train[int(len(data_train) * 0.8):]
            data_train = data_train[:int(len(data_train) * 0.8)]
            data_test = tmp

        if positions is not None:
            data_train = get_samples_by_position(data_train, positions[0])
            data_dev = get_samples_by_position(data_dev, positions[1])
            data_test = get_samples_by_position(data_test, positions[2])

        if return_data:
            return data_train, data_dev, data_test

            #print(f"Train samples: {len(data_train)}\nDev samples: {len(data_dev)}\nTest samples: {len(data_test)}")

        train_set = dataset(data_train)
        dev_set = dataset(data_dev)
        test_set = dataset(data_test)

        train_dataloader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
        dev_dataloader = DataLoader(dev_set, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

        return train_dataloader, dev_dataloader, test_dataloader, num_classes

def defend(predictor, dataloader, trigger_ids=None, verbose=False):
    val_labels, val_preds = [], []
    for batch in dataloader:
        batch = tuple(
            t.to(get_device()) if not isinstance(t, list) and not isinstance(t, str) else t for t in batch)
        if trigger_ids is None:
            claim, evidence, ids_sent1, segs_sent1, att_mask_sent1, labels = batch
            trigger_mask = None
        else:
            claim, evidence, ids_sent1, segs_sent1, att_mask_sent1, trigger_mask, labels = batch

        with torch.no_grad():
            out = predictor(ids_sent1, segs_sent1, att_mask_sent1, trigger_mask=trigger_mask, trigger_ids=trigger_ids, verbose=True)
            preds = torch.max(out, 1)[1].cpu().numpy().tolist()
            labels_pos = torch.max(labels, 1)[1].cpu().numpy().tolist()
            val_labels.extend(labels_pos)
            val_preds.extend(preds)

    return val_preds, val_labels

def get_matching_samples(predictions, targets):
    assert len(predictions) == len(targets)
    matching_samples = []
    for i in range(len(predictions)):
        if predictions[i] == targets[i]:
            matching_samples.append(i)

    return matching_samples

def evaluation_fn(predictions, labels):
    assert len(predictions) == len(labels)
    preds = torch.max(predictions, 1)[1].cpu().numpy().tolist()
    labels_pos = torch.max(labels, 1)[1].cpu().numpy().tolist()

    return sum(p == l for p, l in zip(preds, labels_pos))

def get_new_labels(labels, args, to_class):
    batch_size = labels.size(0)
    if to_class == "support":
        if args.num_classes == 3:
            new_labels = torch.tensor([[1, 0, 0]] * batch_size, dtype=torch.float, device=labels.device)
        else:
            new_labels = torch.tensor([[1, 0]] * batch_size, dtype=torch.float, device=labels.device)
    elif to_class == "refute":
        if args.num_classes == 3:
            new_labels = torch.tensor([[0, 1, 0]] * batch_size, dtype=torch.float, device=labels.device)
        else:
            new_labels = torch.tensor([[0, 1]] * batch_size, dtype=torch.float, device=labels.device)
    else:
        new_labels = torch.tensor([[0, 0, 1]] * batch_size, dtype=torch.float, device=labels.device)

    return new_labels

def evaluate(predictor, tokenizer, embeddings, args, train_loader, test_loader, vocabulary, topklist, to_class):
    tmp_train_loader = deepcopy(train_loader)
    tmp_test_loader = deepcopy(test_loader)
    device = get_device()
    embedding_gradient = GradientStorage(embeddings)
    trigger_ids = [tokenizer.mask_token_id] * args.num_trigger_tokens
    trigger_ids = torch.tensor(trigger_ids, device=device).unsqueeze(0)

    memory = []
    loss_fn = torch.nn.CrossEntropyLoss()

    for i in tqdm(range(args.iters), "Number of epoch"):
        predictor.fc_model.zero_grad()

        pbar = tqdm(range(args.accumulation_steps))
        train_iter = iter(tmp_train_loader)
        averaged_grad = None

        # Accumulate
        for step in pbar:
            try:
                batch = next(train_iter)
                batch = tuple(t.to(device) if not isinstance(t, list) and not isinstance(t, str) else t for t in batch)
                claim, evidence, ids_sent, segs_sent, att_mask_sent, trigger_mask, labels = batch
            except:
                """logger.warning(
                    'Insufficient data for number of accumulation steps. '
                    'Effective batch size will be smaller than specified.'
                )"""
                break

            new_labels = get_new_labels(labels, args, to_class)
            predict_logits = predictor(ids_sent, segs_sent, att_mask_sent, trigger_mask, trigger_ids)
            loss = loss_fn(predict_logits, new_labels.float())  # get_loss(predict_logits, labels).mean()
            loss.backward()

            grad = embedding_gradient.get()
            bsz, _, emb_dim = grad.size()
            selection_mask = trigger_mask.unsqueeze(-1)  # probably needs a .squeeze(0) also
            selection_mask = selection_mask.to(torch.bool)
            grad = torch.masked_select(grad, selection_mask)
            grad = grad.view(bsz, args.num_trigger_tokens, emb_dim)

            if averaged_grad is None:
                averaged_grad = grad.sum(dim=0) / args.accumulation_steps
            else:
                averaged_grad += grad.sum(dim=0) / args.accumulation_steps

        pbar = tqdm(range(args.accumulation_steps))

        #token_to_flip = random.randrange(args.num_trigger_tokens)
        candidates, similarity = hotflip_attack(averaged_grad, #[token_to_flip],
                                    embeddings.weight,
                                    vocabulary,
                                    opposite=args.opposite,
                                    num_candidates=args.k,
                                    memory=memory)

        memory = candidates.tolist()
        current_score = 0
        candidate_scores = torch.zeros(args.k, device=device)
        denom = 0
        #tmp_test_loader = deepcopy(test_loader)
        test_iter = iter(tmp_test_loader)

        predictor.fc_model.eval()
        for step in pbar:
            try:
                batch = next(test_iter)
                batch = tuple(t.to(device) if not isinstance(t, list) and not isinstance(t, str) else t for t in batch)
                claim, evidence, ids_sent, segs_sent, att_mask_sent, trigger_mask, labels = batch
            except:
                """logger.warning(
                    'Insufficient data for number of accumulation steps. '
                    'Effective batch size will be smaller than specified.'
                )"""
                break

            new_labels = get_new_labels(labels, args, to_class)
            with torch.no_grad():
                predict_logits = predictor(ids_sent, segs_sent, att_mask_sent, trigger_mask, trigger_ids)
                eval_metric = evaluation_fn(predict_logits, new_labels)

            # Update current score
            current_score += eval_metric
            denom += labels.size(0)

            # NOTE: Instead of iterating over tokens to flip we randomly change just one each
            # time so the gradients don't get stale.
            for i, candidate in enumerate(candidates):
                temp_trigger = trigger_ids.clone()
                values = vocabulary[args.num_trigger_tokens][list(vocabulary[args.num_trigger_tokens].keys())[candidate]]
                assert len(values) == args.num_trigger_tokens

                temp_trigger[:, :args.num_trigger_tokens] = values
                with torch.no_grad():
                    predict_logits = predictor(ids_sent, segs_sent, att_mask_sent, trigger_mask, temp_trigger)
                    eval_metric = evaluation_fn(predict_logits, new_labels)

                candidate_scores[i] += eval_metric

        for j in range(len(candidates)):
            candidate_scores[j] /= denom
            topklist.add(candidate_scores, j, list(vocabulary[args.num_trigger_tokens].keys())[candidates[j]], similarity[j], args.num_trigger_tokens)

        current_score = current_score / denom

        if topklist.data_candidates[0][0] > current_score:
            logger.info('Better trigger detected.')
            best_candidate_score = topklist.data_candidates[0][0]
            best_candidate_idx = topklist.data_candidates[0][1]
            values = vocabulary[args.num_trigger_tokens][list(vocabulary[args.num_trigger_tokens].keys())[candidates[best_candidate_idx]]]
            trigger_ids[:, :args.num_trigger_tokens] = values
            logger.info(f"Old score: {current_score: 0.4f}")
            logger.info(f'Test metric: {best_candidate_score: 0.4f}')
        else:
            logger.info(f'Found trigger is worse. Keeping triggers {trigger_ids[0,:args.num_trigger_tokens]} with ')
            #continue

        topklist.merge()

    return topklist

def run_model(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train, dev, test, num_classes = get_data(vars(args), add_trigger_tokens=False)
    args.num_classes = num_classes

    model, tokenizer = load_pretrained(args.model_name, args.backbone, args)
    model.to(device)
    embeddings = get_embeddings(model)
    predictor = PredictWrapper(model)

    words_df = pd.read_csv("antonym_pairs.csv")
    words = []
    for i, row in words_df.iterrows():
        words.append(words_df.loc[i,"Word1"])
        words.append(words_df.loc[i,"Word2"])
    words = sorted(set(words)) #list(set(words))

    vocabulary = {}
    for i, word in enumerate(words):
        tokens = tokenizer(word, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        num_tokens = len(tokens)
        if num_tokens not in vocabulary.keys():
            vocabulary[num_tokens] = {}
        vocabulary[num_tokens][word] = tokens

    vocabulary = dict(sorted(vocabulary.items(), key=lambda x: x[0], reverse=False))

    # get fc correct predictions
    print("Predicting train samples...")
    predictions_train, targets_train = defend(predictor, train)
    print("Predicting dev samples...")
    predictions_dev, targets_dev = defend(predictor, dev)
    print("Predicting test samples...")
    predictions_test, targets_test = defend(predictor, test)

    train_match, dev_match, test_match = (get_matching_samples(predictions_train, targets_train),
                                          get_matching_samples(predictions_dev, targets_dev),
                                          get_matching_samples(predictions_test, targets_test))

    topklist_support = TopKList(args.k, args.sort_by_similarity, args.opposite)
    topklist_refute = TopKList(args.k, args.sort_by_similarity, args.opposite)
    topklist_nei = TopKList(args.k, args.sort_by_similarity, args.opposite)
    dir_path = f"results/{args.dataset}/iters{args.iters}/k{args.k}/batch_size{args.batch_size}/acc_steps_{args.accumulation_steps}/train_{args.train}/"
    os.makedirs(dir_path, exist_ok=True)
    results = []
    for i,k in enumerate(vocabulary.keys()):
        result = []
        args.num_trigger_tokens = k
        train, dev, test = get_data(vars(args), positions=[train_match, dev_match, test_match], return_data=True,
                                      add_trigger_tokens=True)
        test = pd.DataFrame(test)
        train = pd.DataFrame(train)
        dev = pd.DataFrame(dev)

        test_samples = test[test.iloc[:, -1].apply(lambda x: x[0] == 0)].values.tolist()
        test_loader = DataLoader(dataset(test_samples), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_trigger)
        dev_samples = dev[dev.iloc[:, -1].apply(lambda x: x[0] == 0)].values.tolist()
        dev_loader = DataLoader(dataset(dev_samples), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_trigger)

        if args.train:
            train_samples = train[train.iloc[:, -1].apply(lambda x: x[0] == 0)].values.tolist()
            train_loader = DataLoader(dataset(train_samples), batch_size=args.batch_size, shuffle=False,
                                      collate_fn=collate_fn_trigger)

        print("Evaluating support...")
        if args.train:
            topklist_support = evaluate(predictor, tokenizer, embeddings, args, train_loader, dev_loader, vocabulary, topklist_support, to_class="support")
        else:
            topklist_support = evaluate(predictor, tokenizer, embeddings, args, test_loader, dev_loader, vocabulary, topklist_support, to_class="support")

        mean = sum([el[0] for el in topklist_support.data])/len(topklist_support.data)
        result.append((mean, str(topklist_support.data)))

        print(f"Current best support at tokens {args.num_trigger_tokens}: {topklist_support.data}")
        print(f"Mean score: {mean}")

        test_samples = test[test.iloc[:, -1].apply(lambda x: x[1] == 0)].values.tolist()
        test_loader = DataLoader(dataset(test_samples), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_trigger)
        dev_samples = dev[dev.iloc[:, -1].apply(lambda x: x[1] == 0)].values.tolist()
        dev_loader = DataLoader(dataset(dev_samples), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_trigger)

        if args.train:
            train_samples = train[train.iloc[:, -1].apply(lambda x: x[1] == 0)].values.tolist()
            train_loader = DataLoader(dataset(train_samples), batch_size=args.batch_size, shuffle=False,
                                      collate_fn=collate_fn_trigger)
        print("Evaluating refute...")
        if args.train:
            topklist_refute = evaluate(predictor, tokenizer, embeddings, args, train_loader, dev_loader, vocabulary, topklist_refute,
                                       to_class="refute")
        else:
            topklist_refute = evaluate(predictor, tokenizer, embeddings, args, test_loader, dev_loader, vocabulary, topklist_refute, to_class="refute")

        mean = sum([el[0] for el in topklist_refute.data])/len(topklist_refute.data)
        result.append((mean, str(topklist_refute.data)))

        print(f"Current best refute at tokens {args.num_trigger_tokens}: {topklist_refute.data}")
        print(f"Mean score: {mean}")

        if num_classes == 3:
            test_samples = test[test.iloc[:, -1].apply(lambda x: x[2] == 0)].values.tolist()
            test_loader = DataLoader(dataset(test_samples), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_trigger)
            dev_samples = dev[dev.iloc[:, -1].apply(lambda x: x[2] == 0)].values.tolist()
            dev_loader = DataLoader(dataset(dev_samples), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_trigger)
            if args.train:
                train_samples = train[train.iloc[:, -1].apply(lambda x: x[2] == 0)].values.tolist()
                train_loader = DataLoader(dataset(train_samples), batch_size=args.batch_size, shuffle=False,
                                          collate_fn=collate_fn_trigger)
            print("Evaluating nei...")
            if args.train:
                topklist_nei = evaluate(predictor, tokenizer, embeddings, args, train_loader, dev_loader, vocabulary, topklist_nei,to_class="nei")
            else:
                topklist_nei = evaluate(predictor, tokenizer, embeddings, args, test_loader, dev_loader, vocabulary, topklist_nei, to_class="nei")

            mean = sum([el[0] for el in topklist_nei.data]) / len(topklist_nei.data)
            result.append((mean, str(topklist_nei.data)))
            print(f"Current best nei at tokens {args.num_trigger_tokens}: {topklist_nei.data}")
            print(f"Mean score: {mean}")
            results.append(result)

    results = pd.DataFrame(results, columns=["Support", "Refute", "Nei"] if num_classes == 3 else ["Support", "Refute"])
    results.to_csv(os.path.join(dir_path, "results.csv"), index=False)

    if args.train:
        support_score = 0
        trigger_ids = [tokenizer.mask_token_id] * args.num_trigger_tokens
        trigger_ids = torch.tensor(trigger_ids, device=device).unsqueeze(0)
        topklist_support.data.sort(key=lambda el: el[2])

        for el in topklist_support.data:
            args.num_trigger_tokens = el[-2]
            word = el[2]
            _, _, test = get_data(vars(args), positions=[train_match, dev_match, test_match], return_data=True,
                                      add_trigger_tokens=True)
            test = pd.DataFrame(test)
            test_samples = test[test.iloc[:, -1].apply(lambda x: x[0] == 0)].values.tolist()
            for i in range(len(test_samples)):
                test_samples[i][-1] = [1,0,0] if num_classes == 3 else [1,0]
            test_loader = DataLoader(dataset(test_samples), batch_size=args.batch_size, shuffle=False,
                                     collate_fn=collate_fn_trigger)
            temp_trigger = trigger_ids.clone()
            values = vocabulary[args.num_trigger_tokens][word]
            temp_trigger[:, :args.num_trigger_tokens] = values
            predictions, targets = defend(predictor, test_loader, temp_trigger, verbose=True)

            support_score += sum([pred == target for pred, target in zip(predictions, targets)]) / len(predictions)

        refute_score = 0

        for el in topklist_refute.data:
            args.num_trigger_tokens = el[-2]
            word = el[2]
            _, _, test = get_data(vars(args), positions=[train_match, dev_match, test_match], return_data=True,
                                  add_trigger_tokens=True)
            test = pd.DataFrame(test)
            test_samples = test[test.iloc[:, -1].apply(lambda x: x[1] == 0)].values.tolist()
            for i in range(len(test_samples)):
                test_samples[i][-1] = [0,1,0] if num_classes == 3 else [0,1]
            test_loader = DataLoader(dataset(test_samples), batch_size=args.batch_size, shuffle=False,
                                     collate_fn=collate_fn_trigger)
            temp_trigger = trigger_ids.clone()
            values = vocabulary[args.num_trigger_tokens][word]
            temp_trigger[:, :args.num_trigger_tokens] = values
            predictions, targets = defend(predictor, test_loader, temp_trigger)
            refute_score += sum([pred == target for pred, target in zip(predictions, targets)]) / len(predictions)

        nei_score = 0

        if num_classes == 3:
            for el in topklist_nei.data:
                args.num_trigger_tokens = el[-2]
                word = el[2]
                _, _, test = get_data(vars(args), positions=[train_match, dev_match, test_match], return_data=True,
                                      add_trigger_tokens=True)
                test = pd.DataFrame(test)
                test_samples = test[test.iloc[:, -1].apply(lambda x: x[2] == 0)].values.tolist()
                for i in range(len(test_samples)):
                    test_samples[i][-1] = [0, 0, 1]
                test_loader = DataLoader(dataset(test_samples), batch_size=args.batch_size, shuffle=False,
                                         collate_fn=collate_fn_trigger)
                temp_trigger = trigger_ids.clone()
                values = vocabulary[args.num_trigger_tokens][word]
                temp_trigger[:, :args.num_trigger_tokens] = values
                predictions, targets = defend(predictor, test_loader, temp_trigger)
                nei_score += sum([pred == target for pred, target in zip(predictions, targets)]) / len(predictions)

        print("*** TEST RESULT for AUTOPROMPT ***")
        print(f"Support: {round(support_score/len(topklist_support.data), 4)}")
        print(f"Refute: {round(refute_score/len(topklist_refute.data), 4)}")
        results = [[round(support_score / len(topklist_support.data), 4), round(refute_score / len(topklist_refute.data), 4)]]
        if num_classes == 3:
            print(f"Nei: {round(nei_score/len(topklist_nei.data), 4)}")
            results[0].append(round(nei_score/len(topklist_nei.data), 4))

        results = pd.DataFrame(results, columns=["Support", "Refute", "Nei"] if num_classes == 3 else ["Support", "Refute"])
        results.to_csv(os.path.join(dir_path, "results_test.csv"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='roberta-base', help='Model name')
    parser.add_argument('--backbone', type=str, default=None, help='Model name')
    parser.add_argument('--avg_tokens', action='store_true', help='Use average tokens (default: False)')
    parser.add_argument('--freeze', action='store_true', help='Freeze backbone layers (default: False)')
    parser.add_argument('--embed_size', type=int, default=768, help='Embedding size')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--max_sent_len', type=int, default=512, help='Maximum sentence length')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--ms_train', type=int, default=None, help='Max number of samples for train split')
    parser.add_argument('--ms_dev', type=int, default=None, help='Max number of samples for dev split')
    parser.add_argument('--ms_test', type=int, default=None, help='Max number of samples for test split')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--test_only', action="store_true", help='Do not train')
    parser.add_argument('--balanced', action="store_true", help='Use balanced dataset')
    parser.add_argument('--return_sentences', action="store_true", help='Make the dataset processor return also textual claims and evidence')
    parser.add_argument('--facteval', action="store_true", help='Use data generated by FactEval')
    parser.add_argument('--adv_llm', action="store_true", help='Use data generated by adversarial LLM')
    parser.add_argument('--scheduler', action="store_true", help='Use lr scheduler')
    parser.add_argument('--potency', action="store_true", help='Extract "potency" words to flip model predictions')
    parser.add_argument('--stereotype', action="store_true", help='Extract "stereotype" words to flip model predictions')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--iters', type=int, default=1,
                        help='Number of iterations to run trigger search algorithm')
    parser.add_argument('--accumulation-steps', type=int, default=10)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--sort_by_similarity', action="store_true", help='Sort by similarity (HotFlip)')
    parser.add_argument('--train', action="store_true", help='Use training set')
    parser.add_argument('--opposite', action="store_true", help='Find the most dissimilar values')

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    run_model(args)
