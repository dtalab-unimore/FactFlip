import torch
import codecs
import json
import os
import random
import pandas as pd
import ast

from torch.utils.data import Dataset
from transformers import AutoTokenizer, pipeline
from sklearn.preprocessing import OneHotEncoder
from transformers import pipeline
from tqdm import tqdm
import unicodedata

import re

from autoprompt.utils import add_task_specific_tokens

def remove_accents(text):
    # Remove all patterns like \uXXXX (where X is a hex digit)
    return re.sub(r'\\u[0-9a-fA-F]{4}', '', text)


class dataset(Dataset):
    def __init__(self, examples):
        super(dataset, self).__init__()
        self.examples = examples

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)


def collate_fn(examples):
    claim, evidence, ids_sent1, segs_sent1, att_mask_sent1, labels = map(list, zip(*examples))

    ids_sent1 = torch.tensor(ids_sent1, dtype=torch.long)
    segs_sent1 = torch.tensor(segs_sent1, dtype=torch.long)
    att_mask_sent1 = torch.tensor(att_mask_sent1, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return claim, evidence, ids_sent1, segs_sent1, att_mask_sent1, labels

def collate_fn_trigger(examples):
    claim, evidence, ids_sent1, segs_sent1, att_mask_sent1, trigger_mask, labels = map(list, zip(*examples))

    ids_sent1 = torch.tensor(ids_sent1, dtype=torch.long)
    segs_sent1 = torch.tensor(segs_sent1, dtype=torch.long)
    att_mask_sent1 = torch.tensor(att_mask_sent1, dtype=torch.long)
    trigger_mask1 = torch.tensor(trigger_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return claim, evidence, ids_sent1, segs_sent1, att_mask_sent1, trigger_mask1, labels

def collate_fn_antonym(examples):
    """print(examples[0])
    print(len(examples[0]))"""
    try:
        sent1, sent2, ids_sent1, segs_sent1, att_mask_sent1, ids_sent2, segs_sent2, att_mask_sent2, label = map(list, zip(*examples))
    except:
        sent1, sent2, ids_sent1, segs_sent1, att_mask_sent1, ids_sent2, segs_sent2, att_mask_sent2 = map(list, zip(*examples))
        label = None

    ids_sent1 = torch.tensor(ids_sent1, dtype=torch.long)
    segs_sent1 = torch.tensor(segs_sent1, dtype=torch.long)
    att_mask_sent1 = torch.tensor(att_mask_sent1, dtype=torch.long)
    ids_sent2 = torch.tensor(ids_sent2, dtype=torch.long)
    segs_sent2 = torch.tensor(segs_sent2, dtype=torch.long)
    segs_sent2 = torch.tensor(segs_sent2, dtype=torch.long)
    att_mask_sent2 = torch.tensor(att_mask_sent2, dtype=torch.long)
    if label is not None:
        label = torch.tensor(label, dtype=torch.long)
        return sent1, sent2, ids_sent1, segs_sent1, att_mask_sent1, ids_sent2, segs_sent2, att_mask_sent2, label

    return sent1, sent2, ids_sent1, segs_sent1, att_mask_sent1, ids_sent2, segs_sent2, att_mask_sent2


class DataProcessor:

  def __init__(self,config):
    self.config = config
    if "num_trigger_tokens" in config.keys():
        self.num_trigger_tokens = config['num_trigger_tokens']
    else:
        self.num_trigger_tokens = None

    if self.config["backbone"] is None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
    else:
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["backbone"])

    add_task_specific_tokens(self.tokenizer)
    self.max_sent_len = config["max_sent_len"]

  def __str__(self,):
    pattern = """General data processor: \n\n Tokenizer: {}\n\nMax sentence length: {}""".format(self.config["model_name"], self.max_sent_len)
    return pattern

  def _get_examples(self, dataset, dataset_type="train", add_space=False, add_trigger_tokens=False):
    examples = []
    count_truncated_samples = 0

    for row in tqdm(dataset, desc="tokenizing..."):
      id, claim, evidence, label = row
      if add_space:
          claim = ". "+claim

      """
      for the first sentence
      """

      claim_length = len(self.tokenizer.encode(claim))
      evidence_length = len(self.tokenizer.encode(evidence))

      ids_sent1 = self.tokenizer.encode(claim, evidence)
      segs_sent1 = [0] * claim_length + [1] * (evidence_length)

      if self.num_trigger_tokens is not None:
          ids_sent1 = [ids_sent1[0]] + [self.tokenizer.trigger_token_id] * self.num_trigger_tokens + ids_sent1[1:]
          segs_sent1 = [0] * self.num_trigger_tokens + segs_sent1

      assert len(ids_sent1) == len(segs_sent1)

      pad_id = self.tokenizer.encode(self.tokenizer.pad_token, add_special_tokens=False)[0]

      if len(ids_sent1) < self.max_sent_len:
        res = self.max_sent_len - len(ids_sent1)
        att_mask_sent1 = [1] * len(ids_sent1) + [0] * res
        ids_sent1 += [pad_id] * res
        segs_sent1 += [0] * res
      else:
        ids_sent1 = ids_sent1[:self.max_sent_len]
        segs_sent1 = segs_sent1[:self.max_sent_len]
        att_mask_sent1 = [1] * self.max_sent_len
        count_truncated_samples += 1

      if self.num_trigger_tokens is not None and add_trigger_tokens:
          trigger_mask = [0] * (len(ids_sent1) - self.num_trigger_tokens - 1)
          trigger_mask = [0] + [1] * self.num_trigger_tokens + trigger_mask # the initial [0] is for the <s> token
          assert len(ids_sent1) == len(trigger_mask)

          example = [claim, evidence, ids_sent1, segs_sent1, att_mask_sent1, trigger_mask, label]
      else:
          example = [claim, evidence, ids_sent1, segs_sent1, att_mask_sent1, label]

      examples.append(example)

    print(f"finished preprocessing examples in {dataset_type}: {count_truncated_samples} samples truncated out of {len(dataset)}")

    return examples

class AntonymsProcessor(DataProcessor):

    def __init__(self, config):
        super(AntonymsProcessor, self).__init__(config)

    def _get_examples(self, dataset, dataset_type="train", add_space=False):
        examples = []
        count_truncated_samples = 0
        pad_id = self.tokenizer.encode(self.tokenizer.pad_token, add_special_tokens=False)[0]

        for row in tqdm(dataset, desc="tokenizing..."):
            if len(row) == 4:
                id, sentence1, sentence2, label = row
            else:
                id, sentence1, sentence2 = row

            ids_sent1 = self.tokenizer.encode(sentence1)
            segs_sent1 = [0] * len(ids_sent1)

            ids_sent2 = self.tokenizer.encode(sentence2)
            segs_sent2 = [0] * len(ids_sent2)

            if len(ids_sent1) < self.max_sent_len:
                res = self.max_sent_len - len(ids_sent1)
                att_mask_sent1 = [1] * len(ids_sent1) + [0] * res
                ids_sent1 += [pad_id] * res
                segs_sent1 += [0] * res
            else:
                ids_sent1 = ids_sent1[:self.max_sent_len]
                segs_sent1 = segs_sent1[:self.max_sent_len]
                att_mask_sent1 = [1] * self.max_sent_len
                count_truncated_samples += 1

            if len(ids_sent2) < self.max_sent_len:
                res = self.max_sent_len - len(ids_sent2)
                att_mask_sent2 = [1] * len(ids_sent2) + [0] * res
                ids_sent2 += [pad_id] * res
                segs_sent2 += [0] * res
            else:
                ids_sent2 = ids_sent2[:self.max_sent_len]
                segs_sent2 = segs_sent2[:self.max_sent_len]
                att_mask_sent2 = [1] * self.max_sent_len
                count_truncated_samples += 1

            if len(row) == 4:
                example = [sentence1, sentence2, ids_sent1, segs_sent1, att_mask_sent1, ids_sent2, segs_sent2, att_mask_sent2, label]
            else:
                example = [sentence1, sentence2, ids_sent1, segs_sent1, att_mask_sent1, ids_sent2, segs_sent2, att_mask_sent2]

            examples.append(example)

        print(
            f"finished preprocessing examples in {dataset_type}: {count_truncated_samples} samples truncated out of {len(dataset)}")

        return examples

    def read_input_files(self, file_path, name="train", return_sentences=False, **kwargs):
        df = pd.read_csv(file_path)
        df = df.reset_index(drop=False)
        result = df.values.tolist()

        if return_sentences and not self.config["skip_tokenizer"]:
            raise ValueError("return_sentence and skip_tokenizer are not mutually exclusive")
        if return_sentences:
            return result

        examples = self._get_examples(result, name)
        return examples

class FeverProcessor(DataProcessor):

    def __init__(self, config):
        super(FeverProcessor, self).__init__(config)
        self.data_path = "data/fever/wiki-pages/"
        self.data = {}

    def load_data(self):
        if len(self.data) > 0:
            return
        self.data = {}
        for file_name in os.listdir(self.data_path):
            if file_name.split(".")[-1] != "jsonl":
                continue
            path = os.path.join(self.data_path, file_name)
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line)
                    n_line_id = line["id"]
                    if n_line_id in self.data.keys():
                        raise ValueError("duplicate evidence ids")
                    self.data[n_line_id] = line["lines"]


    def get_random_sentence(self):
        files = []
        for file_name in os.listdir(self.data_path):
            if file_name.split(".")[-1] != "jsonl":
                continue
            path = os.path.join(self.data_path, file_name)
            files.append(path)

        file_path = random.choice(files)
        with open(file_path, 'r', encoding='utf-8') as f:
            file_size = os.path.getsize(file_path)
            random_pos = random.randint(0, file_size - 1)
            f.seek(random_pos)

            # discard partial line
            f.readline()
            # read next full line
            # print(json.loads(f.readline()))
            random_line = json.loads(f.readline())["lines"].split("\n")
            random_line = random.choice(random_line).split("\t",1)[-1]

        return random_line

    def read_input_files(self, file_path, name="train", return_sentences=False, add_space=False, add_trigger_tokens=False):
        if "balanced" in file_path:
            df = pd.read_csv(file_path)
            df['label'] = df['label'].apply(ast.literal_eval)
            if not return_sentences:
                df = df.drop(["topic", "balanced"], axis="columns")
            else:
                df['topic'] = df['topic'].astype(int)
            result = df.values.tolist()
        else:
            claims, evidences, labels = [], [], []
            self.load_data()

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line)
                    if line["label"] == "NOT ENOUGH INFO": #todo: fix 3labels when extracting random sentence
                        continue

                    if line["label"] == "SUPPORTS":
                        label = [1,0,0]
                    elif line["label"] == "REFUTES":
                        label = [0,1,0]
                    elif line["label"] == "NOT ENOUGH INFO":
                        label = [0,0,1]
                    else:
                        raise ValueError(f"unknown label {line['label']}")

                    if line["label"] == "NOT ENOUGH INFO":
                        evidence = [self.get_random_sentence()]
                    else:
                        evidence = []
                        for evidence_set in line["evidence"]:
                            text = ""
                            for i, evidence_piece in enumerate(evidence_set):
                                if evidence_piece[2] not in self.data.keys():
                                    continue
                                sentence = self.data[evidence_piece[2]].split("\n")[evidence_piece[3]].split("\t",1)[-1]
                                if i > 0:
                                    text+="\n"
                                text+=sentence
                            evidence.append(text)

                    found = False
                    for ev in evidence:
                        if len(ev) != 0:
                            found = True
                            evidences.append(evidence[0]) #for simplicity, I only consider the first evidence set
                            break
                    if not found:
                        continue

                    claims.append(line["claim"])
                    labels.append(label)


            result = []
            for i, (claim, evidence, label) in enumerate(zip(claims, evidences, labels)):
                result.append([i, claim, evidence, label]) #int, string, string, list[int]

        if return_sentences and not self.config["skip_tokenizer"]:
            raise ValueError("return_sentence and skip_tokenizer are not mutually exclusive")
        if return_sentences:
            return result

        examples = self._get_examples(result, name, add_space=add_space, add_trigger_tokens=add_trigger_tokens)
        return examples


class FeverSymmetricProcessor(DataProcessor):

    def __init__(self, config):
        super(FeverSymmetricProcessor, self).__init__(config)

    def read_input_files(self, file_path, name="train", return_sentences=False, add_space=False, add_trigger_tokens=False):
        if "balanced" in file_path:
            df = pd.read_csv(file_path)
            df['label'] = df['label'].apply(ast.literal_eval)
            if not return_sentences:
                df = df.drop(["topic", "balanced"], axis="columns")
            else:
                df['topic'] = df['topic'].astype(int)
            result = df.values.tolist()
        else:
            claims, evidences, labels = [], [], []

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line)
                    claims.append(line["claim"])
                    if line["label"] == "SUPPORTS":
                        label = [1,0,0]
                    elif line["label"] == "REFUTES":
                        label = [0,1,0]
                    elif line["label"] == "NOT ENOUGH INFO":
                        label = [0,0,1]
                    else:
                        raise ValueError(f"unknown label {line['label']}")
                    labels.append(label)

                    evidences.append(line["evidence_sentence"])

            result = []
            for i, (claim, evidence, label) in enumerate(zip(claims, evidences, labels)):
                result.append([i, claim, evidence, label]) #int, string, string, list[int]

        if return_sentences and not self.config["skip_tokenizer"]:
            raise ValueError("return_sentence and skip_tokenizer are not mutually exclusive")
        if return_sentences:
            return result

        examples = self._get_examples(result, name, add_space=add_space, add_trigger_tokens=add_trigger_tokens)
        return examples


class VitamincProcessor(DataProcessor):

    def __init__(self, config):
        super(VitamincProcessor, self).__init__(config)

    def read_input_files(self, file_path, name="train", add_space=False, add_trigger_tokens=False):
        claims, evidences, labels = [], [], []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                claims.append(line["claim"])
                if line["label"] == "SUPPORTS":
                    label = [1,0,0]
                elif line["label"] == "REFUTES":
                    label = [0,1,0]
                elif line["label"] == "NOT ENOUGH INFO":
                    label = [0,0,1]
                else:
                    raise ValueError(f"unknown label {line['label']}")
                labels.append(label)

                evidences.append(line["evidence"])

        result = []
        for i, (claim, evidence, label) in enumerate(zip(claims, evidences, labels)):
            result.append([i, claim, evidence, label]) #int, string, string, list[int]

        examples = self._get_examples(result, name, add_space=add_space, add_trigger_tokens=add_trigger_tokens)
        return examples


class SciFactProcessor(DataProcessor):

    def __init__(self, config):
        super(SciFactProcessor, self).__init__(config)
        self.data_path = "data/scifact/corpus.jsonl"
        self.data = {}

    def load_data(self):
        if len(self.data) > 0:
            return
        self.data = {}
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                if line["doc_id"] in self.data.keys():
                    raise ValueError("duplicate evidence ids")
                self.data[line["doc_id"]] = line["abstract"]

    def get_random_sentence(self):
        while True:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                file_size = os.path.getsize(self.data_path)
                random_pos = random.randint(0, file_size - 1)
                f.seek(random_pos)

                # discard partial line
                f.readline()
                # read next full line
                try:
                    random_line = json.loads(f.readline())["abstract"]
                except:
                    continue
                random_line = random.choice(random_line)
                break

        return random_line

    def read_input_files(self, file_path, name="train", add_space=False, add_trigger_tokens=False):
        claims, evidences, labels = [], [], []
        count_support, count_refute, count_not_enough = 0, 0, 0
        self.load_data()

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                claims.append(line["claim"])
                if "evidence" not in line.keys() or len(line["evidence"]) == 0:
                    label_txt = "NOT ENOUGH INFORMATION"
                    label = [0,0,1]
                    count_not_enough += 1
                else:
                    first_key = next(iter(line["evidence"]))
                    label_txt = line["evidence"][first_key][0]["label"] #each claim has one unique label (either SUPPORT or CONTRADICT)
                    if label_txt == "SUPPORT":
                        label = [1,0,0]
                        count_support += 1
                    elif label_txt == "CONTRADICT":
                        label = [0,1,0]
                        count_refute += 1
                    else:
                        raise ValueError(f"unknown label {label_txt}")

                labels.append(label)

                if label_txt == "NOT ENOUGH INFORMATION":
                    evidence = self.get_random_sentence()
                else:
                    evidence = ""
                    text = ""
                    for k,v in line["evidence"].items():
                        for i, evidence_piece in enumerate(v):
                            sentence = self.data[int(k)][evidence_piece["sentences"][0]]
                            if i > 0:
                                text+="\n"
                            text+=sentence
                    evidence = text
                evidences.append(evidence)

        result = []
        for i, (claim, evidence, label) in enumerate(zip(claims, evidences, labels)):
            result.append([i, claim, evidence, label]) #int, string, string, list[int]

        examples = self._get_examples(result, name, add_space=add_space, add_trigger_tokens=add_trigger_tokens)
        return examples


class AVTCProcessor(DataProcessor):

    def __init__(self, config):
        super(AVTCProcessor, self).__init__(config)

    def get_random_sentence(self, data):
        found = False
        sentence = ""

        while not found:
            sample = random.choice(data)
            if sample["label"] in ["Refuted", "Supported"]:
                found = True
                sentence = sample["questions"][0]["question"]
                if sample["questions"][0]["question"].strip()[-1] != "?":
                    sentence += "?"
                sentence += sample["questions"][0]["answers"][0]["answer"]

        return sentence

    def read_input_files(self, file_path, name="train", add_space=False, add_trigger_tokens=False):
        claims, evidences, labels = [], [], []

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for sample in data:
            if sample["label"] == "Supported":
                label = [1,0,0]
            elif sample["label"] == "Refuted":
                label = [0,1,0]
            elif sample["label"] == "Not Enough Evidence":
                label = [0,0,1]
            elif sample["label"] == "Conflicting Evidence/Cherrypicking":
                continue
            else:
                raise ValueError(f"unknown label: {sample['label']}")

            claims.append(sample["claim"])
            labels.append(label)

            evidence = ""
            if sample["label"] == "Not Enough Evidence":
                evidence = self.get_random_sentence(data)
            else:
                for qa in sample["questions"]:
                    evidence += qa["question"]
                    if qa["question"].strip()[-1] != "?":
                        evidence += "?"
                    evidence += qa["answers"][0]["answer"]+"\n"

            evidences.append(evidence)

        result = []
        for i, (claim, evidence, label) in enumerate(zip(claims, evidences, labels)):
            result.append([i, claim, evidence, label]) #int, string, list[string], list[int]

        examples = self._get_examples(result, name, add_space=add_space, add_trigger_tokens=add_trigger_tokens)
        return examples

class FM2Processor(DataProcessor):

    def __init__(self, config):
        super(FM2Processor, self).__init__(config)

    def read_input_files(self, file_path, name="train", add_space=False, add_trigger_tokens=False):
        claims, evidences, labels = [], [], []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)

                if line["label"] == "SUPPORTS":
                    label = [1,0]
                elif line["label"] == "REFUTES":
                    label = [0,1]
                else:
                    raise ValueError(f"unknown label {line['label']}")

                claims.append(line["text"])
                labels.append(label)
                evidence_text = ""
                for i, evidence in enumerate(line["gold_evidence"]):
                    if i > 0:
                        evidence_text += "\n"
                    evidence_text += evidence["text"]
                evidences.append(evidence_text)

            result = []
            for i, (claim, evidence, label) in enumerate(zip(claims, evidences, labels)):
                result.append([i, claim, evidence, label]) #int, string, string, list[int]

        examples = self._get_examples(result, name, add_space=add_space, add_trigger_tokens=add_trigger_tokens)
        return examples

class PolitiHopProcessor(DataProcessor):

    def __init__(self, config):
        super(PolitiHopProcessor, self).__init__(config)

    def read_input_files(self, file_path, name="train", add_space=False, add_trigger_tokens=False):
        claims, evidences, labels = [], [], []
        df = pd.read_csv(file_path, sep="\t")

        for i, line in df.iterrows():
            if line["annotated_label"] == "true":
                label = [1,0]
            elif line["annotated_label"] == "false":
                label = [0,1]
            elif line["annotated_label"] == "half-true":
                continue
            else:
                raise ValueError(f"unknown label {line['annotated_label']}")

            claims.append(line["statement"])
            evidence_text = ""
            rulings = eval(line["ruling"].strip())
            for k,v in eval(line["annotated_evidence"].strip()).items():
                elements = []
                for ev in v:
                    elements.extend(ev.split(","))
                for ev in elements:
                    if evidence_text != "":
                        evidence_text += "\n"
                    evidence_text += rulings[int(ev)]
            evidences.append(evidence_text)
            labels.append(label)

        result = []
        for i, (claim, evidence, label) in enumerate(zip(claims, evidences, labels)):
            result.append([i, claim, evidence, label]) #int, string, string, list[int]

        examples = self._get_examples(result, name, add_space=add_space, add_trigger_tokens=add_trigger_tokens)
        return examples

class HoverProcessor(DataProcessor):

    def __init__(self, config):
        super(HoverProcessor, self).__init__(config)


    def read_input_files(self, file_path, name="train", add_space=False, add_trigger_tokens=False):
        claims, evidences, labels = [], [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for line in data:
            if line["label"] == 0:
                label = [1, 0]
            elif line["label"] == 1:
                label = [0, 1]
            else:
                raise ValueError(f"unknown label {line['label']}")

            claims.append(line["claim"])
            labels.append(label)
            evidences.append(line["evidence"])

        result = []
        for i, (claim, evidence, label) in enumerate(zip(claims, evidences, labels)):
            result.append([i, claim, evidence, label])  # int, string, string, list[int]

        examples = self._get_examples(result, name, add_space=add_space, add_trigger_tokens=add_trigger_tokens)
        return examples

class FactEvalProcessor(DataProcessor):
    def __init__(self, config):
        super(FactEvalProcessor, self).__init__(config)

    def read_input_files(self, file_path, name="train", max_size=None):
        claims, evidences, labels = [], [], []
        df = pd.read_csv(file_path)
        if max_size is not None:
            freq = {}
            for i, line in df.iterrows():
                if line["type"] not in freq.keys():
                    freq[line["type"]] = []

                freq[line["type"]].append(line)

            """for key in freq:
                random.shuffle(freq[key])"""

            dict_keys = list(freq.keys())
            el_visited = [0 for _ in range(len(dict_keys))]
            for i in range(max_size):
                idx = i % len(dict_keys)
                key = dict_keys[idx]
                pos = el_visited[idx]
                line = freq[key][pos]
                if isinstance(line["label"], float) or isinstance(line["adv_claim"], float) or isinstance(
                        line["gold_evidence_text"], float):
                    continue
                el_visited[idx] += 1
                label = eval(line["label"].strip())
                claims.append(line["adv_claim"])
                labels.append(label)
                evidences.append(line["gold_evidence_text"])
        else:
            for i, line in df.iterrows():
                #if isinstance(line["label"])
                if isinstance(line["label"], float) or isinstance(line["adv_claim"], float) or isinstance(line["gold_evidence_text"], float):
                    continue
                label = eval(line["label"].strip())

                claims.append(line["adv_claim"])
                labels.append(label)
                evidences.append(line["gold_evidence_text"])

        result = []
        for i, (claim, evidence, label) in enumerate(zip(claims, evidences, labels)):
            result.append([i, claim, evidence, label])  # int, string, string, list[int]

        print(result[-1])

        examples = self._get_examples(result, name)
        return examples

class AdvLLMProcessor(DataProcessor):
    def __init__(self, config):
        super(AdvLLMProcessor, self).__init__(config)

    def read_input_files(self, file_path, name="train", max_size=None):
        claims, evidences, labels = [], [], []
        df = pd.read_csv(file_path)

        for i, line in df.iterrows():
            if not df.iloc[i,6]:
                continue

            label = df.iloc[i,5]
            if isinstance(label, str):
                label = eval(label.strip())

            claims.append(df.iloc[i,0])
            labels.append(label)
            evidences.append(df.iloc[i,1])

        result = []
        for i, (claim, evidence, label) in enumerate(zip(claims, evidences, labels)):
            result.append([i, claim, evidence, label])  # int, string, string, list[int]

        examples = self._get_examples(result, name)
        return examples