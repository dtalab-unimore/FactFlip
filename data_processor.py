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

def remove_accents(text):
    # removing all patterns like \uXXXX (where X is a hex digit)
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

def collate_fn_antonym(examples):
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
    att_mask_sent2 = torch.tensor(att_mask_sent2, dtype=torch.long)
    if label is not None:
        label = torch.tensor(label, dtype=torch.long)
        return sent1, sent2, ids_sent1, segs_sent1, att_mask_sent1, ids_sent2, segs_sent2, att_mask_sent2, label

    return sent1, sent2, ids_sent1, segs_sent1, att_mask_sent1, ids_sent2, segs_sent2, att_mask_sent2


class DataProcessor:

  def __init__(self,config):
    self.config = config

    if self.config["backbone"] is None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
    else:
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["backbone"])

    self.max_sent_len = config["max_sent_len"]
    self.prompt = """You are a fact checking system. You must indicate whether the claim is supported, refuted or "not enough information" based on the given evidence.\nAfter "Answer: ", write exclusively "support", "refute" or "not enough information". Do not write anything else.\n\n{input}\nAnswer:"""
    self.is_generative = ("llama" in self.config["model_name"].lower()
                     or "qwen" in self.config["model_name"].lower()
                     or "gpt"  in self.config["model_name"].lower())

  def __str__(self,):
    pattern = """General data processor: \n\n Tokenizer: {}\n\nMax sentence length: {}""".format(self.config["model_name"], self.max_sent_len)
    return pattern

  def add_word(self, claim, label, word=None, to_class=None):
    if word is None or to_class is None:
      return claim, label
    claim = f"{word}. {claim}"

    to_class = to_class.lower().strip()
    num_classes = len(label)
    if to_class == "support":
      if num_classes == 2:
        label = [1,0]
      else:
        label = [1,0,0]
    elif to_class == "refute":
      if num_classes == 2:
        label = [0,1]
      else:
        label = [0,1,0]
    elif to_class == "nei":
      if num_classes == 2:
        raise ValueError("The model cannot predict the NEI class")
      label = [0,0,1]
    else:
      raise ValueError(f"Unknown target class provided: {to_class}")

    return claim, label

  def _get_examples_causal_lm(self, claim, evidence):
    count_truncated_samples = 0

    text = f"Claim: {claim}\nEvidence: {evidence.strip()}"
    prompt = self.prompt.format(input=text)

    ids_sent1 = self.tokenizer.encode(prompt)
    segs_sent1 = [0] * len(ids_sent1)

    pad_id = self.tokenizer.encode(self.tokenizer.pad_token, add_special_tokens=False)[0]

    if len(ids_sent1) < self.max_sent_len:
      res = self.max_sent_len - len(ids_sent1)
      att_mask_sent1 = [0] * res + [1] * len(ids_sent1) # left padding for causal lm
      ids_sent1 = [pad_id] * res + ids_sent1
      segs_sent1 += [0] * res
    else:
      ids_sent1 = ids_sent1[:self.max_sent_len]
      segs_sent1 = segs_sent1[:self.max_sent_len]
      att_mask_sent1 = [1] * self.max_sent_len
      count_truncated_samples += 1

    return prompt, ids_sent1, segs_sent1, att_mask_sent1, count_truncated_samples

  def _get_examples(self, dataset, dataset_type="train", add_space=False):
    examples = []
    count_truncated_samples = 0

    if self.tokenizer.pad_token is None:
        # safe fallback if the tokenizer does not have a pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    for i,row in enumerate(tqdm(dataset, desc="tokenizing...")):
      id, claim, evidence, label = row
      if add_space:
          claim = ". "+claim

      """
      for the first sentence
      """
      if len(evidence.strip()) == 0:
          evidence = "no evidence"

      if not self.is_generative:
          claim_length = len(self.tokenizer.encode(claim))
          evidence_length = len(self.tokenizer.encode(evidence))

          ids_sent1 = self.tokenizer.encode(claim, evidence)
          segs_sent1 = [0] * claim_length + [1] * evidence_length

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
      else:
        prompt_text, ids_sent1, segs_sent1, att_mask_sent1, truncated_count = self._get_examples_causal_lm(claim, evidence)
        count_truncated_samples += truncated_count

      example = [claim, evidence, ids_sent1, segs_sent1, att_mask_sent1, label]
      examples.append(example)

    print(f"finished preprocessing examples in {dataset_type}: {count_truncated_samples} samples truncated out of {len(dataset)}")

    return examples

class AntonymsProcessor(DataProcessor):

    def __init__(self, config):
        super(AntonymsProcessor, self).__init__(config)

    def _get_examples(self, dataset, dataset_type="train", add_space=False, template=0):
        examples = []
        count_truncated_samples = 0
        if self.tokenizer.pad_token is not None:
            pad_id = self.tokenizer.encode(self.tokenizer.pad_token, add_special_tokens=False)[0]
        else:
            pad_id = self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens=False)[0]

        for row in tqdm(dataset, desc="tokenizing..."):
            if len(row) == 4:
                id, sentence1, sentence2, label = row
            else:
                id, sentence1, sentence2 = row

            if template == 1:
                sentence1 = f"""The statement "{sentence1}" is true"""
                sentence2 = f"""The statement "{sentence2}" is true"""
            elif template == 2:
                sentence1 = f"""The statement "{sentence1}" is not true"""
                sentence2 = f"""The statement "{sentence2}" is not true"""

            if self.is_generative:
                sentence1 = f"Claim: {sentence1}"
                sentence2 = f"Claim: {sentence2}"

            if not self.is_generative: #self.tokenizer.pad_token is not None:
                ids_sent1 = self.tokenizer.encode(sentence1) #(f"The statement '{sentence1}' is not true") #sentence1)
            else:
                prompt = self.prompt.format(input=sentence1) #f"The statement '{sentence1}' is not true") #sentence1)
                ids_sent1 = self.tokenizer.encode(prompt)
            segs_sent1 = [0] * len(ids_sent1)

            if not self.is_generative: #self.tokenizer.pad_token is not None:
                ids_sent2 = self.tokenizer.encode(sentence2) #f"The statement '{sentence2}' is not true") #sentence2)
            else:
                prompt = self.prompt.format(input=sentence2) #f"The statement '{sentence2}' is not true") #sentence2)
                ids_sent2 = self.tokenizer.encode(prompt)
            segs_sent2 = [0] * len(ids_sent2)

            if not self.is_generative:
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
            else:
                if len(ids_sent1) < self.max_sent_len:
                    res = self.max_sent_len - len(ids_sent1)
                    att_mask_sent1 = [0] * res + [1] * len(ids_sent1) # left padding for causal lm
                    ids_sent1 = [pad_id] * res + ids_sent1
                    segs_sent1 += [0] * res
                else:
                    ids_sent1 = ids_sent1[:self.max_sent_len]
                    segs_sent1 = segs_sent1[:self.max_sent_len]
                    att_mask_sent1 = [1] * self.max_sent_len
                    count_truncated_samples += 1

                if len(ids_sent2) < self.max_sent_len:
                    res = self.max_sent_len - len(ids_sent2)
                    att_mask_sent2 = [0] * res + [1] * len(ids_sent2) # left padding for causal lm
                    ids_sent2 = [pad_id] * res + ids_sent2
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

        print(f"finished preprocessing examples in {dataset_type}: {count_truncated_samples} samples truncated out of {len(dataset)}")

        return examples

    def read_input_files(self, file_path, name="train", return_sentences=False, template_type=0, **kwargs):
        df = pd.read_csv(file_path)
        df = df.reset_index(drop=False)
        result = df.values.tolist()

        if return_sentences and not self.config["skip_tokenizer"]:
            raise ValueError("return_sentence and skip_tokenizer are not mutually exclusive")
        if return_sentences:
            return result

        examples = self._get_examples(result, name, template=template_type)
        return examples

class VitamincProcessor(DataProcessor):

    def __init__(self, config):
        super(VitamincProcessor, self).__init__(config)

    def read_input_files(self, file_path, name="train", add_space=False, return_sentences=False, word=None, to_class=None, matches=[]):
        claims, evidences, labels = [], [], []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                claims.append(line["claim"])
                evidences.append(line["evidence"])
                if line["label"] == "SUPPORTS":
                    label = [1,0,0]
                elif line["label"] == "REFUTES":
                    label = [0,1,0]
                elif line["label"] == "NOT ENOUGH INFO":
                    label = [0,0,1]
                else:
                    raise ValueError(f"unknown label {line['label']}")
                labels.append(label)

        result = []
        for i, (claim, evidence, label) in enumerate(zip(claims, evidences, labels)):
                if word is not None:
                    to_class = to_class.lower().strip()
                    if len(matches) > 0:
                        if i not in matches:
                            continue
                        if (to_class == "support" and label[0] == 1) or (to_class == "refute" and label[1] == 1) or (len(label) == 3 and to_class == "nei" and label[2] == 1):
                            continue 
                        claim, label = self.add_word(claim, label, word=word, to_class=to_class)
                result.append([i, claim, evidence, label]) #int, string, string, list[int]

        if return_sentences:
            return result

        examples = self._get_examples(result, name, add_space=add_space)
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

    def read_input_files(self, file_path, name="train", add_space=False, return_sentences=False, word=None, to_class=None, matches=[]):
        claims, evidences, labels = [], [], []

        self.load_data()

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                claims.append(line["claim"])
                if "evidence" not in line.keys() or len(line["evidence"]) == 0:
                    label_txt = "NOT ENOUGH INFORMATION"
                    label = [0,0,1]
                else:
                    first_key = next(iter(line["evidence"]))
                    label_txt = line["evidence"][first_key][0]["label"] #each claim has one unique label (either SUPPORT or CONTRADICT)
                    if label_txt == "SUPPORT":
                        label = [1,0,0]
                    elif label_txt == "CONTRADICT":
                        label = [0,1,0]
                    else:
                        raise ValueError(f"unknown label {label_txt}")

                labels.append(label)

                if label_txt == "NOT ENOUGH INFORMATION":
                    evidence = self.get_random_sentence()
                else:
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
                if word is not None:
                    to_class = to_class.lower().strip()
                    if len(matches) > 0:
                        if i not in matches:
                            continue
                        if (to_class == "support" and label[0] == 1) or (to_class == "refute" and label[1] == 1) or (len(label) == 3 and to_class == "nei" and label[2] == 1):
                            continue
                        claim, label = self.add_word(claim, label, word=word, to_class=to_class)
                result.append([i, claim, evidence, label]) #int, string, string, list[int]

        if return_sentences:
            return result

        examples = self._get_examples(result, name, add_space=add_space)
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

    def read_input_files(self, file_path, name="train", add_space=False, return_sentences=False, word=None, to_class=None, matches=[]):
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
                if word is not None:
                    to_class = to_class.lower().strip()
                    if len(matches) > 0:
                        if i not in matches:
                            continue
                        if (to_class == "support" and label[0] == 1) or (to_class == "refute" and label[1] == 1) or (len(label) == 3 and to_class == "nei" and label[2] == 1):
                            continue 
                        claim, label = self.add_word(claim, label, word=word, to_class=to_class)
                result.append([i, claim, evidence, label]) #int, string, string, list[int]


        if return_sentences:
            return result

        examples = self._get_examples(result, name, add_space=add_space)
        return examples

class FM2Processor(DataProcessor):

    def __init__(self, config):
        super(FM2Processor, self).__init__(config)

    def read_input_files(self, file_path, name="train", add_space=False, return_sentences=False, word=None, to_class=None, matches=[]):
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
                if word is not None:
                    to_class = to_class.lower().strip()
                    if len(matches) > 0:
                        if i not in matches:
                            continue
                        if (to_class == "support" and label[0] == 1) or (to_class == "refute" and label[1] == 1) or (len(label) == 3 and to_class == "nei" and label[2] == 1):
                            continue 
                        claim, label = self.add_word(claim, label, word=word, to_class=to_class)
                result.append([i, claim, evidence, label]) #int, string, string, list[int]

        if return_sentences:
            return result

        examples = self._get_examples(result, name, add_space=add_space)
        return examples

class PolitiHopProcessor(DataProcessor):

    def __init__(self, config):
        super(PolitiHopProcessor, self).__init__(config)

    def read_input_files(self, file_path, name="train", add_space=False, return_sentences=False, word=None, to_class=None, matches=[]):
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
                if word is not None:
                    to_class = to_class.lower().strip()
                    if len(matches) > 0:
                        if i not in matches:
                            continue
                        if (to_class == "support" and label[0] == 1) or (to_class == "refute" and label[1] == 1) or (len(label) == 3 and to_class == "nei" and label[2] == 1):
                            continue 
                        claim, label = self.add_word(claim, label, word=word, to_class=to_class)
                result.append([i, claim, evidence, label]) #int, string, string, list[int]

        if return_sentences:
            return result

        examples = self._get_examples(result, name, add_space=add_space)
        return examples

class HoverProcessor(DataProcessor):

    def __init__(self, config):
        super(HoverProcessor, self).__init__(config)

    def read_input_files(self, file_path, name="train", add_space=False, return_sentences=False, word=None, to_class=None, matches=[]):
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
                if word is not None:
                    to_class = to_class.lower().strip()
                    if len(matches) > 0:
                        if i not in matches:
                            continue
                        if (to_class == "support" and label[0] == 1) or (to_class == "refute" and label[1] == 1) or (len(label) == 3 and to_class == "nei" and label[2] == 1):
                            continue 
                        claim, label = self.add_word(claim, label, word=word, to_class=to_class)
                result.append([i, claim, evidence, label]) #int, string, string, list[int]

        if return_sentences:
            return result

        examples = self._get_examples(result, name, add_space=add_space)
        return examples

class OpenAIProcessor(DataProcessor):

    def __init__(self, config, num_classes):
        super(OpenAIProcessor, self).__init__(config)
        self.highly_perturbing = config["highly_perturbing"]
        self.num_classes = num_classes

    def read_input_files(self, file_path, name="train", add_space=False, return_sentences=False):
        claims, evidences, labels = [], [], []

        df = pd.read_csv(file_path)
        if self.highly_perturbing:
            constraint = "highly_perturbing"
        else:
            constraint = "highly_unperturbing"

        df = df[df["4"].str.contains(constraint)]
        sample_target = df.iloc[0,-2]
        if "support" in sample_target:
            label = [1,0] if self.num_classes == 2 else [1,0,0]
        elif "refute" in sample_target:
            label = [0,1] if self.num_classes == 2 else [0,1,0]
        elif "nei" in sample_target:
            label = [0,0,1]
        else:
            raise ValueError(f"unexpected value {sample_target}")

        for i,sample in df.iterrows():
            claims.append(sample["1"])
            evidences.append(sample["2"])
            labels.append(label)

        result = []
        for i, (claim, evidence, label) in enumerate(zip(claims, evidences, labels)):
            result.append([i, claim, evidence, label]) #int, string, list[string], list[int]

        if return_sentences:
            return result

        examples = self._get_examples(result, name, add_space=add_space)
        return examples
