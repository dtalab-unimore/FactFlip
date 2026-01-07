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
    self.prompt = """You are a fact checking system. You must indicate whether the claim is supported, refuted or "not enough information" based on the given evidence.\nAfter "Answer: ", write exclusively "support", "refute" or "not enough information". Do not write anything else.\n\n{input}\nAnswer:"""
    self.is_generative = ("llama" in self.config["model_name"].lower()
                     or "qwen" in self.config["model_name"].lower()
                     or "gpt"  in self.config["model_name"].lower())

  def __str__(self,):
    pattern = """General data processor: \n\n Tokenizer: {}\n\nMax sentence length: {}""".format(self.config["model_name"], self.max_sent_len)
    return pattern

  def _get_examples_causal_lm_chat(self, claim, evidence):
      count_truncated_samples = 0
      fewshot_text = ""
      for ex in getattr(self, "few_shot_examples", []):
          # ex: (fs_claim, fs_evidence, one_hot_label)
          ans = "support" if ex[2][0] == 1 else "refute" if ex[2][1] == 1 else "not enough information"
          fewshot_text += f"Claim: {ex[0]}\nEvidence: {ex[1]}\nAnswer: {ans}\n\n"

      user_pair = f"Claim: {claim}\nEvidence: {evidence}\nAnswer:"

      use_chat = hasattr(self.tokenizer, "apply_chat_template") and (self.tokenizer.chat_template is not None)
      if use_chat:
          messages = [{"role": "system",
                       "content": "You are a fact-checking assistant. Answer with: support, refute, or not enough information."}]
          for ex in getattr(self, "few_shot_examples", []):
              continue
              ans = "support" if ex[2][0] == 1 else "refute" if ex[2][1] == 1 else "not enough information"
              messages.append({"role": "user",
                               "content": f"Claim: {ex[0]}\nEvidence: {ex[1]}\nAnswer:"})
              messages.append({"role": "assistant", "content": ans})
          messages.append({"role": "user", "content": user_pair})

          prompt_text = self.tokenizer.apply_chat_template(
              messages,
              tokenize=False,
              add_generation_prompt=True  # leaves assistant turn empty to generate
          )
      else:
          prompt_text = (
                  "You are a fact-checking assistant. Answer with exactly one of: support, refute, not enough information.\n\n"
                  + fewshot_text
                  + user_pair
                  + "\nAnswer:"
          )

      enc = self.tokenizer(
          prompt_text,
          add_special_tokens=True,
          truncation=True,
          max_length=self.max_sent_len,
          return_attention_mask=True
      )

      ids_sent1 = enc["input_ids"]
      att_mask_sent1 = enc["attention_mask"]
      segs_sent1 = [0] * len(ids_sent1)  # dummy to keep shape compatibility with RoBERTa path
      pad_id = self.tokenizer.pad_token_id

      if len(ids_sent1) < self.max_sent_len:
          res = self.max_sent_len - len(ids_sent1)
          ids_sent1 += [pad_id] * res
          segs_sent1 += [0] * res
          att_mask_sent1 += [0] * res
      else:
          # already truncated by tokenizer, keeping counter for compatibility with roberta
          count_truncated_samples += int(len(enc["input_ids"]) == self.max_sent_len)

      # I return the prompt_text for debugging reasons
      return prompt_text, ids_sent1, segs_sent1, att_mask_sent1, count_truncated_samples

  def _get_examples_causal_lm(self, claim, evidence):
    count_truncated_samples = 0

    text = ""
    """for sample in self.few_shot_examples:
        label = "support" if sample[2][0] == 1 else "refute" if sample[2][1] == 1 else "not enough information"
        tmp = f"Claim: {sample[0]}\nEvidence: {sample[1].strip()}\nAnswer: {label}\n\n"
        text += tmp"""
    text += f"Claim: {claim}\nEvidence: {evidence.strip()}"
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

  def _get_examples(self, dataset, dataset_type="train", add_space=False, add_trigger_tokens=False):
    examples = []
    count_truncated_samples = 0

    if self.tokenizer.pad_token is None:
        # safe fallback for llama
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    for i,row in enumerate(tqdm(dataset, desc="tokenizing...")):
      id, claim, evidence, label = row
      if add_space:
          claim = ". "+claim

      """
      for the first sentence
      """
      if len(evidence.strip()) == 0: # fever dataset contains blank pieces of evidence
          evidence = "no evidence"

      if not self.is_generative:
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
      else:
        prompt_text, ids_sent1, segs_sent1, att_mask_sent1, truncated_count = self._get_examples_causal_lm(claim, evidence)
        #if i == 0:
        #    print(prompt_text)
        count_truncated_samples += truncated_count

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
        if self.tokenizer.pad_token is not None:
            pad_id = self.tokenizer.encode(self.tokenizer.pad_token, add_special_tokens=False)[0]
        else:
            pad_id = self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens=False)[0]

        for row in tqdm(dataset, desc="tokenizing..."):
            if len(row) == 4:
                id, sentence1, sentence2, label = row
            else:
                id, sentence1, sentence2 = row

            if self.tokenizer.pad_token is not None:
                ids_sent1 = self.tokenizer.encode(sentence1)
            else:
                prompt = self.prompt.format(input=sentence1)
                ids_sent1 = self.tokenizer.encode(prompt)
            segs_sent1 = [0] * len(ids_sent1)

            if self.tokenizer.pad_token is not None:
                ids_sent2 = self.tokenizer.encode(sentence2)
            else:
                prompt = self.prompt.format(input=sentence2)
                ids_sent2 = self.tokenizer.encode(prompt)
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
            lines = f.readlines()

        sample = json.loads(random.choice(lines))["lines"]
        random_line = self.get_sentence_by_number(0, sample)

        return random_line

    def get_sentence_by_number(self, num, text):
        """
        Find text that starts with num<tab> and ends at the next ' .'
        """
        # Regex explanation:
        # \b{num}\t  → match the number at a word boundary followed by a tab
        # (.*?)       → lazily capture everything
        # (?= \.)     → stop right before ' .'
        pattern = rf"\b{num}\t(.*?)(?= \.)"
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            #return text.strip("\n")[int(num)].strip()
            return ""

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
                for line in tqdm(f):
                    line = json.loads(line)

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
                                sentence = self.data[evidence_piece[2]]
                                sentence = self.get_sentence_by_number(evidence_piece[3], sentence)
                                if i > 0:
                                    text+="\n"
                                text+=sentence
                            evidence.append(text)

                    found = False
                    evs = []
                    for ev in evidence:
                        if len(ev) != 0:
                            found = True
                            evs.append(ev) #for simplicity, I only consider the first evidence set
                            #break

                    if not found:
                        #print(f"no evidence found for this sample. Label: {label}, Evidence: {evs}")
                        continue

                    claims.append(line["claim"])
                    evidences.append("\n".join(evs))
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

    def read_input_files(self, file_path, name="train", return_sentences=False, add_space=False):
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

        examples = self._get_examples(result, name, add_space=add_space)
        return examples


class VitamincProcessor(DataProcessor):

    def __init__(self, config):
        super(VitamincProcessor, self).__init__(config)
        self.few_shot_examples = []

    def read_input_files(self, file_path, name="train", add_space=False, return_sentences=False, add_trigger_tokens=False):
        claims, evidences, labels = [], [], []
        few_shot_support_counter, few_shot_refute_counter, few_shot_nei_counter = 0,0,0

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                claims.append(line["claim"])
                evidences.append(line["evidence"])
                if line["label"] == "SUPPORTS":
                    label = [1,0,0]
                    if name == "train" and few_shot_support_counter < 2:
                        few_shot_support_counter += 1
                        self.few_shot_examples.append([line["claim"], line["evidence"], label])
                elif line["label"] == "REFUTES":
                    label = [0,1,0]
                    if name == "train" and few_shot_refute_counter < 2:
                        few_shot_refute_counter += 1
                        self.few_shot_examples.append([line["claim"], line["evidence"], label])
                elif line["label"] == "NOT ENOUGH INFO":
                    label = [0,0,1]
                    if name == "train" and few_shot_nei_counter < 2:
                        few_shot_nei_counter += 1
                        self.few_shot_examples.append([line["claim"], line["evidence"], label])
                else:
                    raise ValueError(f"unknown label {line['label']}")
                labels.append(label)

        result = []
        for i, (claim, evidence, label) in enumerate(zip(claims, evidences, labels)):
            result.append([i, claim, evidence, label]) #int, string, string, list[int]

        if return_sentences:
            return result

        examples = self._get_examples(result, name, add_space=add_space, add_trigger_tokens=add_trigger_tokens)
        return examples


class SciFactProcessor(DataProcessor):

    def __init__(self, config):
        super(SciFactProcessor, self).__init__(config)
        self.data_path = "data/scifact/corpus.jsonl"
        self.data = {}
        self.few_shot_examples = []

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

    def read_input_files(self, file_path, name="train", add_space=False, return_sentences=False, add_trigger_tokens=False):
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

                if name == "train" and count_support < 2 and label_txt == "SUPPORT":
                    count_support += 1
                    self.few_shot_examples.append([line["claim"], evidence, label])
                elif name == "train" and count_refute < 2 and label_txt == "CONTRADICT":
                    count_refute += 1
                    self.few_shot_examples.append([line["claim"], evidence, label])
                elif name == "train" and count_not_enough < 2 and label_txt == "NOT ENOUGH INFORMATION":
                    count_not_enough += 1
                    self.few_shot_examples.append([line["claim"], evidence, label])

        result = []
        for i, (claim, evidence, label) in enumerate(zip(claims, evidences, labels)):
            result.append([i, claim, evidence, label]) #int, string, string, list[int]

        if return_sentences:
            return result

        examples = self._get_examples(result, name, add_space=add_space, add_trigger_tokens=add_trigger_tokens)
        return examples


class AVTCProcessor(DataProcessor):

    def __init__(self, config):
        super(AVTCProcessor, self).__init__(config)
        self.few_shot_examples = []

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

    def read_input_files(self, file_path, name="train", add_space=False, return_sentences=False, add_trigger_tokens=False):
        claims, evidences, labels = [], [], []
        few_shot_support_counter, few_shot_refute_counter, few_shot_nei_counter = 0,0,0

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
            if name == "train" and few_shot_support_counter < 2 and sample["label"] == "Supported":
                few_shot_support_counter += 1
                self.few_shot_examples.append([sample["claim"], evidence, label])
            elif name == "train" and few_shot_refute_counter < 2 and sample["label"] == "Refuted":
                few_shot_refute_counter += 1
                self.few_shot_examples.append([sample["claim"], evidence, label])
            elif name == "train" and few_shot_nei_counter < 2 and sample["label"] == "Not Enough Evidence":
                few_shot_nei_counter += 1
                self.few_shot_examples.append([sample["claim"], evidence, label])

        result = []
        for i, (claim, evidence, label) in enumerate(zip(claims, evidences, labels)):
            result.append([i, claim, evidence, label]) #int, string, list[string], list[int]

        if return_sentences:
            return result

        examples = self._get_examples(result, name, add_space=add_space, add_trigger_tokens=add_trigger_tokens)
        return examples

class FM2Processor(DataProcessor):

    def __init__(self, config):
        super(FM2Processor, self).__init__(config)
        self.few_shot_examples = []

    def read_input_files(self, file_path, name="train", add_space=False, return_sentences=False, add_trigger_tokens=False):
        claims, evidences, labels = [], [], []
        few_shot_support_counter, few_shot_refute_counter = 0,0

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

                if name == "train" and few_shot_support_counter < 3 and line["label"] == "SUPPORTS":
                    few_shot_support_counter += 1
                    self.few_shot_examples.append([line["text"], evidence_text, label])
                elif name == "train" and few_shot_refute_counter < 3 and line["label"] == "REFUTES":
                    few_shot_refute_counter += 1
                    self.few_shot_examples.append([line["text"], evidence_text, label])

            result = []
            for i, (claim, evidence, label) in enumerate(zip(claims, evidences, labels)):
                result.append([i, claim, evidence, label]) #int, string, string, list[int]

        if return_sentences:
            return result

        examples = self._get_examples(result, name, add_space=add_space, add_trigger_tokens=add_trigger_tokens)
        return examples

class PolitiHopProcessor(DataProcessor):

    def __init__(self, config):
        super(PolitiHopProcessor, self).__init__(config)
        self.few_shot_examples = []

    def read_input_files(self, file_path, name="train", add_space=False, return_sentences=False, add_trigger_tokens=False):
        claims, evidences, labels = [], [], []
        few_shot_support_counter, few_shot_refute_counter = 0,0
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

            if name == "train" and few_shot_support_counter < 3 and line["annotated_label"] == "true":
                few_shot_support_counter += 1
                self.few_shot_examples.append([line["statement"], evidence_text, label])
            elif name == "train" and few_shot_refute_counter < 3 and line["annotated_label"] == "false":
                few_shot_refute_counter += 1
                self.few_shot_examples.append([line["statement"], evidence_text, label])

        result = []
        for i, (claim, evidence, label) in enumerate(zip(claims, evidences, labels)):
            result.append([i, claim, evidence, label]) #int, string, string, list[int]

        if return_sentences:
            return result

        examples = self._get_examples(result, name, add_space=add_space, add_trigger_tokens=add_trigger_tokens)
        return examples

class HoverProcessor(DataProcessor):

    def __init__(self, config):
        super(HoverProcessor, self).__init__(config)
        self.few_shot_examples = []

    def read_input_files(self, file_path, name="train", add_space=False, return_sentences=False, add_trigger_tokens=False):
        claims, evidences, labels = [], [], []
        few_shot_support_counter, few_shot_refute_counter = 0,0
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

            if name == "train" and few_shot_support_counter < 3 and line["label"] == 0:
                few_shot_support_counter += 1
                self.few_shot_examples.append([line["claim"], line["evidence"], label])
            elif name == "train" and few_shot_refute_counter < 3 and line["label"] == 1:
                few_shot_refute_counter += 1
                self.few_shot_examples.append([line["claim"], line["evidence"], label])

        result = []
        for i, (claim, evidence, label) in enumerate(zip(claims, evidences, labels)):
            result.append([i, claim, evidence, label])  # int, string, string, list[int]

        if return_sentences:
            return result

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
