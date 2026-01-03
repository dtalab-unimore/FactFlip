import os
from pprint import pprint

import torch
import datasets
import pickle
import pandas as pd

import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import AdamW

from data_processor import AVTCProcessor, FeverProcessor, FeverSymmetricProcessor, SciFactProcessor, VitamincProcessor, \
  dataset, collate_fn, collate_fn_antonym, FM2Processor, PolitiHopProcessor, HoverProcessor, AntonymsProcessor, OpenAIProcessor
from model import RobertaModel, GenerativeModel
from trainer import Trainer, AntonymTrainer
from utils import set_random_seeds, get_config, get_device, class_weights

def get_data(config):
  if config["dataset"] == "avtc":
      processor = AVTCProcessor(config)
      num_classes = 3

      path_train = "./data/avtc/train.json"
      path_dev = "./data/avtc/dev.json"
      path_test = "./data/avtc/test.json"

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

  elif config["dataset"] == "antonym":
    processor = AntonymsProcessor(config)
    if "llama" in config["model_name"].lower() or "qwen" in config["model_name"].lower():
      num_classes = 3
    else:
      model_name = config["model_name"].split("/")[-1].split("_")[0]
      num_classes = len(class_weights[model_name])

    if not config["test_only"]:
      raise ValueError("cannot train on antonyms")

    if config["stereotype"]:
      path_train = "./data/antonym/stereotype_words.csv"
      path_dev = "./data/antonym/stereotype_words.csv"
      path_test = "./data/antonym/stereotype_words.csv"
    else:
      path_train = "./data/antonym/antonym_pairs.csv"
      path_dev = "./data/antonym/antonym_pairs.csv"
      path_test = "./data/antonym/antonym_pairs.csv"
  elif config["dataset"] == "from_openai_generated":
    if "avtc" in config["openai_path"] or "scifact" in config["openai_path"] or "vitaminc" in config["openai_path"]:
      num_classes = 3
    else:
      num_classes = 2

    processor = OpenAIProcessor(config, num_classes)

    if not config["test_only"]:
      raise ValueError("cannot train on openai generated data")

    path_train = path_dev = path_test = config["openai_path"]
  else:
    raise ValueError(f"{config['dataset']} is not a valid database name (choose between 'avtc', 'scifact', 'hover', 'fm2', 'politihop', 'vitaminc', 'antonym', 'from_openai_generated')")

  return processor, num_classes, path_train, path_dev, path_test



def run():
  config = get_config()
  device = get_device()
  set_random_seeds(config["seed"])

  processor, num_classes, path_train, path_dev, path_test = get_data(config)

  global collate_fn
  if config["dataset"] == "antonym":
    collate_fn = collate_fn_antonym

  config["num_classes"] = num_classes

  if config["test_only"]:
    if config["dataset"] == "scifact":
      data_test = processor.read_input_files(path_dev, name="dev")
    elif config["dataset"] == "antonym":
      data_test = processor.read_input_files(path_test, name="test")
      data_test_nei1 = processor.read_input_files(path_test, name="test", template_type=1)
      data_test_nei2 = processor.read_input_files(path_test, name="test", template_type=2)
    else:
        data_test = processor.read_input_files(path_test, name="test")

    if config["dataset"] != "antonym" and "qwen" in config["model_name"].lower():
        if config["dataset"] == "hover":
            data_test = data_test[:250] + data_test[1000:1250] #hover's first 500 test samples have all the same label
        else:
            data_test = data_test[:500]

    test_set = dataset(data_test)
    test_dataloader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
    if config["dataset"] == "antonym":
        test_set_nei1 = dataset(data_test_nei1)
        test_set_nei2 = dataset(data_test_nei2)

        test_dataloader_nei1 = DataLoader(test_set_nei1, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
        test_dataloader_nei2 = DataLoader(test_set_nei2, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
  else:
    data_train = processor.read_input_files(path_train, name="train")
    data_dev = processor.read_input_files(path_dev, name="dev")
    data_test = processor.read_input_files(path_test, name="test")
    if config["dataset"] == "scifact":
      # scifact test set is blind, so we use 20% of train as dev, and the dev as test
      tmp = data_dev
      data_dev = data_train[int(len(data_train)*0.8):]
      data_train = data_train[:int(len(data_train)*0.8)]
      data_test = tmp

    train_set = dataset(data_train)
    dev_set = dataset(data_dev)
    test_set = dataset(data_test)

    train_dataloader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_set, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

  if "qwen" in config["model_name"].lower():
    model = GenerativeModel(config)
  else:
    model = RobertaModel(config)

  model.to(device)

  if config["model_name"].split(".")[-1] == "pt" and config["backbone"] is not None:
    # Load the finetuned model
    state_dict = torch.load(config["model_name"], map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)

  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
    {
      "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
      "weight_decay": float(config["weight_decay"]),
    },
    {
      "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
      "weight_decay": 0.0
    },
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=config["lr"], weight_decay=config["weight_decay"])

  loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(config["class_weight"])).to(device)

  best_dev_f1 = -1
  result_metrics = []

  if config["dataset"] == "antonym":
    trainer = AntonymTrainer(config, device)
  else:
    trainer = Trainer(config, device)

  best_test_acc, best_test_pre, best_test_rec, best_test_f1 = None, None, None, None

  if not config["test_only"]:
    for epoch in range(config["epochs"]):
      print('===== Start training: epoch {} ====='.format(epoch + 1))
      trainer.train(epoch, model, loss_fn, optimizer, train_dataloader)
      dev_a, dev_p, dev_r, dev_f1 = trainer.val(model, dev_dataloader)
      test_a, test_p, test_r, test_f1 = trainer.val(model, test_dataloader)
      if dev_f1 > best_dev_f1:
        best_dev_f1 = dev_f1
        best_test_acc, best_test_pre, best_test_rec, best_test_f1 = test_a, test_p, test_r, test_f1
        os.makedirs(f"./models/{config['model_name'].split('/')[-1]}/seed_{config['seed']}/{config['dataset']}/", exist_ok=True)
        torch.save(model.state_dict(), f"./models/{config['model_name'].split('/')[-1]}/seed_{config['seed']}/{config['dataset']}/{config['dataset']}_model.pt")

    print('*** best result on test set ***')
    print(best_test_acc)
    print(best_test_pre)
    print(best_test_rec)
    print(best_test_f1, end="\n")
    result_metrics.append([best_test_acc, best_test_pre, best_test_rec, best_test_f1])

    print(f"Overall metrics: {result_metrics}")
  else:
    path = "/".join(config["model_name"].split("/")[:-1])+f"/{config['dataset']}_predictions.pkl"
    if config["dataset"] == "antonym":
      concept_vectors = trainer.val_potency(model, test_dataloader, num_labels=num_classes)
      concept_vectors_nei1 = trainer.val_potency(model, test_dataloader_nei1, num_labels=num_classes)
      concept_vectors_nei2 = trainer.val_potency(model, test_dataloader_nei2, num_labels=num_classes)

      if config["stereotype"]:
        path = f"data/antonym/{config['model_name'].split('/')[-1].split('_')[0]}_potency_stereotype/"
      else:
        path = f"data/antonym/{config['model_name'].split('/')[-1].split('_')[0]}_potency/"

      list_concept_vectors = []
      for k,v in concept_vectors.items():
        tmp = []
        tmp.append(k)
        for k1,v1 in v.items():
          tmp.append(v1)
        list_concept_vectors.append(tmp)

      list_concept_vectors_nei1 = []
      for k,v in concept_vectors_nei1.items():
        tmp = []
        tmp.append(k)
        for k1,v1 in v.items():
          tmp.append(v1)
        list_concept_vectors_nei1.append(tmp)

      list_concept_vectors_nei2 = []
      for k,v in concept_vectors_nei2.items():
        tmp = []
        tmp.append(k)
        for k1,v1 in v.items():
          tmp.append(v1)
        list_concept_vectors_nei2.append(tmp)

      if num_classes == 3:
        columns = ["Word pair", "Support", "Refute", "Nei"]
      else:
        columns = ["Word pair", "Support", "Refute"]

      df = pd.DataFrame(list_concept_vectors, columns=columns)
      df_nei1 = pd.DataFrame(list_concept_vectors_nei1, columns=columns)
      df_nei2 = pd.DataFrame(list_concept_vectors_nei2, columns=columns)
      if num_classes == 3:
          df["Nei"] = df_nei1["Nei"] + df_nei2["Nei"]
      os.makedirs(path, exist_ok=True)
      if config["extract_words_from_dev"] is None:
          df.to_csv(os.path.join(path, "concept_vectors.csv"), index=False)
      else:
          from data_processor import collate_fn
          config["dataset"] = config["extract_words_from_dev"]
          processor, num_classes, path_train, path_dev, path_test = get_data(config)
          if config["dataset"] == "scifact":
              data_dev = processor.read_input_files(path_train, name="train")
              data_dev = data_dev[int(len(data_dev)*0.8):]
          else:
              data_dev = processor.read_input_files(path_dev, name="dev")

          if "qwen" in config["model_name"].lower():
              if config["dataset"] == "hover":
                  data_dev = data_dev[:250] + data_dev[1000:1250] #hover's first 500 test samples have all the same label
              else:
                  data_dev = data_dev[:500]

          dev_set = dataset(data_dev)
          dev_dataloader = DataLoader(dev_set, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
          trainer = Trainer(config, device)

          dev_a, dev_p, dev_r, dev_f1, predictions, labels = trainer.val(model, dev_dataloader, return_preds=True)
          match_idxs = []
          if config["dataset"] == "hover" and "qwen" in config["model_name"].lower():
              for i, (pred, lab) in enumerate(zip(predictions[:250], labels[:250])):
                  if np.argmax(pred) == np.argmax(lab):
                      match_idxs.append(i)

              for i, (pred, lab) in enumerate(zip(predictions[250:], labels[250:])):
                  if np.argmax(pred) == np.argmax(lab):
                      match_idxs.append(1000+i)
          else:
              for i, (pred, lab) in enumerate(zip(predictions, labels)):
                  if np.argmax(pred) == np.argmax(lab):
                      match_idxs.append(i)

          performances = {"Support": {}, "Refute": {}, "Nei": {}}
          for i, to_class in enumerate(["Support", "Refute", "Nei"]):
              if i == num_classes:
                  break # we do not apply NEI when the class doesn't have it

              df = df.sort_values(to_class, ascending=False)
              words = df["Word pair"].iloc[:50].values.tolist()
              for word in words:
                  processor, num_classes, path_train, path_dev, path_test = get_data(config)

                  if config["dataset"] == "scifact":
                      data_dev = processor.read_input_files(path_train, name="train", word=word, to_class=to_class, matches=match_idxs)
                      data_dev = data_dev[int(len(data_dev)*0.8):]
                  else:
                      data_dev = processor.read_input_files(path_dev, name="dev", word=word, to_class=to_class, matches=match_idxs)

                  if "llama" in config["model_name"].lower() or "qwen" in config["model_name"].lower():
                      num_data = 10
                  else:
                      num_data = 320

                  if config["dataset"] == "hover":
                      data_dev = data_dev[:num_data//2] + data_dev[-num_data//2:] #hover's first 500 test samples have all the same label
                  else:
                      data_dev = data_dev[:num_data]

                  dev_set = dataset(data_dev)
                  dev_dataloader = DataLoader(dev_set, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

                  dev_a, dev_p, dev_r, dev_f1 = trainer.val(model, dev_dataloader)
                  performances[to_class][word] = dev_a
          for label, word_dict in performances.items():
              sorted_items = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
              top_5_items = sorted_items[:5]
              performances[label] = dict(top_5_items)
          print(f"*** FINAL WORDS based on DEV PERFORMANCE with {num_data} SAMPLES")
          pprint(performances)
    elif config["stereotype"]:
      test_a, test_p, test_r, test_f1, predictions, labels = trainer.val(model, test_dataloader, return_preds=True)
      num_samples_per_word = len(predictions) / 250
      if num_samples_per_word // 1 != num_samples_per_word: # not int
          raise ValueError(f"num_samples_per_word is not int: {num_samples_per_word}")

      grouped_acc = []
      count = 0
      for i in range(len(predictions)):
          if np.argmax(predictions[i]) == np.argmax(labels[i]):
              count += 1
          if (i+1) % num_samples_per_word == 0:
              grouped_acc.append(count / num_samples_per_word)
              count = 0

      print(grouped_acc)
    else:
      test_a, test_p, test_r, test_f1, predictions, labels = trainer.val(model, test_dataloader, return_preds=True)

      count = 0
      assert len(data_test) == len(predictions) == len(labels)
      for test_sample, pred, lab in zip(data_test, predictions, labels):
          if pred == lab:
              print(test_sample)
              count += 1
      print(count / len(data_test))
      os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
      with open(path, "wb") as f:
        pickle.dump(predictions, f)
      print(predictions)
      print("Accuracy: {}\nPrecision: {}\nRecall: {}\n F1: {}".format(test_a, test_p, test_r, test_f1))


if __name__ == "__main__":
  run()
