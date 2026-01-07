import torch
import random

import argparse
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from functools import lru_cache

# support, refute, not enough info
class_weights = {
    "avtc": [0.7, 0.39, 0.91],
    "scifact": [0.59, 0.79, 0.62],
    "vitaminc": [0.5, 0.64, 0.86],
    "fm2": [0.51, 0.49],
    "politihop": [0.83, 0.17],
    "hover": [0.39, 0.61],
    "antonym": [1,1,1],
    "from_openai_generated": [1,1,1]
}

def get_config():
    parser = argparse.ArgumentParser(description="Argument parser for model configuration")

    parser.add_argument('--model_name', type=str, default='roberta-base', help='Model name')
    parser.add_argument('--backbone', type=str, default=None, help='Model name')
    parser.add_argument('--embed_size', type=int, default=768, help='Embedding size')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--max_sent_len', type=int, default=512, help='Maximum sentence length')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--openai_path', type=str, required=False, help='path to openai generated claims')
    parser.add_argument('--highly_perturbing', action="store_true", help='select highly perturbing claims for openai')
    parser.add_argument('--test_only', action="store_true", help='Do not train')
    parser.add_argument('--potency', action="store_true", help='Extract "potency" words to flip model predictions')
    parser.add_argument('--stereotype', action="store_true", help='Extract "stereotype" words to flip model predictions')
    parser.add_argument('--extract_words_from_dev', type=str, required=False, default=None, help="Dev set for word extraction")

    args = vars(parser.parse_args())

    if "class_weight" not in args.keys():
        args["class_weight"] = class_weights[args["dataset"]]

    return args

def get_config_adversarial():
    parser = argparse.ArgumentParser(description="Argument parser for model configuration")

    parser.add_argument('--model_name', type=str, default='roberta-base', help='Model name')
    parser.add_argument('--backbone', type=str, default=None, help='Model name')
    parser.add_argument('--embed_size', type=int, default=768, help='Embedding size')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--k', type=int, default=5, help='Number of concepts to analyze')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to analyze')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_sent_len', type=int, default=512, help='Maximum sentence length')
    parser.add_argument('--dataset', type=str, default='avtc', help='Dataset to test the concept vectors')
    parser.add_argument('--test_only', action="store_true", help='Do not train')
    parser.add_argument('--no_compute_predictions', action="store_false", help='Don\'t compute model predictions on original claims')
    parser.add_argument('--use_similarity', action="store_true", help='Extract words based on similarity to claims')
    parser.add_argument('--use_dev_tuning', action="store_true", help='Extract words by first performing dev set tuning')
    parser.add_argument('--num_words', type=int, default=1, help='Number of adversarial words to add')
    parser.add_argument('--stereotype', action="store_true", help='Extract "stereotype" words to flip model predictions')
    parser.add_argument('--not_from_template', action="store_false", help='Create new claims with openai')
    parser.add_argument('--list_of_words', type=list, default=[], help='List of adversarial words to use to generate the claims')

    args = vars(parser.parse_args())

    if "class_weight" not in args.keys():
        args["class_weight"] = class_weights[args["dataset"]]

    if len(class_weights[args["dataset"]]) == 2:
        class_weights[args["dataset"]] = [0.5, 0.5]
    else:
        class_weights[args["dataset"]] = [0.33, 0.33, 0.33]

    return args

@lru_cache()
def get_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")

    return device

def set_random_seeds(seed):
    """
    set random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def output_metrics(labels, preds):
    """

    :param labels: ground truth labels
    :param preds: prediction labels
    :return: accuracy, precision, recall, f1
    """
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    f1 = f1_score(labels, preds, average="macro")

    print("{:15}{:<.6f}".format('accuracy:', accuracy))
    print("{:15}{:<.6f}".format('precision:', precision))
    print("{:15}{:<.6f}".format('recall:', recall))
    print("{:15}{:<.6f}".format('f1:', f1))

    return accuracy, precision, recall, f1
