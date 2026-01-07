import os

from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import random
import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from torch.utils.data import DataLoader
from torch.optim import AdamW

from collections import defaultdict
from data_processor import AVTCProcessor, SciFactProcessor, VitamincProcessor, \
    FM2Processor, PolitiHopProcessor, HoverProcessor, dataset, collate_fn, collate_fn_antonym, DataProcessor, \
    AntonymsProcessor
from utils import set_random_seeds, get_config_adversarial, get_device
from model import RobertaModel, GenerativeModel
from openai_model import remove_markdown_syntax, extract_result, OpenAIModel

from copy import deepcopy

load_dotenv(override=True)

from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
import warnings
warnings.filterwarnings("ignore")

tracer_provider = register(
    project_name="factcheckingbias",
    endpoint="http://localhost:6006/v1/traces",
)

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

class AdversarialModel:
    def __init__(self, config):
        self.openai_model = OpenAIModel(model_name="gpt-4o-mini", temperature=0.7, top_p=0.8)
        self.openai_model_check = OpenAIModel(model_name="gpt-4o-mini", temperature=0.01, top_p=0.8)

        self.config = config
        self.fc_model_name = config['model_name']

        device = get_device()
        self.is_generative = ("llama" in self.fc_model_name.lower()
                     or "qwen" in self.fc_model_name.lower()
                     or "gpt"  in self.fc_model_name.lower())

        if not config["no_compute_predictions"]:
            if self.is_generative:
                self.fc_model = GenerativeModel(config)
            else:
                self.fc_model = RobertaModel(config)
                state_dict = torch.load(self.fc_model_name, map_location=device)
                self.fc_model.load_state_dict(state_dict)
                self.fc_model.to(device)

        self.use_similarity = config["use_similarity"]
        if self.use_similarity:
            self.similarity_model = AutoModel.from_pretrained("roberta-base")
            self.similarity_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            self.similarity_model.eval()
            self.similarity_model.to(device)

        self.processor = DataProcessor(config)

        self.claim_generation_prompt = """You will be given a claim and a (series of) word(s).
        You must add all the provided words inside the given claim. The words added must be exactly like the one provided: any sort of stemming, lemmatization or similar is not allowed.
        Also, the rest of the claim must be exactly like the original: no existing word must be removed or modified, and only the provided words must be added.
        The word(s) must be added inside the claim, not at the beginning. In case of multiple words, you can add them in different positions: you are not forced to add them consecutively.
        The novel claim must exclusively satisfy the following rule:
        "{label}"
        
        First reason step-by-step. Then write "Final answer: " followed exclusively by the generated claim. Do not write anything else after "Final answer: ".
        Ensure that the final claim is exactly like the original, except for the added word and other stopwords.
        
        Claim: {claim}
        Word: {word}
        
        Let's think step-by-step. """

        self.check_claim_equivalence = """You will be given an original claim and a novel claim derived from the original.
        You must determine if the following rule is satisfied or not:
        "{label}"
        
        First think step-by-step, then write "Final answer:" followed exclusively by "yes" if the rule is satisfied, "no" otherwise. Do not write anything else after "Final answer:".
        
        Original claim: {claim1}
        Novel claim: {claim2}
        
        Let's think step-by-step."""

    def get_testing_concepts(self, concept_vectors, k=5):
        if k == -1:
            # used when evaluating stereotype words
            # this output format is to be consistent with the general pipeline for evaluating adversarial words
            return {
                "highly_perturbing": concept_vectors.values.tolist(),
                "highly_unperturbing": [], #concept_vectors.values.tolist()
            }
        concept_vectors_list = concept_vectors.values.tolist()
        highly_perturbing = concept_vectors_list[:k]
        highly_unperturbing = concept_vectors_list[-k:]

        return {
            "highly_perturbing": highly_perturbing,
            "highly_unperturbing": highly_unperturbing
        }

    def generate_claim_with_openai(self, orig_claim, word, label_txt, label):
        claim1, _ = self.openai_model.query(self.claim_generation_prompt, attr={"claim": orig_claim, "word": word, "label": label_txt})
        if "final answer:" not in claim1.lower():
            return -1
        claim1 = extract_result(remove_markdown_syntax(claim1), "Final answer:").strip()

        for w in word.split(","):
            w = w.strip().lower()
            if w not in claim1.strip().lower():
                return -1

        check2, _ = self.openai_model_check.query(self.check_claim_equivalence, attr={"claim1": orig_claim, "claim2": claim1, "label": label_txt})
        check2 = extract_result(remove_markdown_syntax(check2), "Final answer:").strip().lower()
        if check2 != "yes":
            return -1

        return claim1

    def extract_embeddings(self, concepts):
        concepts_tok = self.similarity_tokenizer(concepts, return_tensors="pt", padding="longest").to(get_device())
        with torch.no_grad():
            embeddings = self.similarity_model(**concepts_tok, output_hidden_states=True).hidden_states[-1]

        return embeddings[:,0,:]

    def get_most_similar_words_positions(self, concepts, claims, k=3):
        S = cosine_similarity(claims.cpu().numpy(), concepts.cpu().numpy())
        topk_indices = np.argsort(S, axis=1)[:, -k:][:, ::-1]
        return topk_indices

    def attack_with_similarity(self, test, sampled_concepts, to="support"):
        test_samples = test
        perturbing_samples = [[] for _ in range(len(sampled_concepts))]
        for i, perturbation_type in enumerate(tqdm(sampled_concepts.keys(), desc="Iterating through different perturbation types...")):
            concepts = []
            for j, concept in enumerate(sampled_concepts[perturbation_type]):
                word = concept[0]
                concepts.append(word)

            embeddings = self.extract_embeddings(concepts)
            lst = [[i, el[0], el[1], el[-1]] for i, el in enumerate(test_samples)]
            samples = self.processor._get_examples(lst)
            dl = DataLoader(dataset(samples), batch_size=self.config["batch_size"], shuffle=False, collate_fn=collate_fn)
            chosen_words = []

            for batch in dl:
                claims = batch[0]
                sent_embs = self.extract_embeddings(claims)
                indices = self.get_most_similar_words_positions(embeddings, sent_embs)
                for index_lst in indices:
                    tmp = []
                    for index in index_lst:
                        tmp.append(concepts[index])
                    chosen_words.append(tmp)

            for sample, words in tqdm(zip(test_samples, chosen_words), desc="Iterating through different test samples..."):
                for word in words:
                    if sample[-1][0] == 1:
                        label_txt = "the factual content of the original claim must entail the factual content of the new claim. Inside the claims, subjective statements are not factual content, and thus they must not be considered in the final evaluation"
                    elif sample[-1][1] == 1:
                        label_txt = "the factual content of the new claim must entail the factual content of the original claim. Inside the claims, subjective statements are not factual content, and thus they must not be considered in the final evaluation"
                    else:
                        label_txt = "the factual content of the original claim must entail the factual content of the new claim and viceversa. Inside the claims, subjective statements are not factual content, and thus they must not be considered in the final evaluation"

                    if int(self.config["num_words"]) > 1:
                        raise NotImplementedError()

                    claim1 = self.generate_claim_with_openai(sample[0], word, label_txt, sample[-1])
                    if claim1 == -1:
                        continue

                    perturbing_samples[i].append([len(perturbing_samples[i]) - 1, claim1, sample[1], sample[-1], f"{perturbation_type}_{to}"])

        if config["stereotype"]:
            path = f"data/antonym/{self.config['dataset']}_openai_generated_similar_stereotype/"
        else:
            path = f"data/antonym/{self.config['dataset']}_openai_generated_similar/"

        model_name = "qwen_similar" if self.is_generative else "roberta_similar"
        os.makedirs(path, exist_ok=True)
        pd.DataFrame([row + [i] for i, sublist in enumerate(perturbing_samples) for row in sublist]).to_csv(
            os.path.join(path, f"generated_test_set_{to}_{model_name}.csv"), index=False
        )

        return perturbing_samples, []

    def attack(self, test, sampled_concepts, to="support"):
        test_samples = test
        perturbing_samples = [[] for _ in range(len(sampled_concepts))]

        for i, perturbation_type in enumerate(tqdm(sampled_concepts.keys())):
            for j, concept in enumerate(sampled_concepts[perturbation_type]):
                for sample in tqdm(test_samples):
                    if sample[-1][0] == 1:
                        label_txt = "the original claim must logically entail the new claim"
                    elif sample[-1][1] == 1:
                        label_txt = "the new claim must logically entail the original claim"
                    else:
                        label_txt = "the original claim must logically entail the new claim and vice versa"

                    word = concept[0]
                    if int(self.config["num_words"]) > 1:
                        words = [word]
                        for k in range(1,int(self.config["num_words"])):
                            words.append(sampled_concepts[perturbation_type][(j+k) % len(sampled_concepts[perturbation_type])][0])
                        word = ", ".join(words)

                    claim1 = self.generate_claim_with_openai(sample[0], word, label_txt, sample[-1])
                    if claim1 == -1:
                        continue

                    perturbing_samples[i].append([len(perturbing_samples[i])-1, claim1, sample[1], sample[-1], f"{perturbation_type}_{to}"])

        if config["stereotype"]:
            path = f"data/antonym/{self.config['dataset']}_openai_generated_stereotype/"
        else:
            path = f"data/antonym/{self.config['dataset']}_openai_generated/"

        os.makedirs(path, exist_ok=True)
        model_name = "qwen" if self.is_generative else "roberta"
        if int(self.config["num_words"]) > 1:
            model_name += f"_numwords{self.config['num_words']}"
        elif self.config["use_dev_tuning"]:
            model_name += "_dev_tuning"

        pd.DataFrame([row + [i] for i, sublist in enumerate(perturbing_samples) for row in sublist]).to_csv(
            os.path.join(path, f"generated_test_set_{to}_{model_name}.csv"), index=False
        )

    def attack_from_template(self, test, sampled_concepts, to="support"):
        test_samples = test
        perturbing_samples = [[] for _ in range(len(sampled_concepts))]

        for i, perturbation_type in enumerate(tqdm(sampled_concepts.keys())):
            for j, concept in enumerate(sampled_concepts[perturbation_type]):
                for sample in test_samples:
                    word = concept[0]
                    if int(self.config["num_words"]) > 1:
                        words = [word]
                        for k in range(1,int(self.config["num_words"])):
                            words.append(sampled_concepts[perturbation_type][(j+k) % len(sampled_concepts[perturbation_type])][0])
                        word = ". ".join(words)

                    claim = f"{word}. {sample[0]}"
                    perturbing_samples[i].append([len(perturbing_samples[i]) - 1, claim, sample[1], sample[-1], f"{perturbation_type}_{to}"])

        if config["stereotype"]:
            path = f"data/antonym/{self.config['dataset']}_openai_generated_stereotype/"
        else:
            #raise ValueError()
            path = f"data/antonym/{self.config['dataset']}_from_template/"

        os.makedirs(path, exist_ok=True)
        model_name = "qwen" if self.is_generative else "roberta"
        if int(self.config["num_words"]) > 1:
            model_name += f"_numwords{self.config['num_words']}"

        pd.DataFrame([row + [i] for i, sublist in enumerate(perturbing_samples) for row in sublist]).to_csv(
            os.path.join(path, f"generated_test_set_{to}_{model_name}.csv"), index=False
        )

    def defend(self, dataloader):
        val_labels, val_preds, claims, evidences = [], [], [], []
        for batch in dataloader:
            batch = tuple(t.to(get_device()) if not isinstance(t, list) and not isinstance(t, str) else t for t in batch)
            claim, evidence, ids_sent1, segs_sent1, att_mask_sent1, labels = batch

            with torch.no_grad():
                out = self.fc_model(ids_sent1, segs_sent1, att_mask_sent1)
                if out.shape[-1] > 3: # generative model
                    if labels.shape[-1] != 2:
                        # hardcoded initial ids for "support", "refute", "not enough information" for Qwen tokenizer
                        out.data = out.data[:, torch.tensor([1824, 83177, 537])]
                    else:
                        # hardcoded initial ids for "support" and "refute" for Qwen tokenizer
                        out.data = out.data[:, torch.tensor([1824, 83177])]

                preds = torch.max(out.data, 1)[1].cpu().numpy().tolist()
                labels_pos = torch.max(labels, 1)[1].cpu().numpy().tolist()
                val_labels.extend(labels_pos)
                val_preds.extend(preds)

            claims.extend(claim)
            evidences.extend(evidence)

        return val_labels, val_preds

    def get_examples(self, lst):
        lst = [[i, el[1], el[2], el[-2]] for i, el in enumerate(lst)]
        lst = self.processor._get_examples(lst)

        return lst

    def cv_attack(self, test, sampled_concepts, to="support", from_template=True):
        if from_template:
            self.attack_from_template(test, sampled_concepts, to)
        elif self.similarity_model:
            self.attack_with_similarity(test, sampled_concepts, to)
        else:
            self.attack(test, sampled_concepts, to)

    def run_from_concept_vectors(self, test, concept_vectors, k=5, num_samples=50, from_template=True):
        if "Nei" in concept_vectors.columns:
            num_classes = 3
        else:
            num_classes = 2

        # trying to flip to support prediction
        # we take non-supporting predictions and we analyze which perturbations modify the predictions the most

        test_samples = test[test.iloc[:, -1].apply(lambda x: x[0] == 0)]
        test_samples = deepcopy(get_random_samples(test_samples.values.tolist(), num_samples)) # identity function if num_samples == -1

        # support
        print("Generating support...")
        concept_vectors = concept_vectors.sort_values(by='Support', ascending=False)
        testing_concepts = self.get_testing_concepts(concept_vectors, k)

        self.cv_attack(test_samples, testing_concepts, to="support", from_template=from_template)

        # we do the same for refute
        test_samples = test[test.iloc[:, -1].apply(lambda x: x[1] == 0)]
        test_samples = deepcopy(get_random_samples(test_samples.values.tolist(), num_samples))

        print("Generating refute...")
        concept_vectors = concept_vectors.sort_values(by='Refute', ascending=False)
        testing_concepts = self.get_testing_concepts(concept_vectors, k)

        self.cv_attack(test_samples, testing_concepts, to="refute", from_template=from_template)

        if num_classes == 3:
            # nei
            test_samples = test[test.iloc[:, -1].apply(lambda x: x[2] == 0)]
            test_samples = deepcopy(get_random_samples(test_samples.values.tolist(), num_samples))
            print("Generating nei...")

            concept_vectors = concept_vectors.sort_values(by='Nei', ascending=False)
            testing_concepts = self.get_testing_concepts(concept_vectors, k)

            self.cv_attack(test_samples, testing_concepts, to="nei", from_template=from_template)

def get_data(config, positions=None, return_data=False):
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

    else:
        raise ValueError(
            f"{config['dataset']} is not a valid database name (choose between 'avtc', 'scifact', 'hover', 'fm2', 'politihop', 'vitaminc')")

    config["num_classes"] = num_classes

    data_train = processor.read_input_files(path_train, name="train")
    data_dev = processor.read_input_files(path_dev, name="dev")
    data_test = processor.read_input_files(path_test, name="test")
    if config["dataset"] == "scifact":
        # scifact test set is blind, so we use 20% of train as dev, and the dev as test
        tmp = data_dev
        data_dev = data_train[int(len(data_train) * 0.8):]
        data_train = data_train[:int(len(data_train) * 0.8)]
        data_test = tmp

    is_generative = ("llama" in config["model_name"].lower()
                     or "qwen" in config["model_name"].lower()
                     or "gpt"  in config["model_name"].lower())

    if positions is not None:
        data_train = get_samples_by_position(data_train, positions[0])
        data_dev = get_samples_by_position(data_dev, positions[1])
        if config["dataset"] == "hover" and is_generative:
            data_test = data_test[:250] + data_test[1000:1250]
        data_test = get_samples_by_position(data_test, positions[2])

        """
        for roberta-base, out of all the test samples, we take only 500 random samples that are predicted correctly by the model
        for generative models, out of the 500 tested samples, we keep the samples that are predicted correctly by the model
        """

    if return_data:
        return data_train, data_dev, data_test

    train_set = dataset(data_train)
    dev_set = dataset(data_dev)

    if is_generative:
        if config["dataset"] == "hover" and positions is None:
            data_test = data_test[:250] + data_test[1000:1250] #hover's first 500 test samples have all the same label
        else:
            data_test = data_test[:500]

    test_set = dataset(data_test)

    train_dataloader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_set, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

    return train_dataloader, dev_dataloader, test_dataloader

def get_samples_by_position(data, positions):
    new_data = [data[position] for position in positions]
    return new_data

def get_random_samples(data, num_samples):
    if num_samples >= len(data) or num_samples == -1:
        return data

    return random.sample(data, num_samples)

def get_matching_samples(predictions, targets):
    assert len(predictions) == len(targets)
    matching_samples = []
    for i in range(len(predictions)):
        if predictions[i] == targets[i]:
            matching_samples.append(i)

    return matching_samples

if __name__ == "__main__":
    config = get_config_adversarial()
    set_random_seeds(config["seed"])

    print("Extracting the whole datasets...")
    #train, dev, test = get_data(config)
    _, _, test = get_data(config)
    adv = AdversarialModel(config)

    if adv.is_generative:
        dirname = "predict_correct_qwen"
    else:
        dirname = "predict_correct"

    if not config["no_compute_predictions"]:
        # get fc correct predictions
        print("Predicting test samples...")
        predictions_test, targets_test = adv.defend(test)

        test_match = get_matching_samples(predictions_test, targets_test)

        # re-create the dataloaders, this time only with samples predicted correctly by the model
        print("Extracting samples in the datasets predicted correctly...")
        _, _, test = get_data(config, positions=[[], [], test_match], return_data=True)

        # saving the datasets...
        os.makedirs(dirname, exist_ok=True)

        if config["dataset"] == "hover":
            #hover test set first 500 samples have all the same label, so we take 250 from the start and 250 later on
            test = pd.concat([pd.DataFrame(test).iloc[:250], pd.DataFrame(test).iloc[1000:1250]])
        else:
            test = pd.DataFrame(test).iloc[:500]

        test.to_csv(os.path.join(dirname, f"{config['dataset']}_test.csv"))
    else:
        test = pd.read_csv(f"{dirname}/{config['dataset']}_test.csv")
        test = test.iloc[:,1:]
        cols = test.columns[2:]
        for col in cols:
            test[col] = test[col].apply(eval)

    if adv.is_generative:
        # if we are using a generative model, we use the model specific concept vectors, otherwise the ones from bert (different based on the dataset)
        path = f"data/antonym/Qwen-2.5-14B-Instruct_potency/concept_vectors.csv"
    else:
        if config["stereotype"]:
            path = f"data/antonym/{config['dataset']}_potency_stereotype/concept_vectors.csv"
        else:
            path = f"data/antonym/{config['dataset']}_potency/concept_vectors.csv"

    concept_vectors = pd.read_csv(path)

    if len(config["list_of_words"]) > 0:
        # keep only the words in the provided list
        # this is useful when we already know which words we want to test
        words_filt = config["list_of_words"]
        concept_vectors = concept_vectors[concept_vectors["Word pair"].isin(words_filt)]

    adv.run_from_concept_vectors(test, concept_vectors, k=config["k"], num_samples=config["num_samples"], from_template=config["not_from_template"])
