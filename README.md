# To calculate the stereotype pairs

```
python3 main.py --model_name models/roberta-base/cls/nofreeze/seed_1/vitaminc/vitaminc_model.pt --backbone roberta-base --dataset antonym --test_only --stereotype --batch_size 16 --potency
```

# Compute aggregates of the stereotype pairs

```
python3 compute_score_aggregates.py --path data/antonym/vitaminc/concept_vectors.csv --category-path data/antonym/category.csv
```

# To run llama test

add --potency and change dataset to antonym to compute the words

```
python3 main.py --model_name meta-llama/Meta-Llama-3-8B --dataset avtc --test_only --batch_size 4 --max_sent_len 8000
```
