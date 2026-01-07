# Command Hotflip

```
python3 -m autoprompt.create_trigger --model_name ../FactCheckingBias/models/roberta-base/cls/nofreeze/seed_1/avtc/avtc_model.pt --backbone roberta-base --dataset avtc --potency --batch_size 16 --iters 1 --accumulation-steps 1000 --k 5 --sort_by_similarity
```

# Command HotFlip, not sorted by similarity

```
python3 -m autoprompt.create_trigger --model_name ../FactCheckingBias/models/roberta-base/cls/nofreeze/seed_1/vitaminc/vitaminc_model.pt --backbone roberta-base --dataset vitaminc --train --iters 1
```
