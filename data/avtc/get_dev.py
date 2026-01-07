import json
import random

with open("train.json", "r") as reader:
	train = json.load(reader)

new_train = []
for sample in train:
	if sample["label"] != "Conflicting Evidence/Cherrypicking":
		new_train.append(sample)

train = new_train
num_samples = int(len(train) * 0.15)
indices = random.sample(range(len(train)), num_samples)

new_train = []
new_dev = []
for i, sample in enumerate(train):
	if i in indices:
		new_dev.append(sample)
	else:
		new_train.append(sample)

with open("train.json", "w") as writer:
	json.dump(new_train, writer)

with open("dev.json", "w") as writer:
	json.dump(new_dev, writer)

print(len(new_train))
print(len(new_dev))
