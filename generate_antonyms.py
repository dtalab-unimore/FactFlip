import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import wordnet as wn
from tqdm import tqdm
import pandas as pd
import os

def get_all_antonyms():
    antonyms = set()

    for synset in tqdm(wn.all_synsets()):
        for lemma in synset.lemmas():
            for antonym in lemma.antonyms():
                word1 = lemma.name()
                word2 = antonym.name()
                #pair = tuple(sorted((word1, word2)))
                antonyms.add(word1)
                antonyms.add(word2)

    return antonyms

antonyms = get_all_antonyms()
print(f"Total antonyms found: {len(antonyms)}")
df = pd.DataFrame(antonyms, columns=["Word"])
os.makedirs("data/antonyms/", exist_ok=True)
df.to_csv("data/antonyms/antonyms.csv", index=False)
