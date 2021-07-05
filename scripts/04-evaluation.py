#script to evaluate performance: calculate BLEU with sacreBLEU for both cleaned and uncleaned version of the outputs
#from https://blog.machinetranslation.io/compute-bleu-score/ 

import sys
import sacrebleu
from sacremoses import MosesDetokenizer
from pathlib import Path

md_ro = MosesDetokenizer(lang='ro')
base_path = Path(__file__).parent

#label translations. We need the word-level tokenized files because joeynmt reverses the bpe step...
path_test_raw=(base_path / "../data/02-preprocessed/02-01-preprocessed/test.30000.ro").resolve()
path_test_cleaned=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/test.30000.ro").resolve()

#MT output translations 
path_hypothesis_raw=(base_path / "../models/02-01-preprocessed/beamsize5/output.test").resolve()
path_hypothesis_cleaned= (base_path / "../models/02-02-bicleaner-preprocessed/beamsize5/output.test").resolve()

# Open the test dataset human translation file and detokenize the references
refs_raw = []

with open(path_test_raw) as raw_test:
    for line in raw_test: 
        line = line.strip().split() 
        line = md_ro.detokenize(line) 
        refs_raw.append(line)

refs_cleaned = []

with open(path_test_cleaned) as cleaned_test:
    for line in cleaned_test: 
        line = line.strip().split() 
        line = md_ro.detokenize(line) 
        refs_cleaned.append(line)   

print("Reference of raw sentence:", refs_raw[:3])
print("Reference of cleaned sentence:", refs_cleaned[:3])

refs_raw = [refs_raw]  # Yes, it is a list of list(s) as required by sacreBLEU
refs_cleaned = [refs_cleaned]

# Open the translation file by the NMT model and detokenize the predictions
preds_raw, preds_cleaned = [], []

with open(path_hypothesis_raw) as raw_pred:  
    for line in raw_pred: 
        line = line.strip().split() 
        line = md_ro.detokenize(line) 
        preds_raw.append(line)

#print("MTed raw 1st sentence:", preds_raw[0])    
#print("MTed cleaned 1st sentence:", preds_cleaned[0])    


with open(path_hypothesis_cleaned) as cleaned_pred:  
    for line in cleaned_pred: 
        line = line.strip().split() 
        line = md_ro.detokenize(line) 
        preds_cleaned.append(line)

# Calculate and print the BLEU score
bleu_raw = sacrebleu.corpus_bleu(preds_raw, refs_raw)
bleu_cleaned=sacrebleu.corpus_bleu(preds_cleaned, refs_cleaned)


#score is better on cleaned and on uncleaned test set --> make better structure for this
print("Bleu score on the uncleaned data on the test set is: ", bleu_raw.score)
print("Bleu score on the biclenaer cleaned data is: ", bleu_cleaned.score)