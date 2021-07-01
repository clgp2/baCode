#script to evaluate performance: calculate BLEU with sacreBLEU
import sys
import sacrebleu
from sacremoses import MosesDetokenizer
from pathlib import Path

md_ro = MosesDetokenizer(lang='ro')
base_path = Path(__file__).parent

#label translations
path_test_raw=(base_path / "../data/02-preprocessed/02-01-preprocessed/test.ro").resolve()
path_test_cleaned=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/test.ro").resolve()

#MT output translations --> probably somewhere in the models/ directory
# path_hypothesis_raw=
# path_hypothesis_cleaned=

# Open the test dataset human translation file and detokenize the references
refs = []

with open(path_test_raw) as test:
    for line in test: 
        line = line.strip().split() 
        line = md_ro.detokenize(line) 
        refs.append(line)
    
print("Reference 1st sentence:", refs[0])

refs = [refs]  # Yes, it is a list of list(s) as required by sacreBLEU


# Open the translation file by the NMT model and detokenize the predictions
preds = []

with open("target.pred") as pred:  
    for line in pred: 
        line = line.strip().split() 
        line = md_ro.detokenize(line) 
        preds.append(line)

print("MTed 1st sentence:", preds[0])    


# Calculate and print the BLEU score
bleu = sacrebleu.corpus_bleu(preds, refs)
print(bleu.score)
