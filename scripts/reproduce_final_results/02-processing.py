###############################################################################
# #############################################################################
# #############################################################################
# this script is not ready! use 02-processing notebook instead

#todo: create meaningful functions for all this stuff
#todo for final version start timer for each script to know how long it takes to execute on specified cpu
#general language pair with .src and .trg instead of en and ro --> provide yaml file
#use dictionary to reduce number of file paths
#disclaminer: code partially inspired by joey-demo.ipynb
#os.system is bad practice

"""""
Script to turn data into modelling input

Steps performed in this file:

    1. word-level tokenization with sacremoses
    2. subword-level tokenization with BPE
    
for each L1, L2 and L3 datasets

"""""
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
#from subword_nmt.apply_bpe import BPE
import urllib.request
from sacremoses import MosesTokenizer

src_lang="en"
trg_lang="ro"

mt_en = MosesTokenizer(lang='en')
mt_ro = MosesTokenizer(lang='ro')

base_path = Path(__file__).parent

path_build_vocab= (base_path / "../data/02-preprocessed/build_vocab.py").resolve()

path_fulldata= (base_path / "../data/01-intermediate/EN-RO-bisentences_new.txt").resolve()

path_source_raw= (base_path / "../data/01-intermediate/01-01-fulldata/EN_raw.txt").resolve()
path_target_raw= (base_path / "../data/01-intermediate/01-01-fulldata/RO_raw.txt").resolve()

path_source_raw_tok= (base_path / "../data/02-preprocessed/02-01-preprocessed/EN_raw_tok.txt").resolve()
path_target_raw_tok=(base_path / "../data/02-preprocessed/02-01-preprocessed/RO_raw_tok.txt").resolve()

path_source_train1=(base_path / "../data/02-preprocessed/02-01-preprocessed/train.en").resolve()
path_target_train1=(base_path / "../data/02-preprocessed/02-01-preprocessed/train.ro").resolve()

path_source_test1=(base_path / "../data/02-preprocessed/02-01-preprocessed/test.en").resolve()
path_target_test1=(base_path / "../data/02-preprocessed/02-01-preprocessed/test.ro").resolve()

path_source_dev1=(base_path / "../data/02-preprocessed/02-01-preprocessed/dev.en").resolve()
path_target_dev1=(base_path / "../data/02-preprocessed/02-01-preprocessed/dev.ro").resolve()

path_bpe_codes1= (base_path / "../data/02-preprocessed/02-01-preprocessed/bpe.codes.30000").resolve()
path_vocab1= (base_path / "../data/02-preprocessed/02-01-preprocessed/vocab.txt").resolve()
path_joint_train1= (base_path / "../data/02-preprocessed/02-01-preprocessed/train.en-ro").resolve()

####################################################################################################################################################

path_cleaned= (base_path / "../data/01-intermediate/01-02-bicleaner-cleaned/02-cleaned_bisentences.txt").resolve()

path_source_cleaned= (base_path / "../data/01-intermediate/01-02-bicleaner-cleaned/EN_cleaned.txt").resolve()
path_target_cleaned=(base_path / "../data/01-intermediate/01-02-bicleaner-cleaned/RO_cleaned.txt").resolve()

path_source_tok= (base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/EN_tok.txt").resolve()
path_target_tok=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/RO_tok.txt").resolve()

path_source_train2=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/train.en").resolve()
path_target_train2=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/train.ro").resolve()

path_source_test2=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/test.en").resolve()
path_target_test2=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/test.ro").resolve()

path_source_dev2=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/dev.en").resolve()
path_target_dev2=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/dev.ro").resolve()

path_bpe_codes2= (base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/bpe.codes.30000").resolve()
path_vocab2= (base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/vocab.txt").resolve()
path_joint_train2= (base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/train.en-ro").resolve()

#1. word-level tokenization with sacremoses==================================================================================================
#1.1 tokenize full version
source_raw=pd.read_csv(path_fulldata, usecols=[0])
target_raw=pd.read_csv(path_fulldata, usecols=[1])

with open(path_source_raw) as rawfile, open(path_source_raw_tok, "w") as tokfile:
    for i, line in enumerate(rawfile):
        data=line.rstrip()
        tokfile.write(mt_en.tokenize(data, return_str=True)+"\n")

print("Completed writing EN_raw_tok.txt")

with open(path_target_raw) as rawfile, open(path_target_raw_tok, "w") as tokfile:
    for i, line in enumerate(rawfile):
        data=line.rstrip()
        tokfile.write(mt_ro.tokenize(data, return_str=True)+"\n")

print("Completed writing RO_raw_tok.txt")

#1.2 tokenize cleaned version
source_cleaned=pd.read_csv(path_cleaned, usecols=[0])
target_cleaned=pd.read_csv(path_cleaned, usecols=[1])

with open(path_source_cleaned) as rawfile, open(path_source_tok, "w") as tokfile:
    for i, line in enumerate(rawfile):
        data=line.rstrip()
        tokfile.write(mt_en.tokenize(data, return_str=True)+"\n")

print("Completed writing EN_cleaned_tok.txt")

with open(path_target_cleaned) as rawfile, open(path_target_tok, "w") as tokfile:
    for i, line in enumerate(rawfile):
        data=line.rstrip()
        tokfile.write(mt_ro.tokenize(data, return_str=True)+"\n")

print("Completed writing RO_cleaned_tok.txt")


#2. subword-level tokenization with BPE=========================================================================================================

#2.1 subword-level tokenization with BPE for fulldata files
#reading paper "Finding the Optimal Vocabulary Size for NMT"
bpe_size=8000

os.system(f"cat {path_source_train1} {path_target_train1} > {path_joint_train1}")

os.system(f"subword-nmt learn-bpe --input {path_joint_train1} -s {bpe_size} -o {path_bpe_codes1} ")

src_files = {'train': path_source_train1, 'dev': path_source_dev1, 'test': path_source_test1}
trg_files = {'train': path_target_train1, 'dev': path_target_dev1, 'test': path_target_test1}

src_bpe_files = {}
trg_bpe_files = {}

for split in ['train', 'dev', 'test']:
  src_input_file = src_files[split]
  trg_input_file = trg_files[split]
  src_output_file = str(src_input_file).replace(split, f'{split}.{bpe_size}.bpe')
  trg_output_file = str(trg_input_file).replace(split, f'{split}.{bpe_size}.bpe')
  src_bpe_files[split] = src_output_file
  trg_bpe_files[split] = trg_output_file

  os.system(f"subword-nmt apply-bpe -c {path_bpe_codes1} <{src_input_file}> {src_output_file}")
  os.system(f"subword-nmt apply-bpe -c {path_bpe_codes1} <{trg_input_file}> {trg_output_file}")

#3.2 subword-level tokenization with BPE for cleaned files
os.system(f"cat {path_source_train2} {path_target_train2} > {path_joint_train2}")

os.system(f"subword-nmt learn-bpe --input {path_joint_train2} -s {bpe_size} -o {path_bpe_codes2} ")

src_files_cleaned = {'train': path_source_train2, 'dev': path_source_dev2, 'test': path_source_test2}
trg_files_cleaned = {'train': path_target_train2, 'dev': path_target_dev2, 'test': path_target_test2}

src_bpe_files_cleaned = {}
trg_bpe_files_cleaned = {}

for split in ['train', 'dev', 'test']:
  src_input_file_cleaned = src_files_cleaned[split]
  trg_input_file_cleaned = trg_files_cleaned[split]
  src_output_file_cleaned = str(src_input_file_cleaned).replace(split, (f'{split}.{bpe_size}.bpe'))
  trg_output_file_cleaned = str(trg_input_file_cleaned).replace(split, (f'{split}.{bpe_size}.bpe'))
  src_bpe_files_cleaned[split] = src_output_file_cleaned
  trg_bpe_files_cleaned[split] = trg_output_file_cleaned

  os.system(f"subword-nmt apply-bpe -c {path_bpe_codes2} <{src_input_file_cleaned}> {src_output_file_cleaned}")
  os.system(f"subword-nmt apply-bpe -c {path_bpe_codes2} <{trg_input_file_cleaned}> {trg_output_file_cleaned}")

# #vocabulary threshold for subword-nmt ?? -> best rpactices from site
# #make a loop for all these. it is possible that the train.src, etc. deleted before subword-nmt can use them, because os.system starts a new process??
# #if enough time left, run experiments with word-level tokenized input files only
# os.remove(path_source_tok)
# os.remove(path_target_tok)

# os.remove(path_source_train2)
# os.remove(path_target_train2)

# os.remove(path_source_dev2)
# os.remove(path_target_dev2)

# os.remove(path_source_test2)
# os.remove(path_target_test2)

# create vocabulary
urllib.request.urlretrieve ("https://raw.githubusercontent.com/joeynmt/joeynmt/master/scripts/build_vocab.py", path_build_vocab)

# vocab_src_file=src_bpe_files["train"]
# vocab_trg_file=trg_bpe_files["train"]

vocab_src_file_cleaned=src_bpe_files_cleaned["train"]
vocab_trg_file_cleaned=trg_bpe_files_cleaned["train"]

#os.system(f"python {path_build_vocab} {vocab_src_file} {vocab_trg_file} --output_path {path_vocab1}")
os.system(f"python {path_build_vocab} {vocab_src_file_cleaned} {vocab_trg_file_cleaned} --output_path {path_vocab2}")

# os.remove(path_build_vocab)
# os.remove(path_joint_train2)
# os.remove(path_bpe_codes2)
