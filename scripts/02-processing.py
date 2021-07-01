#todo: get rid of repeatable code!
#todo for final version start timer for each script to know how long it takes to execute on specified cpu
#general language pair with .src and .trg instead of en and ro --> provide arguments for script to choose language pair

"""""
Script to turn data into modelling input. creates tokenized and byte-pair-encoded input files for 02-01-preprocessed/ and for 02-02-bicleaner-preprocessed/

-  bicleaner needs kenlm. to install with:

git clone https://github.com/kpu/kenlm
cd kenlm
python3.7 -m pip install . --install-option="--max_order 7"
mkdir -p build && cd build
cmake .. -DKENLM_MAX_ORDER=7 -DCMAKE_INSTALL_PREFIX:PATH=/your/prefix/path
make -j all install


Steps performed in this file:

    1. word-level tokenization with sacremoses
    2. split into traindevtest sets
    3. subword-level tokenization with BPE
    
for each 02-01-preprocessed and 02-02-bicleaner-preprocessed 

"""""
import os
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from subword_nmt.apply_bpe import BPE
import urllib.request

#for performance: better fast-mosestokenizer?
from sacremoses import MosesTokenizer
mt_en = MosesTokenizer(lang='en')
mt_ro = MosesTokenizer(lang='ro')

base_path = Path(__file__).parent

path_build_vocab= (base_path / "../data/02-preprocessed/build_vocab.py").resolve()

#1. word-level tokenization with sacremoses==================================================================================================
path_fulldata= (base_path / "../data/01-intermediate/EN-RO-bisentences_new.txt").resolve()

path_source_raw= (base_path / "../data/01-intermediate/01-01-fulldata/EN_raw.txt").resolve()
path_target_raw= (base_path / "../data/01-intermediate/01-01-fulldata/RO_raw.txt").resolve()

path_source_raw_tok= (base_path / "../data/02-preprocessed/02-01-preprocessed/EN_raw_tok.txt").resolve()
path_target_raw_tok=(base_path / "../data/02-preprocessed/02-01-preprocessed/RO_raw_tok.txt").resolve()

path_source_train1=(base_path / "../data/02-preprocessed/02-01-preprocessed/train.src").resolve()
path_target_train1=(base_path / "../data/02-preprocessed/02-01-preprocessed/train.trg").resolve()

path_source_test1=(base_path / "../data/02-preprocessed/02-01-preprocessed/test.src").resolve()
path_target_test1=(base_path / "../data/02-preprocessed/02-01-preprocessed/test.trg").resolve()

path_source_dev1=(base_path / "../data/02-preprocessed/02-01-preprocessed/dev.src").resolve()
path_target_dev1=(base_path / "../data/02-preprocessed/02-01-preprocessed/dev.trg").resolve()

path_source_train_bpe1=(base_path / "../data/02-preprocessed/02-01-preprocessed/train.bpe.en").resolve()
path_target_train_bpe1=(base_path / "../data/02-preprocessed/02-01-preprocessed/train.bpe.ro").resolve()

path_source_test_bpe1=(base_path / "../data/02-preprocessed/02-01-preprocessed/test.bpe.en").resolve()
path_target_test_bpe1=(base_path / "../data/02-preprocessed/02-01-preprocessed/test.bpe.ro").resolve()

path_source_dev_bpe1=(base_path / "../data/02-preprocessed/02-01-preprocessed/dev.bpe.en").resolve()
path_target_dev_bpe1=(base_path / "../data/02-preprocessed/02-01-preprocessed/dev.bpe.ro").resolve()

path_bpe_codes1= (base_path / "../data/02-preprocessed/02-01-preprocessed/bpe.codes.30000").resolve()
path_vocab1= (base_path / "../data/02-preprocessed/02-01-preprocessed/vocab.txt").resolve()
path_joint_train1= (base_path / "../data/02-preprocessed/02-01-preprocessed/train.en-ro").resolve()

####################################################################################################################################################

path_cleaned= (base_path / "../data/01-intermediate/01-02-bicleaner-cleaned/02-cleaned_bisentences.txt").resolve()

path_source_cleaned= (base_path / "../data/01-intermediate/01-02-bicleaner-cleaned/EN_cleaned.txt").resolve()
path_target_cleaned=(base_path / "../data/01-intermediate/01-02-bicleaner-cleaned/RO_cleaned.txt").resolve()

path_source_tok= (base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/EN_tok.txt").resolve()
path_target_tok=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/RO_tok.txt").resolve()

path_source_train2=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/train.src").resolve()
path_target_train2=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/train.trg").resolve()

path_source_test2=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/test.src").resolve()
path_target_test2=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/test.trg").resolve()

path_source_dev2=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/dev.src").resolve()
path_target_dev2=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/dev.trg").resolve()

path_source_train_bpe2=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/train.bpe.en").resolve()
path_target_train_bpe2=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/train.bpe.ro").resolve()

path_source_test_bpe2=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/test.bpe.en").resolve()
path_target_test_bpe2=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/test.bpe.ro").resolve()

path_source_dev_bpe2=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/dev.bpe.en").resolve()
path_target_dev_bpe2=(base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/dev.bpe.ro").resolve()

path_bpe_codes2= (base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/bpe.codes.30000").resolve()
path_vocab2= (base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/vocab.txt").resolve()
path_joint_train2= (base_path / "../data/02-preprocessed/02-02-bicleaner-preprocessed/train.en-ro").resolve()

#1.1 tokenize full version
source_raw=pd.read_csv(path_fulldata, usecols=[0])
target_raw=pd.read_csv(path_fulldata, usecols=[1])

with open(path_source_raw) as rawfile, open(path_source_raw_tok, "w") as tokfile:
    for i, line in enumerate(rawfile):
        data=line.rstrip()
        tokfile.write(mt_en.tokenize(data, return_str=True)+"\n")

with open(path_target_raw) as rawfile, open(path_target_raw_tok, "w") as tokfile:
    for i, line in enumerate(rawfile):
        data=line.rstrip()
        tokfile.write(mt_ro.tokenize(data, return_str=True)+"\n")

#1.2 tokenize cleaned version
source_cleaned=pd.read_csv(path_cleaned, usecols=[0])
target_cleaned=pd.read_csv(path_cleaned, usecols=[1])

with open(path_source_cleaned) as rawfile, open(path_source_tok, "w") as tokfile:
    for i, line in enumerate(rawfile):
        data=line.rstrip()
        tokfile.write(mt_en.tokenize(data, return_str=True)+"\n")

with open(path_target_cleaned) as rawfile, open(path_target_tok, "w") as tokfile:
    for i, line in enumerate(rawfile):
        data=line.rstrip()
        tokfile.write(mt_ro.tokenize(data, return_str=True)+"\n")

# 2. split into traindevtest sets==============================================================================================================
# 2.1 split full version
df_source_raw_tok= pd.read_csv(path_source_raw_tok, header=None, sep="\n")
df_target_raw_tok= pd.read_csv(path_target_raw_tok, header=None, sep="\n")

source_train_raw, source_test_raw, target_train_raw, target_test_raw = train_test_split(df_source_raw_tok, df_target_raw_tok, test_size=2000, random_state=42)
source_train_raw, source_dev_raw, target_train_raw, target_dev_raw = train_test_split(source_train_raw, target_train_raw, test_size=2000, random_state=42)

source_train_raw.to_csv(path_source_train1, header=None, index=None)
target_train_raw.to_csv(path_target_train1, header=None, index=None)

source_test_raw.to_csv(path_source_test1, header=None, index=None)
target_test_raw.to_csv(path_target_test1, header=None, index=None)

source_dev_raw.to_csv(path_source_dev1, header=None, index=None)
target_dev_raw.to_csv(path_target_dev1, header=None, index=None)

# 2.2 split cleaned version
df_source_tok= pd.read_csv(path_source_tok, header=None, sep="\n")
df_target_tok=pd.read_csv(path_target_tok, header=None, sep="\n")

source_train, source_test, target_train, target_test = train_test_split(df_source_tok, df_target_tok, test_size=2000, random_state=42)
source_train, source_dev, target_train, target_dev = train_test_split(source_train, target_train, test_size=2000, random_state=42)

source_train.to_csv(path_source_train2, header=None, index=None)
target_train.to_csv(path_target_train2, header=None, index=None)

source_test.to_csv(path_source_test2, header=None, index=None)
target_test.to_csv(path_target_test2, header=None, index=None)

source_dev.to_csv(path_source_dev2, header=None, index=None)
target_dev.to_csv(path_target_dev2, header=None, index=None)

#3. subword-level tokenization with BPE=========================================================================================================
#use python API for subword-nmt?

#3.1 subword-level tokenization with BPE for fulldata files
os.system(f"cat {path_source_train1} {path_target_train1} > {path_joint_train1}")

os.system(f"subword-nmt learn-bpe --input {path_joint_train1} -s 30000 -o {path_bpe_codes1} ")

os.system(f"subword-nmt apply-bpe -c {path_bpe_codes1} < {path_source_train1} > {path_source_train_bpe1}")
os.system(f"subword-nmt apply-bpe -c {path_bpe_codes1} < {path_target_train1} > {path_target_train_bpe1}")
 
os.system(f"subword-nmt apply-bpe -c {path_bpe_codes1} < {path_source_dev1} > {path_source_dev_bpe1}")
os.system(f"subword-nmt apply-bpe -c {path_bpe_codes1} < {path_target_dev1} > {path_target_dev_bpe1}")

os.system(f"subword-nmt apply-bpe -c {path_bpe_codes1} < {path_source_test1} > {path_source_test_bpe1}")
os.system(f"subword-nmt apply-bpe -c {path_bpe_codes1} < {path_target_test1} > {path_target_test_bpe1}")

#3.2 subword-level tokenization with BPE for cleaned files
os.system(f"cat {path_source_train2} {path_target_train2} > {path_joint_train2}")

os.system(f"subword-nmt learn-bpe --input {path_joint_train2} -s 30000 -o {path_bpe_codes2} ")

os.system(f"subword-nmt apply-bpe -c {path_bpe_codes2} < {path_source_train2} > {path_source_train_bpe2}")
os.system(f"subword-nmt apply-bpe -c {path_bpe_codes2} < {path_target_train2} > {path_target_train_bpe2}")
 
os.system(f"subword-nmt apply-bpe -c {path_bpe_codes2} < {path_source_dev2} > {path_source_dev_bpe2}")
os.system(f"subword-nmt apply-bpe -c {path_bpe_codes2} < {path_target_dev2} > {path_target_dev_bpe2}")

os.system(f"subword-nmt apply-bpe -c {path_bpe_codes2} < {path_source_test2} > {path_source_test_bpe2}")
os.system(f"subword-nmt apply-bpe -c {path_bpe_codes2} < {path_target_test2} > {path_target_test_bpe2}")


#vocabulary threshold for subword-nmt ?? -> best rpactices from site
#make a loop for all these. it is possible that the train.src, etc. deleted before subword-nmt can use them, because os.system starts a new process??
#if enough time left, run experiments with word-level tokenized input files only
os.remove(path_source_tok)
os.remove(path_target_tok)

os.remove(path_source_train2)
os.remove(path_target_train2)

os.remove(path_source_dev2)
os.remove(path_target_dev2)

os.remove(path_source_test2)
os.remove(path_target_test2)

# create vocabulary
urllib.request.urlretrieve ("https://raw.githubusercontent.com/joeynmt/joeynmt/master/scripts/build_vocab.py", path_build_vocab)
os.system(f"python {path_build_vocab} {path_source_train_bpe1} {path_target_train_bpe1} --output_path {path_vocab1}")
os.system(f"python {path_build_vocab} {path_source_train_bpe2} {path_target_train_bpe2} --output_path {path_vocab2}")

os.remove(path_build_vocab)
os.remove(path_joint_train2)
os.remove(path_bpe_codes2)
