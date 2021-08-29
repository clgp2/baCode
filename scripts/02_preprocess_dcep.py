import shutil
import os
from pathlib import Path
import urllib.request

base_path = Path(__file__).parent
path_preprocessed=(base_path / "../data/DCEP/02-preprocessed/").resolve()
path_preprocessed.mkdir(parents=True, exist_ok=True)

L2_train_en= (base_path / "../data/DCEP/01-intermediate/L2_strong/L2_train.en").resolve()
L2_train_ro=(base_path / "../data/DCEP/01-intermediate/L2_strong/L2_train.ro").resolve()
L2_train_en_tok = str(L2_train_en)+".tok"
L2_train_ro_tok = str(L2_train_ro)+".tok"


L2_test_en=(base_path / "../data/DCEP/01-intermediate/L2_strong/L2_test.en").resolve()
L2_test_ro=(base_path / "../data/DCEP/01-intermediate/L2_strong/L2_test.ro").resolve()
L2_test_en_tok = str(L2_test_en)+".tok"
L2_test_ro_tok = str(L2_test_ro)+".tok"

L2_dev_en=(base_path / "../data/DCEP/01-intermediate/L2_strong/L2_dev.en").resolve()
L2_dev_ro=(base_path / "../data/DCEP/01-intermediate/L2_strong/L2_dev.ro").resolve()
L2_dev_en_tok = str(L2_dev_en)+".tok"
L2_dev_ro_tok = str(L2_dev_ro)+".tok"

source_lang="en"
target_lang="ro"

os.system(f"sacremoses -l {source_lang} -j 8 tokenize < {L2_train_en} > {L2_train_en_tok}")
os.system(f"sacremoses -l {target_lang} -j 8 tokenize < {L2_train_ro} > {L2_train_ro_tok}")

os.system(f"sacremoses -l {source_lang} -j 8 tokenize < {L2_test_en} > {L2_test_en_tok}")
os.system(f"sacremoses -l {target_lang} -j 8 tokenize < {L2_test_ro} > {L2_test_ro_tok}")

os.system(f"sacremoses -l {source_lang} -j 8 tokenize < {L2_dev_en} > {L2_dev_en_tok}")
os.system(f"sacremoses -l {target_lang} -j 8 tokenize < {L2_dev_ro} > {L2_dev_ro_tok}")

L1_train_en=(base_path / "../data/DCEP/01-intermediate/L1_basic/L1_train.en").resolve()
L1_train_ro=(base_path / "../data/DCEP/01-intermediate/L1_basic/L1_train.ro").resolve()

L1_train_en_tok = str(L1_train_en)+".tok"
L1_train_ro_tok = str(L1_train_ro)+".tok"

os.system(f"sacremoses -l {source_lang} -j 8 tokenize < {L1_train_en} > {L1_train_en_tok}")
os.system(f"sacremoses -l {target_lang} -j 8 tokenize < {L1_train_ro} > {L1_train_ro_tok}")

L3_train_en=(base_path / "../data/DCEP/01-intermediate/L3_intermediate/L3_train.en").resolve()
L3_train_ro=(base_path / "../data/DCEP/01-intermediate/L3_intermediate/L3_train.ro").resolve()

L3_train_en_tok = str(L3_train_en)+".tok"
L3_train_ro_tok = str(L3_train_ro)+".tok"

os.system(f"sacremoses -l {source_lang} -j 8 tokenize < {L3_train_en} > {L3_train_en_tok}")
os.system(f"sacremoses -l {target_lang} -j 8 tokenize < {L3_train_ro} > {L3_train_ro_tok}")

#this bpe_size is recommended for small to medium sized datasets (30K-1.3M)
bpe_size=8000

#learn the vocab from the bigger training files resulted after basic cleaning
#the problem with this approach is that the L1 dataset contains wrong languages, which makes it possible to have wrong translations
os.system(f"subword-nmt learn-joint-bpe-and-vocab --input {L1_train_en_tok} {L1_train_ro_tok} -s {bpe_size} -o bpe.codes.{bpe_size} --write-vocabulary vocab.en vocab.ro")

#apply BPE
os.system(f"subword-nmt apply-bpe -c bpe.codes.{bpe_size} --vocabulary vocab.en --vocabulary-threshold 50 < {L1_train_en_tok} > L1_train_tok.bpe.en")
os.system(f"subword-nmt apply-bpe -c bpe.codes.{bpe_size} --vocabulary vocab.ro --vocabulary-threshold 50 < {L1_train_ro_tok} > L1_train_tok.bpe.ro")

os.system(f"subword-nmt apply-bpe -c bpe.codes.{bpe_size} --vocabulary vocab.en --vocabulary-threshold 50 < {L3_train_en_tok} > L3_train_tok.bpe.en")
os.system(f"subword-nmt apply-bpe -c bpe.codes.{bpe_size} --vocabulary vocab.ro --vocabulary-threshold 50 < {L3_train_ro_tok} > L3_train_tok.bpe.ro")

os.system(f"subword-nmt apply-bpe -c bpe.codes.{bpe_size} --vocabulary vocab.en --vocabulary-threshold 50 < {L2_train_en_tok} > L2_train_tok.bpe.en")
os.system(f"subword-nmt apply-bpe -c bpe.codes.{bpe_size} --vocabulary vocab.ro --vocabulary-threshold 50 < {L2_train_ro_tok} > L2_train_tok.bpe.ro")

os.system(f"subword-nmt apply-bpe -c bpe.codes.{bpe_size} --vocabulary vocab.en --vocabulary-threshold 50 < {L2_dev_en_tok} > L2_dev_tok.bpe.en")
os.system(f"subword-nmt apply-bpe -c bpe.codes.{bpe_size} --vocabulary vocab.ro --vocabulary-threshold 50 < {L2_dev_ro_tok} > L2_dev_tok.bpe.ro")

os.system(f"subword-nmt apply-bpe -c bpe.codes.{bpe_size} --vocabulary vocab.en --vocabulary-threshold 50 < {L2_test_en_tok} > L2_test_tok.bpe.en")
os.system(f"subword-nmt apply-bpe -c bpe.codes.{bpe_size} --vocabulary vocab.ro --vocabulary-threshold 50 < {L2_test_ro_tok} > L2_test_tok.bpe.ro")

urllib.request.urlretrieve ("https://raw.githubusercontent.com/joeynmt/joeynmt/master/scripts/build_vocab.py", "./build_vocab.py")

os.system(f"python build_vocab.py L1_train_tok.bpe.en L1_train_tok.bpe.ro --output_path vocab.txt")

shutil.move("bpe.codes.8000", path_preprocessed)
shutil.move("vocab.en", path_preprocessed)
shutil.move("vocab.ro", path_preprocessed)
shutil.move("vocab.txt", path_preprocessed)
shutil.move("build_vocab.py", path_preprocessed)

path_preprocessed_L1=(base_path / "../data/DCEP/02-preprocessed/L1_basic").resolve()
path_preprocessed_L1.mkdir(parents=True, exist_ok=True)

shutil.move("L1_train_tok.bpe.en", path_preprocessed_L1)
shutil.move("L1_train_tok.bpe.ro", path_preprocessed_L1)

path_preprocessed_L3=(base_path / "../data/DCEP/02-preprocessed/L3_intermediate").resolve()
path_preprocessed_L3.mkdir(parents=True, exist_ok=True)

shutil.move("L3_train_tok.bpe.en", path_preprocessed_L3)
shutil.move("L3_train_tok.bpe.ro", path_preprocessed_L3)

path_preprocessed_L2=(base_path / "../data/DCEP/02-preprocessed/L2_strong").resolve()
path_preprocessed_L2.mkdir(parents=True, exist_ok=True)

shutil.move("L2_train_tok.bpe.en", path_preprocessed_L2)
shutil.move("L2_train_tok.bpe.ro", path_preprocessed_L2)

shutil.move("L2_dev_tok.bpe.en", path_preprocessed_L2)
shutil.move("L2_dev_tok.bpe.ro", path_preprocessed_L2)

shutil.move("L2_test_tok.bpe.en", path_preprocessed_L2)
shutil.move("L2_test_tok.bpe.ro", path_preprocessed_L2)
