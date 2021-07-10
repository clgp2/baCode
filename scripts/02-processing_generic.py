#file to preprocess a source and a target txt file into word-level and subword-level tokenized files, input for a nmt system
import os
from pathlib import Path
from subword_nmt import apply_bpe
from sacremoses import MosesTokenizer
import sys

mt_en = MosesTokenizer(lang='en')
mt_ro = MosesTokenizer(lang='ro')

#input files
source_input_file = sys.argv[1]
target_input_file = sys.argv[2]

with open(source_input_file) as rawfile, open("source_input_file_tok", "w") as tokfile:
    for i, line in enumerate(rawfile):
        data=line.rstrip()
        tokfile.write(mt_en.tokenize(data, return_str=True)+"\n")

print("Completed writing EN_tok.txt")

with open(target_input_file) as rawfile, open("target_input_file_tok", "w") as tokfile:
    for i, line in enumerate(rawfile):
        data=line.rstrip()
        tokfile.write(mt_ro.tokenize(data, return_str=True)+"\n")

print("Completed writing RO_tok.txt")

bpe_file="/home/bernadeta/BA_code/data/02-preprocessed/bpe.codes.8000"

with open(bpe_file, "r") as merge_file:
    bpe=apply_bpe.BPE(codes=merge_file)

with open("source_input_file_tok") as src_tok_file, open("train.src.8000.bpe", "w") as src_bpefile:
    for i, line in enumerate(src_tok_file):
        data=line.strip()
        bpedata=bpe.process_line(data)
        src_bpefile.write(bpedata+"\n")

count_trg=0
with open("target_input_file_tok") as trg_tok_file, open("train.trg.8000.bpe", "w") as trg_bpefile:
    for i, line in enumerate(trg_tok_file):
        data=line.strip()
        bpedata=bpe.process_line(data)
        trg_bpefile.write(bpedata+"\n")
        