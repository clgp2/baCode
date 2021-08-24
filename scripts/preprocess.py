#file to preprocess a source and a target txt file into word-level tokenized files, input for a nmt system
#usage example: python preprocess.py source.txt en target.txt ro
from sacremoses import MosesTokenizer
import sys

#inputs
source_input_file = sys.argv[1]
source_lang_id=sys.argv[2]

target_input_file = sys.argv[3]
target_lang_id=sys.argv[4]

mt_source = MosesTokenizer(lang=source_lang_id)
mt_target = MosesTokenizer(lang=target_lang_id)


with open(source_input_file) as rawfile, open("source.txt", "w") as tokfile:
    for i, line in enumerate(rawfile):
        data=line.rstrip()
        tokfile.write(mt_source.tokenize(data, return_str=True)+"\n")

print("Completed tokenizing source file")

with open(target_input_file) as rawfile, open("target.txt", "w") as tokfile:
    for i, line in enumerate(rawfile):
        data=line.rstrip()
        tokfile.write(mt_target.tokenize(data, return_str=True)+"\n")

print("Completed tokenizing target file")

        