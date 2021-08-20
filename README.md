# sequence to sequence neural machine translation English-Romanian
## Creation of a virtual environment
More about venv virtual environments can be found  [here](https://docs.python.org/3/library/venv.html)
venv - Creation of a new virtual environment (tested with python 3.8):
```
python3 -m venv /path/to/new/virtual/environment
```
installing recursive with pip:
```
pip install -r requirements.txt
```
## General workflow to train and evaluate a Neural Machine Translation System:

1. Download the data
* use scripts/00-raw.py to download the English-Romanian language pair from the Digital Corpus of the European Parliament (DCEP)

2. Clean & tokenize the data
 1. Clean
* use notebooks/01-intermediate.ipynb to clean the En-Ro DCEP data into L1, L2 and L3

General:
run bicleaner-hardrules with all or just some rules. Additionally to the documented rules [here](https://github.com/bitextor/bicleaner), You can also use:
* --disable_reply
* --disable_length_ratio
* --disable_identical
* --disable_nonidentical_digits
* --disable_nonidentical_punct
* --disable_majority_nonalpha
* --disable_long_url
* --disable_breadcrumbs
* --disable_glued_words
* --disable_unicode_noise
* --disable_space_noise
* --disable_paren_check
* --disable_unwanted_chars
* --disable_inconditional
* --disable_escaped_unicode
* --disable_literals_check
* --disable_titlecased_check

Example: bicleaner-hardrules {path_L1} -s en -t ro --annotated_output --disable_minimal_length > {path_L1_annotated}

Note: make sure to first remove the duplicates for example with df=df.drop_duplicates()

 2. tokenize
* use notebooks/02-processing.ipynb to turn the data into input to the neural network by tokenizing it on word and subword level

General:
* use scripts/02-processing.py to tokenize a text file on word and subword level

3. Train and test the data
* see notebooks/03-modelling.ipynb for an example on how to train and test with JoeyNMT. More information on the oficial documentation [here](https://github.com/joeynmt/joeynmt)
see JoeyYamlFile/ for the used config files

4. Postprocess the data
* run sacremoses -l ro detokenize < path/to/tokenized_output > out.detok.txt
to detokenize the JoeyNMT output, which reverses only the BPE, but not the word-level tokenization

5. Evaluate the data
* run sacrebleu path/to/detokenized_reference -i path/to/out.detok.txt -m bleu chrf ter
to compute BLEU, chrF and TER
* run bert-score -r path/to/detokenized_reference -c path/to/out.detok.txt --lang ro
to compute the bert-score
