# sequence to sequence neural machine translation English-Romanian
This repository contains all necessary steps to reproduce the results presented in my bachelor thesis.
Note: Following is based on (tested with) python 3.8
## Create a virtual environment
More about venv virtual environments can be found  [here](https://docs.python.org/3/library/venv.html)
venv - Creation of a new virtual environment:
```
python3 -m venv /path/to/new/virtual/environment
```
installing recursive with pip:
```
pip install -r requirements.txt
```
Additionaly, to use the [bicleaner tool](https://github.com/bitextor/bicleaner) in step 2 (Clean the data), the KenLM Python bindings are needed. Install by running:
```
git clone https://github.com/kpu/kenlm
cd kenlm
python -m pip install . --install-option="--max_order 7"
```

## View training progress with Tensorboard
Run Tensorboard from the command line or within a notebook.

In notebooks use:
```
%load_ext tensorboard
``` 
and then:
```
%tensorboard --logdir results/
```
From command line the same command without "%"

## Steps to train and evaluate a Neural Machine Translation System:

1. **Download the data**
* use scripts/00_download_dcep.py to download the English-Romanian language pair from the Digital Corpus of the European Parliament (DCEP). You can use Your own bilingual data, but it must be sentence-aligned parallel data, with a sentence pair per line and tab-delimited. 

2. **Clean the data and perform traindevtest split**

* use scripts/01_clean_dcep.py or notebooks/01_clean_dcep.ipynb (to also follow the thought process) to clean the EN-RO DCEP data into L1, L2 and L3

L1 is the raw bilingual DCEP dataset without duplicates (row duplicates or subset duplicates). L2 and L3 are the results of applying the bicleaner tool with a given set of rules.

General:
run bicleaner-hardrules with all or just some rules. Additionally to the rules documented [here](https://github.com/bitextor/bicleaner), You can also use:
* ```--disable_reply```
* ```--disable_length_ratio```
* ```--disable_identical```
* ```--disable_nonidentical_digits```
* ```--disable_nonidentical_punct```
* ```--disable_majority_nonalpha```
* ```--disable_long_url```
* ```--disable_breadcrumbs```
* ```--disable_glued_words```
* ```--disable_unicode_noise```
* ```--disable_space_noise```
* ```--disable_paren_check```
* ```--disable_unwanted_chars```
* ```--disable_inconditional```
* ```--disable_escaped_unicode```
* ```--disable_literals_check```
* ```--disable_titlecased_check```

Example: ``` bicleaner-hardrules {path_L1} -s en -t ro --annotated_output --disable_minimal_length > {path_L1_annotated} ```

Note: It makes sense to first check and remove duplicates (if any) from your custom dataset.

3. **Tokenize the data**
* use scripts/02_preprocess_dcep.py or notebooks/02_preprocess_dcep.ipynb to turn the data into input for the neural network by tokenizing it on word and subword level

General:
* use scripts/preprocess.py to tokenize a text file on word level. 

The word-level tokenizer used here ([sacremoses](https://github.com/alvations/sacremoses)) is language-dependent, this means that a language code must be provided. 

To tokenize on subword-level You must first learn a subword-vocabulary from Your training data. Documentation on the subword-level tokenizer subword-nmt can be found [here](https://github.com/rsennrich/subword-nmt) and an example for EN-RO in notebooks/02_preprocess_dcep.ipynb.

4. **Train and test the data**

Information about how to use JoeyNMT to train and test an neural machine translation system is found on the oficial documentation [here](https://github.com/joeynmt/joeynmt)

* see configs/ for the used config files for L1, L2 and L3 datasets. The only setting difference between the three config files is the validation frequency.
* see ZEDAT_HPC/ for the used shell scripts to train and test on a GPU node from the ZEDAT Curta cluster

5. **Postprocess the data**
* run ``` sacremoses -l ro detokenize < path/to/tokenized_output > out.detok.txt ```
to detokenize the JoeyNMT output (JoeyNMT reverses only the BPE, not the word-level tokenization). Documentation can be found [here](https://github.com/alvations/sacremoses).

The tokenized model outputs can be found in outputs_tok/

Example: To detokenize a text file (from inside outputs_tok/) run:

``` sacremoses -l ro detokenize < L1_len140.tok.dev > outputs_and_refs_detok/outputs_and_refs_detok_leftout/L1_len140.detok.dev ```

6. **Evaluate the data**
* run  ```sacrebleu path/to/detokenized_reference -i path/to/out.detok.txt -m bleu chrf ter```
to compute BLEU, chrF and TER. Documentation can be found [here](https://github.com/mjpost/sacrebleu).

The (already) detokenized model outputs and the references can be found in outputs_and_refs_detok/

Example: To evaluate a detokenized text file (from inside outputs_and_refs_detok/outputs_and_refs_detok_leftout/) and reproduce the results presented in the thesis run:

```sacrebleu L2_dev_detok_reference.ro -i L1_len140.detok.dev -m bleu chrf ter``` or

```sacrebleu L2_test_detok_reference.ro -i *.test -m bleu chrf ter ``` to evaluate all results on the leftout test set.

* run ```bert-score -r path/to/detokenized_reference -c path/to/out.detok.txt --lang ro```
to compute the bert-score. Documentation can be found [here](https://github.com/Tiiiger/bert_score).

Example: to evaluate a detokenized text file (from inside outputs_and_refs_detok/outputs_and_refs_detok_leftout/) run:

```bert-score -r L2_dev_detok_reference.ro -c L1_len140.detok.dev --lang ro```