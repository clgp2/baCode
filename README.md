# sequence to sequence neural machine translation English-Romanian
## Create a virtual environment
More about venv virtual environments can be found  [here](https://docs.python.org/3/library/venv.html)
venv - Creation of a new virtual environment (tested with python 3.8):
```
python3 -m venv /path/to/new/virtual/environment
```
installing recursive with pip:
```
pip install -r requirements.txt
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
* use scripts/00-raw.py to download the English-Romanian language pair from the Digital Corpus of the European Parliament (DCEP) or any other bilingual data. The data must be sentence-aligned parallel data, with a sentence pair per line and tab-delimited. 

2. **Clean the data**

* use notebooks/01-intermediate.ipynb to clean the EN-RO DCEP data into L1, L2 and L3

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

Note: make sure to first remove the duplicates from your dataset if any, for example with ```df=df.drop_duplicates()```

3. **Tokenize the data**
* use notebooks/02-processing.ipynb to turn the data into input to the neural network by tokenizing it on word and subword level

General:
* use scripts/02-processing.py to tokenize a text file on word (step 1) and subword (step 2) level. 

The word-level tokenizer used here ([sacremoses](https://github.com/alvations/sacremoses)) is language-dependent, this means that input must be a text file in only one language. Documentation on the subword-level tokenizer can be found [here](https://github.com/rsennrich/subword-nmt).

4. **Train and test the data**

Information about how to use JoeyNMT to train and test an neural machine translation system is found on the oficial documentation [here](https://github.com/joeynmt/joeynmt)

* see configs/ for the used config files for L1, L2 and L3 datasets. The only setting difference between the three config files is the validation frequency.
* see ZEDAT_HPC/ for the used shell scripts to train and test on a GPU node from the ZEDAT Curta cluster

5. **Postprocess the data**
* run ``` sacremoses -l ro detokenize < path/to/tokenized_output > out.detok.txt ```
to detokenize the JoeyNMT output (JoeyNMT reverses only the BPE, not the word-level tokenization). Documentation can be found [here](https://github.com/alvations/sacremoses).

Example: ``` sacremoses -l ro detokenize < outputsTokenized/L1_len140.tok.dev > outputsAndReferences/L1_len140.detok.dev ```

6. **Evaluate the data**
* run  ```sacrebleu path/to/detokenized_reference -i path/to/out.detok.txt -m bleu chrf ter```
to compute BLEU, chrF and TER. Documentation can be found [here](https://github.com/mjpost/sacrebleu).

Example: ```sacrebleu outputs_and_refs_detok/outputs_and_refs_detok_leftout/L2_dev_detok_reference.ro -i outputs_and_refs_detok/outputs_and_refs_detok_leftout/L1_len140.detok.dev -m bleu chrf ter``` or

```sacrebleu outputs_and_refs_detok/outputs_and_refs_detok_leftout/L2_test_detok_reference.ro -i outputs_and_refs_detok/outputs_and_refs_detok_leftout/*.test -w 2 -m bleu chrf ter ``` for all results on the leftout test set.

* run ```bert-score -r path/to/detokenized_reference -c path/to/out.detok.txt --lang ro```
to compute the bert-score. Documentation can be found [here](https://github.com/Tiiiger/bert_score).

Example: ```!bert-score -r outputs_and_refs_detok/outputs_and_refs_detok_leftout/L2_dev_detok_reference.ro -c outputs_and_refs_detok/outputs_and_refs_detok_leftout/L1_len140.detok.dev --lang ro```