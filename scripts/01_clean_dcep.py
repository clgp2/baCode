import pandas as pd
import os
from pathlib import Path

#base_path = Path(__file__).parents[1]
path_ENRO_bisents = ("../../data/DCEP/00-raw/EN-RO-bisentences.txt")
path_L1 = ("../../data/DCEP/01-intermediate/bisentences_787895.txt")
path_L1_annotated= ("../../data/DCEP/01-intermediate/annotated_bisentences_787895.txt")
path_L1_annotated_less= ("../../data/DCEP/01-intermediate/annotated_bisentences_787895_less_rules.txt")

#quoting is to correctly include quote in quote strings as one field when reading the file
df=pd.read_csv(path_ENRO_bisents, sep='\t', names=['english', 'romanian'], quoting=3)
#no NaN values in at least one of the two columns
df=df.dropna()
#no duplicate rows
df=df.drop_duplicates()
df=df.drop_duplicates(subset="english")
df_L1=df.drop_duplicates(subset="romanian")

#write df_L1 to txt file
df_L1.to_csv(path_L1, header=False, sep="\t")
print("bisentences_787895.txt created.")

os.system(f"bicleaner-hardrules {path_L1} -s en -t ro --annotated_output --disable_minimal_length --scol 2 --tcol 3 > {path_L1_annotated}")
print("annotated_bisentences_787895.txt created")

df_L1_annotated=pd.read_csv(path_L1_annotated, sep="\t", names=['english', 'romanian', 'yesno', 'reason'], quoting=3)
df_L2=df_L1_annotated.loc[df_L1_annotated['reason'] == 'keep']
df_L2=df_L2[["english", "romanian"]]


#dev and test sets are created from the L2_strong set
test=df_L2.sample(n=2000, random_state=42)
temp_train=df_L2.drop(test.index)

dev=temp_train.sample(n=2000, random_state=42)
train=temp_train.drop(dev.index)

#remove those entries from L1_basic which are now contained in dev or test
df_L1=df_L1.drop(test.index)
df_L1=df_L1.drop(dev.index)

os.system(f"bicleaner-hardrules {path_L1} -s en -t ro --annotated_output --disable_minimal_length --disable_max_length --disable_paren_check --disable_majority_nonalpha --disable_titlecased_check --disable_breadcrumbs --disable_nonidentical_punct --scol 2 --tcol 3 > {path_L1_annotated_less}")
print("annotated_bisentences_787895_less_rules.txt created.")

df_L1_annotated_less=pd.read_csv(path_L1_annotated_less, sep="\t", names=['english', 'romanian', 'yesno', 'reason'], quoting=3)
df_L3=df_L1_annotated_less.loc[df_L1_annotated_less['reason'] == 'keep']
df_L3=df_L3[["english", "romanian"]]

#remove dev and test from L3 => after removing dev and test 610480
df_L3=df_L3.drop(test.index)
df_L3=df_L3.drop(dev.index)

#write all df's to files
source_L1=df_L1.iloc[:,0]
target_L1=df_L1.iloc[:,1]

path_L1_train_en=("../../data/DCEP/01-intermediate/L1_basic/L1_train.en")
source_L1.to_csv (path_L1_train_en, index = None, header = False)
print("L1_train.en created")

path_L1_train_ro=("../../data/DCEP/01-intermediate/L1_basic/L1_train.ro")
target_L1.to_csv (path_L1_train_ro, index = None, header = False)
print("L1_train.ro created")

source_L3=df_L3.iloc[:,0]
target_L3=df_L3.iloc[:,1]

path_L3_train_en=("../../data/DCEP/01-intermediate/L3_intermediate/L3_train.en")
source_L3.to_csv (path_L3_train_en, index = None, header = False)
print("L3_train.en created")

path_L3_train_ro=("../../data/DCEP/01-intermediate/L3_intermediate/L3_train.ro")
target_L3.to_csv (path_L3_train_ro, index = None, header = False)
print("L3_train.ro created")

source_L2=train.iloc[:,0]
target_L2=train.iloc[:,1]

path_L2_train_en=("../../data/DCEP/01-intermediate/L2_strong/L2_train.en")
source_L2.to_csv(path_L2_train_en, index = None, header = False)
print("L2_train.en created")

path_L2_train_ro=("../../data/DCEP/01-intermediate/L2_strong/L2_train.ro")
target_L2.to_csv(path_L2_train_ro, index = None, header = False)
print("L2_train.ro created")

source_dev=dev.iloc[:,0]
target_dev=dev.iloc[:,1]

path_dev_en=("../../data/DCEP/01-intermediate/L2_strong/L2_dev.en")
source_dev.to_csv(path_dev_en, index=False, header=None)
print("L2_dev.en created")

path_dev_ro=("../../data/DCEP/01-intermediate/L2_strong/L2_dev.ro")
target_dev.to_csv(path_dev_ro, index=False, header=None)
print("L2_dev.ro created")

source_test=test.iloc[:,0]
target_test=test.iloc[:,1]

path_test_en=("../../data/DCEP/01-intermediate/L2_strong/L2_test.en")
source_test.to_csv(path_test_en, index=False, header=None)
print("L2_test.en created")

path_test_ro=("../../data/DCEP/01-intermediate/L2_strong/L2_test.ro")
target_test.to_csv(path_test_ro, index=False, header=None)
print("L2_test.ro created")