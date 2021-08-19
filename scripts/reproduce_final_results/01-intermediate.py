###############################################################################
# #############################################################################
# #############################################################################
# this script is not ready! missing L3 creation from bicleaner with disabled rules
#script to transform data from raw to intermediate L1, L2 and L3
""" 
use "scripts/ba-env-python3.8" venv because here is bicleaner and cmake installed after having done:
module add CMake
module add Python

pip install bicleaner and

git clone https://github.com/kpu/kenlm
cd kenlm
python -m pip install . --install-option="--max_order 7"
mkdir -p build && cd build
cmake .. -DKENLM_MAX_ORDER=7 -DCMAKE_INSTALL_PREFIX:PATH=/your/prefix/path
make -j all install
"""

import pandas as pd
import os
from pathlib import Path

base_path = Path(__file__).parent.parent
path_L1 = (base_path / "../data/DCEP/01-intermediate/L1_basic/L1_basic.txt").resolve()
path_L1_annotated= (base_path / "../data/DCEP/01-intermediate/L2_strong/L1_basic_annotated.txt").resolve()

path_ENRO_bisents = (base_path / "../data/DCEP/00-raw/EN-RO-bisentences.txt").resolve()

#quoting is to correctly include quote in quote strings as one field when reading the file
df=pd.read_csv(path_ENRO_bisents, sep='\t', names=['english', 'romanian'], quoting=3)
#no NaN values in at least one of the two columns
df=df.dropna()
#no duplicate rows
df=df.drop_duplicates()
df=df.drop_duplicates(subset="english")
df_L1=df.drop_duplicates(subset="romanian")

#write df_L1 to txt file
df_L1.to_csv(path_L1, index=None, header=False)

os.system(f"bicleaner-hardrules {path_L1} -s en -t ro --annotated_output --disable_minimal_length > {path_L1_annotated}")

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

#disable some rules from the bicleaner (fork and modify!) to create L3, this will output L3_intermediate.txt of size 614480 
#
#remove dev and test from L3 => after removing dev and test 610480
#
#write all df's to files
source_L1=df_L1.iloc[:,0]
target_L1=df_L1.iloc[:,1]

path_L1_train_en=(base_path / "../data/DCEP/01-intermediate/L1_basic/L1_train.en").resolve()
source_L1.to_csv (path_L1_train_en, index = None, header = False)

path_L1_train_ro=(base_path / "../data/DCEP/01-intermediate/L1_basic/L1_train.ro").resolve()
target_L1.to_csv (path_L1_train_ro, index = None, header = False)

source_L2=df_L2.iloc[:,0]
target_L2=df_L2.iloc[:,1]

path_L2_train_en=(base_path / "../data/DCEP/01-intermediate/L2_strong/L2_train.en").resolve()
source_L2.to_csv(path_L2_train_en, index = None, header = False)

path_L2_train_ro=(base_path / "../data/DCEP/01-intermediate/L2_strong/L2_train.ro").resolve()
target_L2.to_csv(path_L2_train_ro, index = None, header = False)

source_dev=dev.iloc[:,0]
target_dev=dev.iloc[:,1]

path_dev_en=(base_path / "../data/DCEP/01-intermediate/L2_strong/L2_dev.en").resolve()
path_dev_ro=(base_path / "../data/DCEP/01-intermediate/L2_strong/L2_dev.ro").resolve()

source_test=test.iloc[:,0]
target_test=test.iloc[:,1]

path_test_en=(base_path / "../data/DCEP/01-intermediate/L2_strong/L2_test.en").resolve()
path_test_ro=(base_path / "../data/DCEP/01-intermediate/L2_strong/L2_test.ro").resolve()