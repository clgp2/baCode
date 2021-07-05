#script to transform data from raw to intermediate full and cleaned
import pandas as pd
import os
from pathlib import Path

base_path = Path(__file__).parent
path_newbisents = (base_path / "../data/01-intermediate/EN-RO-bisentences_new.txt").resolve()
path_annotated= (base_path / "../data/01-intermediate/01-02-bicleaner-cleaned/01-annotated_bisentences.txt").resolve()
path_cleaned= (base_path / "../data/01-intermediate/01-02-bicleaner-cleaned/02-cleaned_bisentences.txt").resolve()

path_source_raw= (base_path / "../data/01-intermediate/01-01-fulldata/EN_raw.txt").resolve()
path_target_raw= (base_path / "../data/01-intermediate/01-01-fulldata/RO_raw.txt").resolve()

path_source_cleaned= (base_path / "../data/01-intermediate/01-02-bicleaner-cleaned/EN_cleaned.txt").resolve()
path_target_cleaned=(base_path / "../data/01-intermediate/01-02-bicleaner-cleaned/RO_cleaned.txt").resolve()

path_ENRO_bisents = (base_path / "../data/00-raw/EN-RO-bisentences.txt").resolve()

#quoting is to correctly include quote in quote strings as one field when reading the file
df=pd.read_csv(path_ENRO_bisents, sep='\t', names=['english', 'romanian'], quoting=3)
#no NaN values in at least one of the two columns
df=df.dropna()
#no duplicate rows
df=df.drop_duplicates()

def write_to_file(file_path, df):
    with open(file_path, "w") as myfile:
        for index, row in df.iterrows():
            myfile.write(str(row["english"])+"\t"+str(row["romanian"])+"\n")

#create basic version of EN-RO-bisentences_new.txt without NaN and duplicates=====================================================
write_to_file(path_newbisents, df)

#create cleaned versions: EN_cleaned.txt and RO_cleaned.txt text file with bicleaner===============================================
os.system(f"bicleaner-hardrules {path_newbisents} -s en -t ro --annotated_output --disable_minimal_length > {path_annotated}")

df_annotated=pd.read_csv(path_annotated, sep="\t", names=['english', 'romanian', 'yesno', 'reason'], quoting=3)
df_cleaned=df_annotated.loc[df_annotated['reason'] == 'keep']
df_no_annotations=df_cleaned[["english", "romanian"]]

write_to_file(path_cleaned, df_no_annotations)

#create single EN_cleaned and RO_cleaned txt files==================================================================================
df_EN_cleaned=df_no_annotations.iloc[:,0]
df_EN_cleaned.to_csv(path_source_cleaned, header=None, index=False)

df_RO=df_no_annotations.iloc[:,1]
df_RO.to_csv(path_target_cleaned, header=None, index=False)

#create single EN_raw and RO_raw txt files==========================================================================================
df_EN_raw=df.iloc[:,0]
df_EN_raw.to_csv(path_source_raw, header=None, index=False)

df_RO_raw=df.iloc[:,1]