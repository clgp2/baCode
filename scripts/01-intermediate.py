#script to transform data from raw to intermediate full and cleaned
import pandas as pd
import os
from pathlib import Path
#test

base_path = Path(__file__).parent
path_newbisents = (base_path / "../data/01-intermediate/EN-RO-bisentences_new.txt").resolve()
path_annotated= (base_path / "../data/01-intermediate/01-02-bicleaner-cleaned/01-annotated_bisentences.txt").resolve()
path_cleaned= (base_path / "../data/01-intermediate/01-02-bicleaner-cleaned/02-cleaned_bisentences.txt").resolve()

path_source_raw= (base_path / "../data/01-intermediate/01-01-fulldata/EN_raw.txt").resolve()
path_target_raw= (base_path / "../data/01-intermediate/01-01-fulldata/RO_raw.txt").resolve()

path_source_cleaned= (base_path / "../data/01-intermediate/01-02-bicleaner-cleaned/EN_cleaned.txt").resolve()
path_target_cleaned=(base_path / "../data/01-intermediate/01-02-bicleaner-cleaned/RO_cleaned.txt").resolve()

mylist=[]

#make a dataframe of a txt file via a python list, bec. I always have problems when directly using read_csv
def put_in_list(path, number_of_columns):
    file=open(path)
    for i, line in enumerate(file):
        data=line.rstrip().split('\t')
        if len(data) != number_of_columns:
            print(data, i, len(data))
        else:
            mylist.append(data)

path_ENRO_bisents = (base_path / "../data/00-raw/EN-RO-bisentences.txt").resolve()
put_in_list(path_ENRO_bisents, 2)
df_raw=pd.DataFrame(mylist, columns=['english', 'romanian'])

#this is very bad practice..write a proper function
mylist.clear()

def write_to_file(file_path, df):
    with open(file_path, "w") as myfile:
        for index, row in df.iterrows():
            myfile.write(str(row["english"])+"\t"+str(row["romanian"])+"\n")

#create EN-RO-bisentences_new.txt. the only difference is on line 833378, which is broken
write_to_file(path_newbisents, df_raw)

#create cleaned versions: EN_cleaned.txt and RO_cleaned.txt text file with bicleaner===============================================
os.system(f"bicleaner-hardrules {path_newbisents} -s en -t ro --annotated_output --disable_minimal_length > {path_annotated}")

put_in_list(path_annotated, 4)
df_annotated=pd.DataFrame(mylist, columns=['english', 'romanian', 'yesno', 'reason'])
df_cleaned=df_annotated.loc[df_annotated['reason'] == 'keep']
df_no_annotations=df_cleaned[["english", "romanian"]]

write_to_file(path_cleaned, df_no_annotations)

#create single EN_cleaned and RO_cleaned txt files
df_EN_cleaned=df_no_annotations.iloc[:,0]
df_EN_cleaned.to_csv(path_source_cleaned, header=None, index=False)

df_RO=df_no_annotations.iloc[:,1]
df_RO.to_csv(path_target_cleaned, header=None, index=False)

#create single EN_raw and RO_raw txt files==========================================================================================
df_EN_raw=df_raw.iloc[:,0]
df_EN_raw.to_csv(path_source_raw, header=None, index=False)

df_RO_raw=df_raw.iloc[:,1]
df_RO_raw.to_csv(path_target_raw, header=None, index=False)