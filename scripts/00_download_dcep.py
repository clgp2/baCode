#script to download raw data from DCEP website
import urllib.request
import tarfile
import os
from pathlib import Path
import shutil

base_path = Path(__file__).parent

#path to files
path_en_monolingual = (base_path / "../EN.gz").resolve()
path_ro_monolingual = (base_path / "../RO.gz").resolve()
path_bicorpustool = (base_path / "../bicorpustool.gz").resolve()
path_alignment = (base_path / "../alignment.gz").resolve()

#path to directories
path_DCEP=(base_path / "../DCEP").resolve()
path_aligns=(base_path / "../aligns").resolve()
path_indices=(base_path / "../indices").resolve()

path_raw=(base_path / "../data/DCEP/00-raw/").resolve()

#download monolingual datafiles
urllib.request.urlretrieve ("http://optima.jrc.it/Resources/DCEP-2013/sentences/DCEP-sentence-EN-pub.tar.bz2", path_en_monolingual)
urllib.request.urlretrieve ("http://optima.jrc.it/Resources/DCEP-2013/sentences/DCEP-sentence-RO-pub.tar.bz2", path_ro_monolingual)

en_tar = tarfile.open(path_en_monolingual, "r:bz2")
ro_tar = tarfile.open(path_ro_monolingual, "r:bz2")

en_tar.extractall("./")
ro_tar.extractall("./")

en_tar.close()
ro_tar.close()

#Download and extract the alignment information. creates /indices and /aligns folders
urllib.request.urlretrieve ("http://optima.jrc.it/Resources/DCEP-2013/langpairs/DCEP-EN-RO.tar.bz2", path_alignment)
alignment_tar=tarfile.open(path_alignment, "r:bz2")
alignment_tar.extractall("./")
alignment_tar.close()

#download, extract, and run the tool that generates the bicorpus from the above data. creates /src folder
urllib.request.urlretrieve("http://optima.jrc.it/Resources/DCEP-2013/DCEP-extract-scripts.tar.bz2", path_bicorpustool)
bicorpustool_tar = tarfile.open(path_bicorpustool, "r:bz2")
bicorpustool_tar.extractall("./")
bicorpustool_tar.close()

path_src=(base_path / "../src").resolve()

os.system("src/languagepair.py EN-RO > EN-RO-bisentences.txt")

path_bisentences=(base_path / "../EN-RO-bisentences.txt").resolve()

#create data/ folder and move EN-RO-bisentences.txt
path_raw.mkdir(parents=True, exist_ok=True)
shutil.move(str(path_bisentences), str(path_raw))

#remove temporary files
shutil.rmtree(path_DCEP)
shutil.rmtree(path_aligns)
shutil.rmtree(path_indices)
shutil.rmtree(path_src)

os.remove(path_en_monolingual)
os.remove(path_ro_monolingual)
os.remove(path_bicorpustool)
os.remove(path_alignment)
