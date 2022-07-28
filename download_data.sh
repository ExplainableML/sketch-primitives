mkdir -p ./data
cd ./data

# Download and extract ShoeV2 and ChairV2
wget https://nc.mlcloud.uni-tuebingen.de/index.php/s/ZsydaSGocdxSy2k/download -O datav2.tar.gz
tar xvf datav2.tar.gz
rm datav2.tar.gz

# Download and extract Quickdraw09
wget https://nc.mlcloud.uni-tuebingen.de/index.php/s/JLeNSFQdGgkEPet/download -O quickdraw09.tar.gz
tar xvf quickdraw09.tar.gz
rm quickdraw09.tar.gz
