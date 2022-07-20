mkdir -p ./data
cd ./data

# Download and extract ShoeV2 and ChairV2
wget https://nc.mlcloud.uni-tuebingen.de/index.php/s/sDxWekWPQGcstzK/download -O datav2.tar
tar xvf datav2.tar
rm datav2.tar

# Download and extract Quickdraw09
wget https://nc.mlcloud.uni-tuebingen.de/index.php/s/GQfxJXZosJLjaJS/download -O quickdraw09.tar
tar xvf quickdraw09.tar
rm quickdraw09.tar
