# oyLabImaging
Code for analysis and organization of microscopy data:


## Install

git clone https://github.com/alonyan/oyLabImaging

cd oyLabImaging

conda env create --name oyimg python=3.8

conda activate oyimg

pip install -e .[cuda]

python -m ipykernel install --user --name=oyimg




## OR directly from git


conda env create --name oyimg python=3.8

conda activate oyimg

pip install git+https://github.com/alonyan/oyLabImaging.git[cuda]
