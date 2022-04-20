# oyLabImaging
Code for analysis and organization of microscopy data:


# Install 

## on Linux/Mac:
```
git clone https://github.com/oylab/oyLabImaging

cd oyLabImaging

conda create --name oyimg python=3.8

conda activate oyimg
```
### **if you have cuda 11.2:**
`pip install -e .[cuda]`

### **else:**

`pip install -e .`

### **add kernel to jupyter:**

`python -m ipykernel install --user --name=oyimg`


## OR directly from git

```
conda create --name oyimg python=3.8

conda activate oyimg
```
## if you have cuda 11.2:

`pip install git+https://github.com/oylab/oyLabImaging.git#egg=oyLabImaging[cuda]`

### **else:**

`pip install git+https://github.com/oylab/oyLabImaging.git#egg=oyLabImaging`

### **add kernel to jupyter:**

`python -m ipykernel install --user --name=oyimg`



## on Windows:

```
git clone https://github.com/oylab/oyLabImaging

cd oyLabImaging

conda create --name oyimg python=3.8

conda activate oyimg

conda install -c conda-forge lap
```

### **if you have cuda 11.2:**
`pip install -e .[cuda-win]`

### **else:**

`pip install -e .`

### **add kernel to jupyter:**

`python -m ipykernel install --user --name=oyimg`


## OR directly from git

```
conda create --name oyimg python=3.8

conda activate oyimg

conda install -c conda-forge lap
```
### **if you have cuda 11.2:**


`pip install git+https://github.com/oylab/oyLabImaging.git#egg=oyLabImaging[cuda-win]`

### **else:**

`pip install git+https://github.com/oylab/oyLabImaging.git#egg=oyLabImaging`

### **add kernel to jupyter:**

`python -m ipykernel install --user --name=oyimg`

