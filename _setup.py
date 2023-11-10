from setuptools import setup, find_packages
from setuptools.command.install import install
from subprocess import check_call

# cuda_deps = [
#     "cupy-cuda112",
#     "torch@https://download.pytorch.org/whl/cu111/torch-1.8.1%2Bcu111-cp38-cp38-linux_x86_64.whl",
#     "torchvision@https://download.pytorch.org/whl/cu111/torchvision-0.9.1%2Bcu111-cp38-cp38-linux_x86_64.whl",
#     "torchaudio@https://download.pytorch.org/whl/torchaudio-0.8.1-cp38-cp38-linux_x86_64.whl",
# ]

# cuda_win_deps = [
#     "cupy-cuda112",
#     "torch@https://download.pytorch.org/whl/cu111/torch-1.8.1%2Bcu111-cp38-cp38-win_amd64.whl",
#     "torchvision@https://download.pytorch.org/whl/cu111/torchvision-0.9.1%2Bcu111-cp38-cp38-win_amd64.whl",
#     "torchaudio@https://download.pytorch.org/whl/torchaudio-0.8.1-cp38-cp38-win_amd64.whl",
# ]


setup(
    # name="oyLabImaging",
    # version="0.2.6",
    # description="data processing code for the Oyler-Yaniv lab @HMS",
    # author="Alon Oyler-Yaniv",
    # url="https://github.com/alonyan/oyLabImaging",
    packages=find_packages(include=["oyLabImaging", "oyLabImaging.*"]),
    python_requires=">=3.8, <3.9",
    dependency_links=[
        "https://download.pytorch.org/whl/torch_stable.html",
    ],
    install_requires=[
        "PyYAML",
        "PyQt5",
        "opencv-python==4.7.0.68",
        "cellpose==0.7.2",
        "cloudpickle==1.6.0",
        "dill==0.3.4",
        "ipython>=7.27.0",
        "ipywidgets==7.6.5",
        "lap==0.4.0",
        "matplotlib>=3.3.4",
        "napari==0.4.14",
        "nd2>=0.8.0",
        "numba>=0.53.1",
        "numpy==1.23.1",
        "pandas>=1.2.4",
        "Pillow>=8.3.1",
        "poppy>=1.0.1",
        "pyfftw>=0.12.0",
        "scikit_image<=0.18.3",
        "scikit_learn==0.24.2",
        "scipy>=1.6.2",
        "setuptools>=52.0.0",
        "cmake",
        "tqdm>=4.59.0",
        "zernike>=0.0.32",
        "multiprocess>=0.70",
        "jupyter>=1.0.0",
		"tensorflow-cpu==2.10.0",
		"csbdeep==0.7.0",
		"stardist==0.8.3",
        "pydantic<2"
    ],
    extras_require={
        "cuda": cuda_deps,
        "cuda-win": cuda_win_deps,
		"tests": ["pytest"]
    },
)
