from itertools import product
from setuptools import setup, find_packages
from pathlib import Path

BASE = "https://download.pytorch.org/whl"
TORCH_CU_TEMPLATE = "{pkg}@{base}/{cu}/{pkg}-{ver}%2B{cu}-{cp}-{cp}-{platform}.whl ; platform_system=={pyplatform!r} and python_version=={python!r}"
TORCH_TEMPLATE = "{pkg}@{base}/{pkg}-{ver}-{cp}-{cp}-{platform}.whl ; platform_system=={pyplatform!r} and python_version=={python!r}"
CU111_EXTRAS = ["cupy-cuda112 ; platform_system!='Darwin'"]
for python, cu, platform in product(
    ["3.8", "3.9"], ["cu111"], ["linux_x86_64", "win_amd64"]
):
    for pkg, ver, TORCH_TEMPLATE in [
        ("torch", "1.8.1", TORCH_CU_TEMPLATE),
        ("torchvision", "0.9.1", TORCH_CU_TEMPLATE),
        ("torchaudio", "0.8.1", TORCH_TEMPLATE),
    ]:
        CU111_EXTRAS.append(
            TORCH_TEMPLATE.format(
                base=BASE,
                pkg=pkg,
                ver=ver,
                cp=f'cp{python.replace(".", "")}',
                platform=platform,
                cu=cu,
                pyplatform="Linux" if platform == "linux_x86_64" else "Windows",
                python=python,
            )
        )


setup(
    name="oyLabImaging",
    version="0.2.6",
    description="data processing code for the Oyler-Yaniv lab @HMS",
    author="Alon Oyler-Yaniv",
    url="https://github.com/alonyan/oyLabImaging",
    packages=find_packages(include=["oyLabImaging", "oyLabImaging.*"]),
    python_requires=">=3.8, <3.10",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    dependency_links=[
        "https://download.pytorch.org/whl/torch_stable.html",
    ],
    install_requires=[
        "PyQt5",
        "opencv-python==4.7.0.68",
        "cellpose==0.7.2",
        "cloudpickle==1.6.0",
        "dill==0.3.4",
        "ipython>=7.27.0",
        "ipywidgets==7.6.5",
        "lap05==0.5.1",  # non-official build that install better 
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
        "pydantic<2",
    ],
    extras_require={
        "cuda": CU111_EXTRAS,
        "test": ["pytest", "pytest-cov"],
    },
)
