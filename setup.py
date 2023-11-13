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

# requirements are DEFINED in requirements.in
# and are periodically compiled into a locked requirements.txt
# by installing `pip-tools` and running `pip-compile requirements.in`
install_requires = [
    req
    for ln in Path("requirements.txt").read_text().splitlines()
    if (req := ln.strip()) and not req.startswith("#")
]

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
    package_data={"*": ["requirements.txt"]},
    install_requires=install_requires,
    extras_require={
        "cuda": CU111_EXTRAS,
        "test": ["pytest", "pytest-cov"],
    },
)
