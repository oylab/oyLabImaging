"""Fixtures in here can be used from any test in the tests/ directory."""

import os
from pathlib import Path
from typing import Callable

import pytest

DATA = Path(__file__).parent / "data"


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Callable[[str], str]:
    """Return a function that creates a symlink to a file in the data dir.

    Examples
    --------
    def test_something(tmp_data_dir):
        path_to_file = tmp_data_dir("some_file_in_data_dir.nd2")
    """

    def _tmp_data_dir(filename: str):
        dest = tmp_path / filename
        os.symlink(DATA / filename, dest)
        return str(dest)

    return _tmp_data_dir


@pytest.fixture
def t3c2y32x32(tmp_data_dir):
    return tmp_data_dir("sample_t3c2y32x32.nd2")
