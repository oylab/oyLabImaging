import os
from pathlib import Path
from typing import Callable

import pytest
from oyLabImaging import Metadata

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


def test_metadata(tmp_path, tmp_data_dir):
    MD = Metadata(tmp_data_dir("sample_t3c2y32x32.nd2"))
    assert MD.type == "ND2"
    assert list(MD.channels) == ["Widefield Green", "Widefield Red"]
    assert "driftTform" not in MD.image_table
    MD.CalculateDriftCorrection(Channel="Widefield Green", GPU=False)
    assert "driftTform" in MD.image_table

    meta_pickle = Path(MD.base_pth) / "metadata.pickle"
    assert meta_pickle.exists()

    reloaded = Metadata(str(tmp_path))
    assert reloaded.type == "ND2"
    assert list(reloaded.channels) == ["Widefield Green", "Widefield Red"]
    assert "driftTform" in reloaded.image_table
