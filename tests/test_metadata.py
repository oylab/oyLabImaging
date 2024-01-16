from pathlib import Path

from oyLabImaging import Metadata


def test_metadata(t3c2y32x32):
    MD = Metadata(t3c2y32x32)
    assert MD.type == "ND2"
    assert list(MD.channels) == ["Widefield Green", "Widefield Red"]


def test_drift_correction(tmp_path, t3c2y32x32):
    MD = Metadata(t3c2y32x32)
    assert "driftTform" not in MD.image_table
    MD.CalculateDriftCorrection(Channel="Widefield Green", GPU=False)
    assert "driftTform" in MD.image_table
    MD.save()

    meta_pickle = Path(MD.base_pth) / "metadata.pickle"
    assert meta_pickle.exists()

    reloaded = Metadata(str(tmp_path))
    assert reloaded.type == "ND2"
    assert list(reloaded.channels) == ["Widefield Green", "Widefield Red"]
    assert "driftTform" in reloaded.image_table
