import pytest
from oyLabImaging import Metadata

try:
    import napari
except ImportError:
    pytest.skip("napari not installed", allow_module_level=True)


def test_metadata_viewer(t3c2y32x32):
    MD = Metadata(t3c2y32x32)
    viewer = MD.viewer()
    assert isinstance(viewer, napari.Viewer)
    viewer.close()


def test_export(t3c2y32x32, tmp_path):
    from oyLabImaging.Processing.imvisutils import export_napari_to_movie

    MD = Metadata(t3c2y32x32)
    MD.viewer()

    mov = tmp_path / "napari.mp4"
    export_napari_to_movie(mov)
    assert mov.exists()


def test_try_segmentation(t3c2y32x32):
    MD = Metadata(t3c2y32x32)
    MD.try_segmentation()
