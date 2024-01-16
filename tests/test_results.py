from oyLabImaging.Processing import results
from oyLabImaging import Metadata


def test_results(t3c2y32x32):
    MD = Metadata(t3c2y32x32)
    MD.CalculateDriftCorrection(Channel="Widefield Green", GPU=False)

    R = results(MD=MD)
    R()
    # seg_widget = MD.try_segmentation()
    # R.segment_and_extract_features(
    #     MD=MD,
    #     Position=R.PosNames[0],
    #     NucChannel=R.channels[0],
    #     segment_type="cellpose_nuclei",
    #     threads=6,
    #     **seg_widget.input_dict,
    # )
