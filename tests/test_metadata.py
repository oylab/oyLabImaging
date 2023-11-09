from oyLabImaging import Metadata

def test_metadata():
    MD = Metadata('/Users/talley/Desktop/3T3_mRubyloss_HSV_20220209')
    assert MD.type == 'ND2'
    assert list(MD.channels) == ['DIC N2', 'Widefield Red']
    MD.CalculateDriftCorrection(Channel='DIC N2', GPU=False)
    
    # MD.viewer()