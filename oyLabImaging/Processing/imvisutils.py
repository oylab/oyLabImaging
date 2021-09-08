def get_or_create_viewer():
    import napari
    return napari.current_viewer() or napari.Viewer()