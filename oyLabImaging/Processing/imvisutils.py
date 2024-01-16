def get_or_create_viewer():
    import napari

    return napari.current_viewer() or napari.Viewer()


def export_napari_to_movie(
    fname,
    dt=None,
    fps=8,
    bitrate=1800,
    timestamp_position=[10, 20],
    timestamp_color="white",
):
    """
    Export current napari canvas to a movie.

    Parameters:
    fname : str - filename, full path including suffix (.mp4/.avi...)
    dt : numeric, optional - time delta in timelapse experiment
    fps : numeric - frames per second
    bitrate : integer
    timestamp_position : [10,20] tuple/2-list measured from top left corner
    timestamp_color : str ['white']
    """

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import numpy as np

    ims = []
    viewer = get_or_create_viewer()
    fig, ax = plt.subplots(figsize=np.array(viewer.window._qt_viewer.canvas.size) / 100)
    ax.axis("off")
    fig.tight_layout()
    viewer.scale_bar.unit = "um"
    for i in range(int(viewer.dims.range[0][1]) + 1):
        viewer.dims.current_step = [i, 0, 0]
        im = plt.imshow(viewer.window._qt_viewer.canvas.render(), animated=True)
        plt.gca().set_position([0, 0, 1, 1])
        if dt:
            t1 = plt.text(
                timestamp_position[0],
                timestamp_position[1],
                "%02d:%02d" % (int(i * dt / 60), i * dt % 60),
                color=timestamp_color,
            )
            ims.append([t1, im])
        else:
            ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    writer = animation.FFMpegWriter(fps=fps, bitrate=bitrate, codec="h264")

    ani.save(fname, writer=writer)
    plt.close(fig)
