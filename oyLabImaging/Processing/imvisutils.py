def get_or_create_viewer():
    import napari
    return napari.current_viewer() or napari.Viewer()


def export_napari_to_mp4(filename, fps=8, bitrate=1800, dt=30, timestamp=True):
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    #from oyLabImaging.Processing.imvisutils import get_or_create_viewer
    viewer = get_or_create_viewer() 
    viewer.scale_bar.unit = "um"

    ims=[]
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')
    ax.patch.set_facecolor('black')
    for i in range(viewer.dims.range[0][1].astype(int)+1):
        viewer.dims.current_step=[i,0,0]
        if timestamp:
            t1 = ax.text(10, 1, "%02d:%02d h"%(int(i*dt/60),i*dt%60),color='w')
        else:
            t1 = ax.text(10, 1, "",color='k')

        im = plt.imshow(viewer.window.qt_viewer.canvas.render(),animated=True)
        plt.gca().set_position([0, 0, 1, 1])
        ims.append([t1, im])


    ani=animation.ArtistAnimation(fig,ims, interval=50,blit=True,repeat_delay=1000)

    writer = animation.FFMpegWriter(fps=fps, bitrate=bitrate)

    ani.save(filename, writer=writer)