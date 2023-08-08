import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np

def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_alignment_to_numpy(alignment, title='', info=None, phoneme_seq=None,
                            vmin=None, vmax=None):
    if phoneme_seq:
        fig, ax = plt.subplots(figsize=(15, 10))
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    if phoneme_seq != None:
        # for debugging of phonemes and durs in maps. Not used by def in training code
        ax.set_yticks(np.arange(len(phoneme_seq)))
        ax.set_yticklabels(phoneme_seq)
        ax.hlines(np.arange(len(phoneme_seq)), xmin=0.0, xmax=max(ax.get_xticks()))

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data