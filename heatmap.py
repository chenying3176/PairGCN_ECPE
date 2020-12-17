import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.


    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)


    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)


    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    im = ax.imshow(data, **kwargs)



    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")


    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im

x_labelss = [
    ['C1', 'C2', 'C3 (Emotion)', 'C4', 'C5', 'C6'], 
    ['C1', 'C2 (Cause)', 'C3', 'C4', 'C5', 'C6'], 
    ['C1', 'C2', 'C3 (Emotion)', 'C4', 'C5', 'C6'], 
    ['C1', 'C2 (Cause)', 'C3', 'C4', 'C5', 'C6']
]
y_labelss = [
    ['C1', 'C2 (Cause)', 'C3', 'C4', 'C5', 'C6'], 
    ['C1', 'C2', 'C3 (Emotion)', 'C4', 'C5', 'C6'], 
    ['C1', 'C2 (Cause)', 'C3', 'C4', 'C5', 'C6'], 
    ['C1', 'C2', 'C3 (Emotion)', 'C4', 'C5', 'C6']
]

filenames = [
    'images/masked_c2e.png',
    'images/masked_e2c.png',
    'images/raw_c2e.png',
    'images/raw_e2c.png'
]

harvests = [
    # masked c2e
    np.array([    
        [0.0000, 0.4727, 0.0000, 0.0000, 0.0000, 0.0000],  
        [0.0000, 0.3674, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.2842, 0.0000, 0.0000, 0.2265, 0.0000], 
        [0.0000, 0.2760, 0.0000, 0.0000, 0.2200, 0.0000], 
        [0.0000, 0.0000, 0.0000, 0.0000, 0.3039, 0.2689], 
        [0.0000, 0.0000, 0.0000, 0.0000, 0.3780, 0.3345],  
    ]),
    # masked e2c
    np.array([
        [0.0000, 0.0000, 0.3442, 0.0000, 0.0000, 0.0000], 
        [0.0000, 0.0000, 0.2502, 0.2731, 0.0000, 0.0000], 
        [0.0000, 0.0000, 0.0000, 0.2139, 0.2169, 0.0000], 
        [0.0000, 0.0000, 0.0000, 0.2064, 0.2093, 0.2133], 
        [0.0000, 0.0000, 0.0000, 0.2523, 0.2558, 0.2608], 
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3391]
    ]), 
    # raw c2e
    np.array([
        [0.9023, 0.0283, 0.0694, 0.0000, 0.0000, 0.0000], 
        [0.5484, 0.0172, 0.0422, 0.3922, 0.0000, 0.0000], 
        [0.2246, 0.0070, 0.0173, 0.1606, 0.5905, 0.0000], 
        [0.0000, 0.0052, 0.0128, 0.1191, 0.4379, 0.4250], 
        [0.0000, 0.0000, 0.0129, 0.1197, 0.4402, 0.4272], 
        [0.0000, 0.0000, 0.0000, 0.1213, 0.4459, 0.4328]
    ]), 
    # raw e2c
    np.array([
        [0.5965, 0.3989, 0.0046, 0.0000, 0.0000, 0.0000], 
        [0.3753, 0.2509, 0.0029, 0.3709, 0.0000, 0.0000], 
        [0.2120, 0.1417, 0.0016, 0.2095, 0.4352, 0.0000], 
        [0.0000, 0.1159, 0.0013, 0.1713, 0.3558, 0.3557], 
        [0.0000, 0.0000, 0.0015, 0.1937, 0.4024, 0.4023], 
        [0.0000, 0.0000, 0.0000, 0.1940, 0.4031, 0.4029]
    ])
]

for x_labels, y_labels, harvest, filename in zip(x_labelss, y_labelss, harvests, filenames):

    fig, ax = plt.subplots()

    im = heatmap(harvest, x_labels, y_labels, ax=ax,
                       cmap="Reds")
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    texts = annotate_heatmap(im, valfmt="{x:.2f}")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="center",
             rotation_mode="anchor")
    fig.tight_layout()
    plt.savefig(filename, dpi=1000, quality=100, transparent=True)



