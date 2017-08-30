import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, filename,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(dpi=450)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)

    ax.set_xlabel('Predicted')
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45)

    ax.set_ylabel('Trueth')
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes, )

    iax = ax.imshow(cm, cmap=cmap)
    fig.colorbar(iax)

    # label cm value on the plot
    brightness_threshold = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            # background color at the point
            bg_color = iax.get_cmap()(iax.norm(cm[i, j]))

            # calculate brightness according to ITU BT.709
            bg_color_brightness = \
                0.2126 * bg_color[0] \
                + 0.7152 * bg_color[1] \
                + 0.0722 * bg_color[2]

            # put text on the plot
            text_color = \
                'white' if bg_color_brightness < brightness_threshold \
                else 'black'
            ax.text(j, i, '%0.2f' % cm[i, j], color=text_color,
                    horizontalalignment="center")

    fig.savefig(filename)
