"""
An example of how to use this:
x ,y , y_target = priors.fast_gp.get_batch(1,100,num_features, hyperparameters=(1e-4,1.,.6), equidistant_x=True)
fig, ax = pyplot.subplots(figsize=[10,10])
plot_model_and_orig_curve(ax, SOME_MODEL, x, y, given_indices[10,40,60])

Don't worry it is normal to be slow...
"""
import matplotlib.patches as patches
import seaborn as sns
import torch


def add_rect(ax, coord, height, width, color):
    rect = patches.Rectangle(coord, height, width, linewidth=1, edgecolor='none', facecolor=color)

    # Add the patch to the Axes
    ax.add_patch(rect)


def heatmap_with_box_sizes(ax, data: torch.Tensor, x_starts, x_ends, y_starts, y_ends,
                           palette=sns.color_palette("rocket", as_cmap=True), set_lims=True):
    """
    Beware all x and y arrays should be sorted from small to large and the data will appear in that same order: Small indexes map to lower x/y-axis values.
    """
    if set_lims:
        ax.set_xlim(x_starts[0], x_ends[-1])
        ax.set_ylim(y_starts[0], y_ends[-1])

    data = (data - data.min()) / (data.max() - data.min())

    for col_i, (col_start, col_end) in enumerate(zip(x_starts, x_ends)):
        for row_i, (row_start, row_end) in enumerate(zip(y_starts, y_ends)):
            add_rect(ax, (col_start, row_start), col_end - col_start, row_end - row_start,
                     palette(data[row_i, col_i].item()))


print(ax.get_ylim())


def plot_bar_distribution(ax, x: torch.Tensor, bar_borders: torch.Tensor, predictions: torch.Tensor, **kwargs):
    x = x.squeeze()
    predictions = predictions.squeeze()
    assert len(x.shape) == 1 and len(predictions.shape) == 2 and len(predictions) == len(x) and len(
        bar_borders.shape) == 1 and len(bar_borders) - 1 == predictions.shape[1]

    y_starts = bar_borders[:-1]
    y_ends = bar_borders[1:]

    x, order = x.sort(0)
    print(x.shape, predictions.shape, order.shape)

    predictions = predictions[order] / (bar_borders[1:] - bar_borders[:-1])
    print(predictions.shape)

    # assume x is sorted
    x_starts = torch.cat([x[0].unsqueeze(0), (x[1:] + x[:-1]) / 2])
    x_ends = torch.cat([(x[1:] + x[:-1]) / 2, x[-1].unsqueeze(0), ])

    heatmap_with_box_sizes(ax, predictions.T, x_starts, x_ends, y_starts, y_ends, **kwargs)


def plot_model_w_eval_pos(ax, model, x, y, single_eval_pos, softmax=False, min_max_y=None, **kwargs):
    with torch.no_grad():
        model.eval()
        y_pred = model((x, y), single_eval_pos=single_eval_pos)
        if softmax:
            y_pred = y_pred.softmax(-1)
    if min_max_y:
        lowest_bar = torch.searchsorted(model.criterion.borders, min_max_y[0])
        highest_bar = min(torch.searchsorted(model.criterion.borders, min_max_y[1]), len(model.criterion.borders))
        borders = model.criterion.borders[lowest_bar:highest_bar]
        y_pred = y_pred[..., lowest_bar:highest_bar - 1]
    else:
        borders = model.criterion.borders
    plot_bar_distribution(ax, x[single_eval_pos:], borders, y_pred, **kwargs)


def plot_model_and_orig_curve(ax, model, x, y, given_indices=[0]):
    """
    :param ax: A standard pyplot ax
    :param model: A Transformer Model with `single_eval_pos`
    :param x: A three-dimensional input tensor with x.shape[0]=1 and x.shape[2]=1
    :param y: A two-dimensional tensor with y.shape[1]=0
    :param given_indices: The indexes in y which should be given to the model (the training points)
    :return:
    """
    x_winput = torch.cat([x[given_indices], x], 0)
    y_winput = torch.cat([y[given_indices], y], 0)

    ax.plot(x.squeeze(), y.squeeze(), color='grey')
    ax.plot(x.squeeze()[given_indices], y.squeeze()[given_indices], 'o', color='black')
    plot_model_w_eval_pos(ax, model, x_winput, y_winput, len(given_indices),
                          min_max_y=(y.min() - .3, y.max() + .3), softmax=True,
                          palette=sns.cubehelix_palette(start=2, rot=0, dark=0.4, light=1, as_cmap=True))


