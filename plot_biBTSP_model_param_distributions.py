__author__ = 'milsteina'
from biBTSP_utils import *
from nested.optimize_utils import *
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from collections import defaultdict
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import click


mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size'] = 11.
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['text.usetex'] = False
mpl.rcParams['axes.titlepad'] = 2.
mpl.rcParams['mathtext.default'] = 'regular'
mpl.rcParams['axes.unicode_minus'] = True


context = Context()

param_labels = {'local_signal_decay': r'$\tau_E$ (ms)',
                'global_signal_decay': r'$\tau_I$ (ms)',
                'k_pot': r'$k^+$ (Hz)',
                'f_pot_th': r'$q^+$ threshold',
                'f_pot_half_width': r'$q^+$ half width',
                'k_dep': r'$k^-$ (Hz)',
                'f_dep_th': r'$q^-$ threshold',
                'f_dep_half_width': r'$q^-$ half width',
                'peak_delta_weight': u'\u0394$W_{max}$',
                'plateau_delta_depo': u'\u0394$V_{max}$'
                }

ordered_param_names = ['local_signal_decay', 'global_signal_decay', 'k_pot', 'k_dep',
                       'f_pot_th', 'f_pot_half_width', 'f_dep_th', 'f_dep_half_width',
                       'peak_delta_weight', 'plateau_delta_depo']


@click.command()
@click.option("--param-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/20190812_biBTSP_WD_B_90cm_best_params.yaml')
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_biBTSP_WD_B_cli_config.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--pcadim", type=int, default=None)
@click.option("--export", is_flag=True)
@click.option("--label", type=str, default=None)
def main(param_file_path, config_file_path, output_dir, pcadim, export, label):
    """

    :param param_file_path: str (path)
    :param config_file_path: str (path)
    :param output_dir: str (dir)
    :param pcadim: int (number or pca components to plot)
    :param export: bool
    :param label: str
    """
    date_stamp = datetime.datetime.today().strftime('%Y%m%d_%H%M')
    if label is None:
        label = date_stamp
    if not os.path.isfile(param_file_path):
        raise IOError('Invalid param_file_path: %s' % param_file_path)
    if not os.path.isfile(config_file_path):
        raise IOError('Invalid config_file_path: %s' % config_file_path)
    if export and not os.path.isdir(output_dir):
        raise IOError('Invalid output_dir: %s' % output_dir)
    param_dict = read_from_yaml(param_file_path)
    config_dict = read_from_yaml(config_file_path)
    context.update(locals())
    plot_BTSP_model_param_cdfs(param_dict, config_dict, export, output_dir, label)
    if pcadim is not None:
        plot_BTSP_model_param_PCA(param_dict, config_dict, pcadim, export, output_dir, label)


def plot_BTSP_model_param_cdfs(param_dict, config_dict, export=False, output_dir=None, label=None):
    """

    :param param_dict: dict
    :param config_dict: dict
    :param export: bool
    :param output_dir: str (dir)
    :param label: str
    """
    cell_keys = [key for key in param_dict if key not in ['all', 'default']]
    params = defaultdict(list)

    bounds = config_dict['bounds']

    for cell in cell_keys:
        for param_name, param_val in viewitems(param_dict[cell]):
            params[param_name].append(param_val)

    fig, axes = plt.figure(figsize=(10, 9)), []
    gs0 = gridspec.GridSpec(4, 4, wspace=0.65, hspace=0.7, left=0.1, right=0.95, top=0.95, bottom=0.075)
    axes = []
    for row in range(4):
        for col in range(4):
            this_axis = fig.add_subplot(gs0[row, col])
            axes.append(this_axis)
    ordered_param_keys = [param_name for param_name in ordered_param_names if param_name in params]
    for param_name in ordered_param_keys:
        print(param_name)
    print('%s:' % label)
    for i, param_name in enumerate(ordered_param_keys):
        params[param_name].sort()
        vals = params[param_name]
        n = len(vals)
        print(u'%.2f \u00B1 %.2f' % (np.mean(vals), np.std(vals) / np.sqrt(n)))
        xlim = np.array(bounds[param_name])
        xlim[0] = xlim[0] - 0.05 * xlim[1]
        xlim[1] *= 1.05
        axes[i].plot(vals, np.add(np.arange(n), 1.)/float(n), c='k')
        axes[i].set_xlim(xlim)
        axes[i].set_ylim(0., axes[i].get_ylim()[1])
        axes[i].set_xlabel(param_labels[param_name])
        if i % 4 == 0:
            axes[i].set_ylabel('Cum. fraction')
    print_beta_values_from_half_widths(param_dict, config_dict, label)
    clean_axes(axes)
    fig.suptitle('%s:' % label, fontsize=mpl.rcParams['font.size'], x=0.1, y=0.99)
    if export:
        fig_path = '%s/%s_biBTSP_model_param_cdfs.svg' % (output_dir, label)
        fig.savefig(fig_path, format='svg')
    fig.show()


def print_V_max_for_VD_models(param_dict, config_dict, data_file_path, param_name='plateau_delta_depo', label=None):
    """

    :param param_dict: dict
    :param config_dict: dict
    :param data_file_path: str (path)
    :param_name: str
    :label: str
    """
    V_max = defaultdict(lambda: 0.)
    with h5py.File(data_file_path, 'r') as f:
        for cell_key in f['data']:
            for induction_key in f['data'][cell_key]:
                if 'before' in f['data'][cell_key][induction_key]['processed']['exp_ramp']:
                    V_max[cell_key] = \
                        max(V_max[cell_key],
                            np.max(f['data'][cell_key][induction_key]['processed']['exp_ramp']['before'][:]))
                if 'after' in f['data'][cell_key][induction_key]['processed']['exp_ramp']:
                    V_max[cell_key] = \
                        max(V_max[cell_key],
                            np.max(f['data'][cell_key][induction_key]['processed']['exp_ramp']['after'][:]))
    for cell_key in context.param_dict:
        V_max[str(cell_key)] += context.param_dict[cell_key][param_name]
    vals = list(V_max.values())
    n = len(vals)
    print('%s:' % label)
    print(u'%.2f \u00B1 %.2f' % (np.mean(vals), np.std(vals) / np.sqrt(n)))


def print_beta_values_from_half_widths(param_dict, config_dict, label=None):
    """

    :param param_dict: dict
    :param config_dict: dict
    :label: str
    """
    cell_keys = [key for key in param_dict if key not in ['all', 'default']]
    params = defaultdict(list)
    for cell in cell_keys:
        for param_name, param_val in viewitems(param_dict[cell]):
            params[param_name].append(param_val)
    param_keys = ['f_pot_half_width', 'f_dep_half_width']
    new_labels = {'f_pot_half_width': 'beta_+', 'f_dep_half_width': 'beta_-'}
    print(label)
    for param_key in (param_key for param_key in param_keys if param_key in params):
        vals = np.array(params[param_key])
        n = len(vals)
        print(param_key)
        print(u'%.2f \u00B1 %.2f' % (np.mean(vals), np.std(vals) / np.sqrt(n)))
        vals = 2. / vals
        print(new_labels[param_key])
        print(u'%.2f \u00B1 %.2f' % (np.mean(vals), np.std(vals) / np.sqrt(n)))


def plot_BTSP_model_param_PCA(param_dict, config_dict, pcadim=None, export=False, output_dir=None, label=None):
    """

    :param param_dict: dict
    :param config_dict: dict
    :param pcadim: int (number of pca components to plot)
    :param export: bool
    :param output_dir: str (dir)
    :param label: str
    """
    param_names = config_dict['param_names']
    ordered_param_keys = [param_name for param_name in ordered_param_names if param_name in param_names]
    cell_keys = [key for key in param_dict if key not in ['all', 'default']]
    data = []

    for cell in cell_keys:
        data.append(param_dict_to_array(param_dict[cell], ordered_param_keys))
    data = np.array(data)
    scaled_data = scale(data)
    corr = np.corrcoef(scaled_data, rowvar=False)

    pca = PCA(whiten=False)  # , n_components=dim)
    fit = pca.fit(scaled_data)
    print('Explained variance ratios: ')
    print(pca.explained_variance_ratio_)
    # print pca.components_
    trans_data = fit.transform(scaled_data)
    plot_data = trans_data

    if pcadim is None:
        pcadim = pca.n_components_-1
    else:
        pcadim = min(pcadim, pca.n_components_-1)
    plotdim = pcadim -1
    context.update(locals())

    x = np.arange(len(ordered_param_keys) + 1)
    y = np.arange(len(ordered_param_keys) + 1)
    labels = [param_labels[key] for key in ordered_param_keys]
    X, Y = np.meshgrid(x, y[::-1])
    fig, axes = plt.subplots(figsize=(12, 10))
    pc = axes.pcolor(X, Y, corr)
    axes.set_xticks(np.arange(len(ordered_param_keys)) + 0.5)
    axes.set_yticks(np.arange(len(ordered_param_keys)) + 0.5)
    axes.set_xticklabels(labels, ha='right', fontsize=16)
    axes.set_yticklabels(labels[::-1], fontsize=16)
    for tick in axes.get_xticklabels():
        tick.set_rotation(45.)
    fig.colorbar(pc)
    clean_axes(axes)
    fig.tight_layout()
    if export:
        fig_path = '%s/%s_BTSP_model_param_corrcoefs.svg' % (output_dir, label)
        fig.savefig(fig_path, format='svg')
    fig.show()

    if plotdim == 1:
        fig, axes = plt.subplots(figsize=(5, 5))
        axes.scatter(plot_data[:, 0], plot_data[:, 1])
        axes.set_ylabel('PC2')
        axes.set_xlabel('PC1')
    else:
        fig, axes = plt.subplots(plotdim, plotdim, figsize=(10, 10))
        for row in range(plotdim):
            axes[row][0].set_ylabel('PC%i' % (row + 2))
            for col in range(plotdim):
                if row == plotdim - 1:
                    axes[row][col].set_xlabel('PC%i' % (col + 1))
                if row >= col:
                    axes[row][col].scatter(plot_data[:, col], plot_data[:, row + 1])
    clean_axes(axes)
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
