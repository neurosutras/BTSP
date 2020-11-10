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
                'k_pot': r'$k^+$ (/s)',
                'f_pot_th': r'$\alpha^+$',
                'f_pot_half_width': r'$\beta^+$',
                'k_dep': r'$k^-$ (/s)',
                'f_dep_th': r'$\alpha^-$',
                'f_dep_half_width': r'$\beta^-$',
                'peak_delta_weight': u'\u0394$W_{max}$',
                'plateau_delta_depo': u'\u0394$V_{max}$'
                }

ordered_param_names = ['local_signal_decay', 'global_signal_decay', 'k_pot', 'k_dep',
                       'f_pot_th', 'f_pot_half_width', 'f_dep_th', 'f_dep_half_width',
                       'peak_delta_weight', 'plateau_delta_depo']


@click.command()
@click.option("--param-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/20200528_biBTSP_WD_D_90cm_best_params.yaml')
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_biBTSP_WD_D_cli_config.yaml')
@click.option("--data-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='data/20200527_biBTSP_data.hdf5')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export", is_flag=True)
@click.option("--label", type=str, default=None)
def main(param_file_path, config_file_path, data_file_path, output_dir, export, label):
    """

    :param param_file_path: str (path)
    :param config_file_path: str (path)
    :param data_file_path: str (path)
    :param output_dir: str (dir)
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
    if not os.path.isfile(data_file_path):
        raise IOError('Invalid data_file_path: %s' % data_file_path)
    if export and not os.path.isdir(output_dir):
        raise IOError('Invalid output_dir: %s' % output_dir)
    param_dict = read_from_yaml(param_file_path)
    config_dict = read_from_yaml(config_file_path)
    context.update(locals())
    plot_BTSP_model_param_cdfs(param_dict, config_dict, data_file_path, export, output_dir, label)


def plot_BTSP_model_param_cdfs(param_dict, config_dict, data_file_path, export=False, output_dir=None, label=None,
                               cols=5):
    """

    :param param_dict: dict
    :param config_dict: dict
    :param data_file_path: str (path)
    :param export: bool
    :param output_dir: str (dir)
    :param label: str
    :param cols: int
    """
    cell_keys = [key for key in param_dict if key not in ['all', 'default']]
    params = defaultdict(list)

    bounds = config_dict['bounds']

    for cell in cell_keys:
        for param_name, param_val in viewitems(param_dict[cell]):
            params[param_name].append(param_val)

    fig, axes = plt.figure(figsize=(12, 4.5)), []
    gs0 = gridspec.GridSpec(2, cols, wspace=0.65, hspace=0.7, left=0.075, right=0.975, top=0.9, bottom=0.15)
    axes = []
    for row in range(2):
        for col in range(cols):
            this_axis = fig.add_subplot(gs0[row, col])
            axes.append(this_axis)
    ordered_param_keys = [param_name for param_name in ordered_param_names if param_name in params]
    for param_name in ordered_param_keys:
        print(param_name)
    print('%s:' % label)
    for i, param_name in enumerate(ordered_param_keys):
        # params[param_name].sort()
        if param_name == 'plateau_delta_depo':
            vals = get_V_max_for_VD_models(param_dict, config_dict, data_file_path, label=label)
        else:
            vals = np.array(params[param_name])
        n = len(vals)
        print(u'%.2f \u00B1 %.2f' % (np.mean(vals), np.std(vals) / np.sqrt(n)))
        if '_half_width' in param_name:
            vals = 2. / vals
            xlim = 2. / np.array(bounds[param_name])
            xlim = np.sort(xlim)
        else:
            xlim = [min(np.min(vals), bounds[param_name][0]), max(np.max(vals), bounds[param_name][1])]
        vals = np.sort(vals)
        xlim[0] = xlim[0] - 0.05 * xlim[1]
        xlim[1] *= 1.05
        axes[i].plot(vals, np.add(np.arange(n), 1.)/float(n), c='k')
        axes[i].set_xlim(xlim)
        axes[i].set_ylim(0., axes[i].get_ylim()[1])
        axes[i].set_xlabel(param_labels[param_name])
        if i % cols == 0:
            axes[i].set_ylabel('Cum. fraction')
    print_beta_values_from_half_widths(param_dict, config_dict, label)
    clean_axes(axes)
    fig.suptitle('%s:' % label, fontsize=mpl.rcParams['font.size'], x=0.1, y=0.99)
    if export:
        fig_path = '%s/%s_biBTSP_model_param_cdfs.svg' % (output_dir, label)
        fig.savefig(fig_path, format='svg')
    fig.show()


def get_V_max_for_VD_models(param_dict, config_dict, data_file_path, param_name='plateau_delta_depo', label=None):
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
    return vals


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


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
