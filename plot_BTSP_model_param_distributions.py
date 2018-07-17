__author__ = 'milsteina'
from BTSP_utils import *
from nested.optimize_utils import *
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from collections import defaultdict
import click


mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size'] = 11.
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['text.usetex'] = False
mpl.rcParams['axes.titlepad'] = 2.
mpl.rcParams['mathtext.default'] = 'regular'
mpl.rcParams['axes.unicode_minus'] = True


context = Context()


@click.command()
@click.option("--param-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/20180511_best_params.yaml')
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_BTSP_CA1_cli_config.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export", is_flag=True)
@click.option("--label", type=str, default=None)
def main(param_file_path, config_file_path, output_dir, export, label):
    """

    :param param_file_path: str (path)
    :param config_file_path: str (path)
    :param output_dir: str (dir)
    :param export: bool
    :param label: str
    """
    date_stamp = datetime.datetime.today().strftime('%Y%m%d_%H%M')
    if label is None:
        label = date_stamp
    else:
        label = '%s_%s' % (date_stamp, label)
    if not os.path.isfile(param_file_path):
        raise IOError('Invalid param_file_path: %s' % param_file_path)
    if export and not os.path.isdir(output_dir):
        raise IOError('Invalid output_dir: %s' % output_dir)
    plot_BTSP_model_param_cdfs(param_file_path, config_file_path, export, output_dir, label)
    context.update(locals())


def plot_BTSP_model_param_cdfs(param_file_path, config_file_path, export=False, output_dir=None, label=None):
    """

    :param param_file_path: str (path)
    :param config_file_path: str (path)
    :param export: bool
    :param output_dir: str (dir)
    :param label: str
    :return: dict
    """
    params_by_cell = read_from_yaml(param_file_path)
    cell_keys = [key for key in params_by_cell if key not in ['all']]
    params = defaultdict(list)
    ordered_param_keys = ['local_signal_rise', 'local_signal_decay', 'global_signal_rise', 'global_signal_decay',
                          'peak_delta_weight', 'rMC0', 'rMC_th', 'rMC_peak', 'rCM0', 'rCM_th1', 'rCM_peak1', 'rCM_th2',
                          'rCM_peak2', 'rCM_min2']
    param_labels = {'local_signal_rise': 'local signal tau$_{rise}$ (ms)',
                    'local_signal_decay': 'local signal tau$_{decay}$ (ms)',
                    'global_signal_rise': 'global signal tau$_{rise}$ (ms)',
                    'global_signal_decay': 'global signal tau$_{decay}$ (ms)',
                    'peak_delta_weight': 'max synaptic weight',
                    'rMC0': 'k$_{pot}$ (Hz)',
                    'rMC_th': 'rate$_{pot}$ th',
                    'rMC_peak': 'rate$_{pot}$ xpeak',
                    'rCM0': u'k$_{de\u2011pot}$ (Hz)',
                    'rCM_th1': u'rate$_{de\u2011pot}$ th1',
                    'rCM_peak1': u'rate$_{de\u2011pot}$ xpeak1',
                    'rCM_th2': u'rate$_{de\u2011pot}$ th2',
                    'rCM_peak2': u'rate$_{de\u2011pot}$ xpeak2',
                    'rCM_min2': u'rate$_{de\u2011pot}$ ymin2'}
    bounds = read_from_yaml(config_file_path)['bounds']

    for cell in cell_keys:
        for param, val in params_by_cell[cell].iteritems():
            if param == 'peak_delta_weight':
                val += 1.
            params[param].append(val)

    fig, axes = plt.figure(figsize=(10, 9)), []
    gs0 = gridspec.GridSpec(4, 4, wspace=0.65, hspace=0.7, left=0.1, right=0.95, top=0.95, bottom=0.075)
    axes = []
    for row in xrange(4):
        for col in xrange(4):
            this_axis = fig.add_subplot(gs0[row, col])
            axes.append(this_axis)
    for i, param_name in enumerate(ordered_param_keys):
        params[param_name].sort()
        vals = params[param_name]
        n = len(vals)
        xlim = np.array(bounds[param_name])
        if param_name == 'peak_delta_weight':
            xlim = np.add(xlim, 1.)
        xlim[0] = xlim[0] -0.05 * xlim[1]
        xlim[1] *= 1.05
        axes[i].plot(vals, np.add(np.arange(n), 1.)/float(n), c='k')
        axes[i].set_xlim(xlim)
        axes[i].set_ylim(0., axes[i].get_ylim()[1])
        axes[i].set_xlabel(param_labels[param_name])
        if i % 4 == 0:
            axes[i].set_ylabel('Cum. fraction')

    clean_axes(axes)
    if export:
        fig_path = '%s/%s_BTSP_model_param_cdfs.svg' % (output_dir, label)
        fig.savefig(fig_path, format='svg')
    plt.show()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
