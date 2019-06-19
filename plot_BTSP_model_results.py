__author__ = 'milsteina'
from nested.optimize_utils import *
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from collections import defaultdict
import click


mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size'] = 12.
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['text.usetex'] = False
mpl.rcParams['axes.titlepad'] = 2.
mpl.rcParams['mathtext.default'] = 'regular'


context = Context()


@click.command()
@click.option("--model-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='data/20190221_BTSP_D_all_cells_merged_exported_data.hdf5')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export", is_flag=True)
@click.option("--show-traces", is_flag=True)
@click.option("--label", type=str, default=None)
def main(model_file_path, output_dir, export, show_traces, label):
    """

    :param model_file_path: str (path)
    :param output_dir: str (dir)
    :param export: bool
    :param show_traces: bool
    :param label: str
    """
    date_stamp = datetime.datetime.today().strftime('%Y%m%d_%H%M')
    if label is None:
        label = date_stamp
    else:
        label = '%s_%s' % (date_stamp, label)
    if not os.path.isfile(model_file_path):
        raise IOError('Invalid model_file_path: %s' % model_file_path)
    if export and not os.path.isdir(output_dir):
        raise IOError('Invalid output_dir: %s' % output_dir)
    ramp_amp, ramp_width, peak_shift, min_val, ramp_mse_dict, ramp_mse_array, delta_exp_ramp, delta_model_ramp = \
        process_BTSP_model_results(model_file_path, show_traces, export, output_dir, label)
    plot_BTSP_model_fit_summary(ramp_amp, ramp_width, peak_shift, min_val, ramp_mse_dict, export, output_dir, label)
    context.update(locals())


def process_BTSP_model_results(file_path, show=False, export=False, output_dir=None, label=None):
    """

    :param file_path: str (path)
    :param show: bool
    :param export: bool
    :param output_dir: str (dir)
    :param label: str
    :return: tuple of dict
    """
    ramp_amp = defaultdict(lambda: defaultdict(dict))
    ramp_width = defaultdict(lambda: defaultdict(dict))
    local_peak_shift = defaultdict(lambda: defaultdict(dict))
    min_val = defaultdict(lambda: defaultdict(dict))
    ramp_mse_dict = defaultdict(dict)
    delta_model_ramp = []
    delta_exp_ramp = []
    ramp_mse_array = []
    with h5py.File(file_path, 'r') as f:
        shared_context_key = 'shared_context'
        group = f[shared_context_key]
        peak_locs = group['peak_locs'][:]
        binned_x = group['binned_x'][:]
        param_names = group['param_names'][:]
        exported_data_key = 'exported_data'
        for cell_key in f[exported_data_key]:
            description = 'model_ramp_features'
            group = f[exported_data_key][cell_key].itervalues().next()[description]
            param_array = group['param_array'][:]
            param_dict = param_array_to_dict(param_array, param_names)
            if 'peak_delta_weight' in param_dict:
                peak_delta_weight = param_dict['peak_delta_weight']
            else:
                peak_delta_weight = 1.

            for induction_key in f[exported_data_key][cell_key]:
                description = 'model_ramp_features'
                group = f[exported_data_key][cell_key][induction_key][description]
                ramp_amp[cell_key][induction_key]['target'] = group.attrs['target_ramp_amp']
                ramp_width[cell_key][induction_key]['target'] = group.attrs['target_ramp_width']
                local_peak_shift[cell_key][induction_key]['target'] = group.attrs['target_local_peak_shift']
                min_val[cell_key][induction_key]['target'] = group.attrs['target_min_val']
                ramp_amp[cell_key][induction_key]['model'] = group.attrs['model_ramp_amp']
                ramp_width[cell_key][induction_key]['model'] = group.attrs['model_ramp_width']
                local_peak_shift[cell_key][induction_key]['model'] = group.attrs['model_local_peak_shift']
                min_val[cell_key][induction_key]['model'] = group.attrs['model_min_val']

            fig3, axes3 = plt.subplots(2, 3, figsize=[16, 6])
            ymin = -1.
            ymax = 10.
            for induction_key in f[exported_data_key][cell_key]:
                i = int(float(induction_key)) - 1
                description = 'model_ramp_features'
                group = f[exported_data_key][cell_key][induction_key][description]
                model_weights = group['model_weights'][:]
                initial_weights = group['initial_weights'][:]
                delta_weights = np.subtract(model_weights, initial_weights)
                peak_delta_weight = max([max(delta_weights), abs(min(delta_weights)), peak_delta_weight])
                initial_model_ramp = group['initial_model_ramp'][:]
                target_ramp = group['target_ramp'][:]
                model_ramp = group['model_ramp'][:]

                this_mse_array = np.square(np.subtract(target_ramp, model_ramp))
                ramp_mse_array.extend(this_mse_array)
                ramp_mse_dict[cell_key][induction_key] = np.mean(this_mse_array)

                ymin = min(ymin, np.min(model_ramp) - 1., np.min(target_ramp) - 1.)
                ymax = max(ymax, np.max(model_ramp) + 1., np.max(target_ramp) + 1.)
                if 'initial_exp_ramp' in group:
                    initial_exp_ramp = group['initial_exp_ramp'][:]
                    axes3[i][0].plot(binned_x, initial_exp_ramp, label='Before', c='darkgrey')
                    ymin = min(ymin, np.min(initial_exp_ramp) - 1.)
                    ymax = max(ymax, np.max(initial_exp_ramp) + 1.)
                else:
                    initial_exp_ramp = np.zeros_like(target_ramp)
                delta_model_ramp.extend(np.subtract(model_ramp, initial_exp_ramp))
                delta_exp_ramp.extend(np.subtract(target_ramp, initial_exp_ramp))
                axes3[i][0].plot(binned_x, target_ramp, label='After', c='r')
                axes3[i][0].set_title('Induction %i\nExperiment Vm:' % (i + 1), fontsize=mpl.rcParams['font.size'])
                axes3[i][1].plot(binned_x, initial_model_ramp, label='Before', c='darkgrey')
                axes3[i][1].plot(binned_x, model_ramp, label='After', c='c')
                axes3[i][1].set_title('\nModel Vm', fontsize=mpl.rcParams['font.size'])
                axes3[i][2].plot(peak_locs, delta_weights, c='k')
                axes3[i][2].set_title('Change in\nSynaptic Weights', fontsize=mpl.rcParams['font.size'])

                axes3[i][0].set_xlabel('Location (cm)')
                axes3[i][0].set_ylabel('Depolarization\namplitude (mV)')
                axes3[i][1].set_xlabel('Location (cm)')
                axes3[i][1].set_ylabel('Depolarization\namplitude (mV)')
                axes3[i][2].set_xlabel('Location (cm)')
                axes3[i][2].set_ylabel('Change in synaptic\nweight (a.u.)')
                xmin, xmax = axes3[i][2].get_xlim()
                axes3[i][2].plot([xmin, xmax], [0., 0.], c='darkgrey', alpha=0.5, ls='--')

            for induction_key in f[exported_data_key][cell_key]:
                i = int(float(induction_key)) - 1
                description = 'model_ramp_features'
                group = f[exported_data_key][cell_key][induction_key][description]
                mean_induction_start_loc = group.attrs['mean_induction_start_loc']
                mean_induction_stop_loc = group.attrs['mean_induction_stop_loc']
                axes3[i][0].set_ylim([ymin, ymax * 1.05])
                axes3[i][1].set_ylim([ymin, ymax * 1.05])
                axes3[i][2].set_ylim([-peak_delta_weight, peak_delta_weight * 1.1])
                axes3[i][0].hlines(ymax, xmin=mean_induction_start_loc, xmax=mean_induction_stop_loc)
                axes3[i][1].hlines(ymax, xmin=mean_induction_start_loc, xmax=mean_induction_stop_loc)
                axes3[i][2].hlines(peak_delta_weight * 1.05, xmin=mean_induction_start_loc,
                                   xmax=mean_induction_stop_loc)
                axes3[i][0].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
                axes3[i][1].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
            clean_axes(axes3)
            fig3.suptitle('Cell: %i' % int(float(cell_key)), fontsize=mpl.rcParams['font.size'])
            fig3.tight_layout()
            fig3.subplots_adjust(top=0.9, hspace=0.75)
            if export:
                fig_path = '%s/%s_BTSP_cell_%s_ramp_traces.svg' % (output_dir, label, cell_key)
                fig3.savefig(fig_path, format='svg')
    if show:
        plt.show()
    else:
        plt.close('all')

    return ramp_amp, ramp_width, local_peak_shift, min_val, ramp_mse_dict, ramp_mse_array, delta_exp_ramp, \
           delta_model_ramp


def plot_BTSP_model_fit_summary(ramp_amp, ramp_width, peak_shift, min_val, ramp_mse_dict, export=False, output_dir=None,
                                label=None):
    """

    :param ramp_amp: dict
    :param ramp_width: dict
    :param peak_shift: dict
    :param min_val: dict
    :param ramp_mse_dict: dict
    :param export: bool
    :param output_dir: str (dir)
    :param label: str
    """
    fig, axes = plt.figure(figsize=(12, 8.5)), []
    gs0 = gridspec.GridSpec(3, 4, wspace=0.55, hspace=0.9, left=0.075, right=0.975, top=0.925, bottom=0.075)
    axes = []
    tick_locs = [np.arange(0., 16., 3.), np.arange(30., 151., 30.), np.arange(-120., 121., 60.), np.arange(-2., 9., 2.)]
    for col, (parameter, label, tick_loc) in \
            enumerate(zip([ramp_amp, ramp_width, peak_shift, min_val],
                          ['Peak amplitude (mV)', 'Width (cm)', 'Peak shift (cm)', 'Minimum amplitude (mV)'],
                          tick_locs)):
        this_axis = fig.add_subplot(gs0[2, col])
        axes.append(this_axis)
        target_vals, model_vals = {}, {}
        for color, induction_key in zip(['darkgrey', 'c'], ['1', '2']):
            target_vals[induction_key] = []
            model_vals[induction_key] = []
            for cell_key in (cell_key for cell_key in parameter if induction_key in parameter[cell_key]):
                target_vals[induction_key].append(parameter[cell_key][induction_key]['target'])
                model_vals[induction_key].append(parameter[cell_key][induction_key]['model'])
            this_axis.scatter(target_vals[induction_key], model_vals[induction_key], color=color, alpha=0.5,
                              label='Induction %s' % induction_key)
        r_val, p_val = pearsonr(target_vals['1'] + target_vals['2'], model_vals['1'] + model_vals['2'])
        this_axis.annotate('R$^{2}$ = %.3f; p < %.3f' % (r_val ** 2., p_val if p_val > 0.001 else 0.001),
                           xy=(0.25, 0.05), xycoords='axes fraction')
        this_axis.set_title(label, fontsize=mpl.rcParams['font.size'])
        if col == 0:
            this_axis.legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, handletextpad=0.5,
                             fontsize=mpl.rcParams['font.size'])
        this_axis.set_xlim(np.min(tick_loc), np.max(tick_loc))
        this_axis.set_ylim(np.min(tick_loc), np.max(tick_loc))
        this_axis.set_xticks(tick_loc)
        this_axis.set_yticks(tick_loc)
        this_axis.plot(tick_loc, tick_loc, c='darkgrey', alpha=0.5, ls='--')
        this_axis.set_xlabel('Experiment')
        this_axis.set_ylabel('Model')
    this_axis = fig.add_subplot(gs0[1, 0])
    axes.append(this_axis)
    vals = defaultdict(list)
    for cell_key in ramp_mse_dict:
        for induction_key in ramp_mse_dict[cell_key]:
            vals[induction_key].append(ramp_mse_dict[cell_key][induction_key])
    max_val = 0.
    for induction_key in vals:
        vals[induction_key].sort()
        max_val = max(max_val, np.max(vals[induction_key]))
        n = len(vals[induction_key])
        this_axis.plot(vals[induction_key], np.add(np.arange(n), 1.) / float(n), label='Induction: %s' % induction_key)
    this_axis.set_ylim(0., 1.05)
    this_axis.set_ylabel('Cum. fraction')
    this_axis.set_xlim(0., math.ceil(max_val))
    int_max_val = int(math.ceil(max_val))
    int_delta_val = max(1, int_max_val / 6)
    this_axis.set_xticks(np.arange(0, int_max_val+1, int_delta_val))
    this_axis.set_xlabel('Mean squared error')
    this_axis.set_title('Model ramp residual error', fontsize=mpl.rcParams['font.size'])
    this_axis.legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, handletextpad=0.5,
                     fontsize=mpl.rcParams['font.size'])
    clean_axes(axes)
    if export:
        fig_path = '%s/%s_BTSP_model_fit_summary.svg' % (output_dir, label)
        fig.savefig(fig_path, format='svg')
    plt.show()


def plot_compare_models_mse_by_induction(model_file_path_dict=None):
    """

    :param model_file_path_dict: {label (str): file_path (str; path)}
    """
    all_mse = dict()
    max_val = defaultdict(lambda: 0.)
    if model_file_path_dict is None:
        # model_file_path_dict = {model: 'data/20190221_BTSP_%s_all_cells_merged_exported_data.hdf5' % model
        #                        for model in ['V_A', 'V_B', 'D', 'E']}
        model_file_path_dict = {label: 'data/20190221_BTSP_%s_all_cells_merged_exported_data.hdf5' % model
                                for label, model in zip(['Voltage-dependent model', 'Bistable synapse model'],
                                                        ['V_B', 'D'])}
    for model, model_file_path in model_file_path_dict.iteritems():
        ramp_amp, ramp_width, peak_shift, min_val, all_mse[model], discard_ramp_mse_array, discard_delta_exp_ramp, \
        discard_delta_model_ramp = process_BTSP_model_results(model_file_path)
    fig, axes = plt.subplots(1, 2)

    for model in all_mse:
        vals = defaultdict(list)
        for cell_key in all_mse[model]:
            for induction_key in all_mse[model][cell_key]:
                vals[induction_key].append(all_mse[model][cell_key][induction_key])
        for induction_key in vals:
            if induction_key == '1':
                col = 0
            elif induction_key == '2':
                col = 1
            vals[induction_key].sort()
            max_val[induction_key] = max(max_val[induction_key], np.max(vals[induction_key]))
            n = len(vals[induction_key])
            axes[col].plot(vals[induction_key], np.add(np.arange(n), 1.) / float(n), label=model)

    for induction_key in vals:
        if induction_key == '1':
            col = 0
        elif induction_key == '2':
            col = 1
        axes[col].set_title('Induction %s' % induction_key, fontsize=mpl.rcParams['font.size'])
        axes[col].set_xlim(0., math.ceil(max_val[induction_key]))
        int_max_val = int(math.ceil(max_val[induction_key]))
        int_delta_val = max(1, int_max_val / 6)
        axes[col].set_xticks(np.arange(0, int_max_val + 1, int_delta_val))
        axes[col].set_ylim(0., 1.05)
        axes[col].set_ylabel('Cum. fraction')
        axes[col].set_xlabel('Mean squared error')

    axes[0].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, handletextpad=0.5,
                   fontsize=mpl.rcParams['font.size'])
    fig.suptitle('Model ramp residual error', fontsize=mpl.rcParams['font.size'])
    clean_axes(axes)
    fig.tight_layout(rect=[0., 0., 1., 0.95])
    fig.show()


def plot_compare_models_mse_vs_delta_ramp(model_file_path_dict=None):
    """

    :param model_file_path_dict: {label (str): file_path (str; path)}
    """
    all_mse = dict()
    max_val = defaultdict(lambda: 0.)
    ramp_mse_array = dict()
    delta_exp_ramp = dict()
    delta_model_ramp = dict()
    if model_file_path_dict is None:
        model_file_path_dict = {label: 'data/20190221_BTSP_%s_all_cells_merged_exported_data.hdf5' % model
                                for label, model in zip(['Voltage-dependent model', 'Bistable synapse model'],
                                                        ['V_B', 'D'])}
    for model, model_file_path in model_file_path_dict.iteritems():
        ramp_amp, ramp_width, peak_shift, min_val, all_mse[model], ramp_mse_array[model], delta_exp_ramp[model],\
            delta_model_ramp[model] = process_BTSP_model_results(model_file_path)

    fig, axes = plt.subplots(2, 3, figsize=(12., 6.5))

    for model in all_mse:
        vals = defaultdict(list)
        for cell_key in all_mse[model]:
            for induction_key in all_mse[model][cell_key]:
                vals[induction_key].append(all_mse[model][cell_key][induction_key])
        for induction_key in vals:
            if induction_key == '1':
                col = 0
            elif induction_key == '2':
                col = 1
            vals[induction_key].sort()
            max_val[induction_key] = max(max_val[induction_key], np.max(vals[induction_key]))
            n = len(vals[induction_key])
            axes[0][col].plot(vals[induction_key], np.add(np.arange(n), 1.) / float(n), label=model)

    for induction_key in vals:
        if induction_key == '1':
            col = 0
        elif induction_key == '2':
            col = 1
        axes[0][col].set_title('Induction %s' % induction_key, fontsize=mpl.rcParams['font.size'])
        axes[0][col].set_xlim(0., math.ceil(max_val[induction_key]))
        int_max_val = int(math.ceil(max_val[induction_key]))
        int_delta_val = max(1, int_max_val / 6)
        axes[0][col].set_xticks(np.arange(0, int_max_val + 1, int_delta_val))
        axes[0][col].set_ylim(0., 1.05)
        axes[0][col].set_ylabel('Cum. fraction')
        axes[0][col].set_xlabel('Mean squared error')

    axes[0][0].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, handletextpad=0.5,
                   fontsize=mpl.rcParams['font.size'])

    for model in ramp_mse_array:
        if model == 'Bistable synapse model':
            axes[0][2].scatter(delta_exp_ramp[model], ramp_mse_array[model], marker='.', label=model, alpha=0.25,
                               s=30., zorder=1, linewidths=0)
        else:
            axes[0][2].scatter(delta_exp_ramp[model], ramp_mse_array[model], marker='.', label=model, alpha=0.25,
                               s=30., zorder=0, linewidths=0)
    axes[0][2].set_ylabel('Squared error')
    axes[0][2].set_xlabel('Change in experimental ramp amplitude (mV)')
    axes[0][2].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, handletextpad=0.5,
                   fontsize=mpl.rcParams['font.size'])
    axes[0][2].set_title('Model ramp residual error', fontsize=mpl.rcParams['font.size'], pad=10.)

    clean_axes(axes)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.65, top=0.9)
    fig.show()
    context.update(locals())


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
