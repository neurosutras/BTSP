__author__ = 'milsteina'
from BTSP_utils import *
from nested.optimize_utils import *
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
import click


mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size'] = 11.
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['text.usetex'] = False
mpl.rcParams['axes.titlepad'] = 2.
mpl.rcParams['mathtext.default'] = 'regular'


context = Context()


@click.command()
@click.option("--model-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='data/20180511_BTSP_all_cells_PopulationAnnealing_optimization_merged_exported_output.hdf5')
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
    ramp_amp, ramp_width, peak_shift, min_val = process_BTSP_model_results(model_file_path, show_traces, export,
                                                                           output_dir, label)
    plot_BTSP_model_fit_summary(ramp_amp, ramp_width, peak_shift, min_val, export, output_dir, label)
    context.update(locals())


def process_BTSP_model_results(file_path, show=False, export=False, output_dir=None, label=None):
    """

    :param file_path: str (path)
    :param show: bool
    :param export: bool
    :param output_dir: str (dir)
    :param label: str
    :return: dict
    """
    ramp_amp, ramp_width, local_peak_shift, min_val = {}, {}, {}, {}
    with h5py.File(file_path, 'r') as f:
        shared_context_key = 'shared_context'
        group = f[shared_context_key]
        peak_locs = group['peak_locs'][:]
        binned_x = group['binned_x'][:]
        param_names = group['param_names'][:]
        exported_data_key = 'exported_data'
        for cell_key in f[exported_data_key]:
            for parameter in ramp_amp, ramp_width, local_peak_shift, min_val:
                if cell_key not in parameter:
                    parameter[cell_key] = {}
            description = 'model_ramp_features'
            group = f[exported_data_key][cell_key].itervalues().next()[description]
            param_array = group['param_array'][:]
            param_dict = param_array_to_dict(param_array, param_names)
            peak_weight = param_dict['peak_delta_weight'] + 1.

            for induction_key in f[exported_data_key][cell_key]:
                for parameter in ramp_amp, ramp_width, local_peak_shift, min_val:
                    if induction_key not in parameter[cell_key]:
                        parameter[cell_key][induction_key] = {}
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
                initial_model_ramp = group['initial_model_ramp'][:]
                target_ramp = group['target_ramp'][:]
                model_ramp = group['model_ramp'][:]
                ymin = min(ymin, np.min(model_ramp) - 1., np.min(target_ramp) - 1.)
                ymax = max(ymax, np.max(model_ramp) + 1., np.max(target_ramp) + 1.)
                if 'initial_exp_ramp' in group:
                    initial_exp_ramp = group['initial_exp_ramp'][:]
                    axes3[i][0].plot(binned_x, initial_exp_ramp, label='Before', c='darkgrey')
                    ymin = min(ymin, np.min(initial_exp_ramp) - 1.)
                    ymax = max(ymax, np.max(initial_exp_ramp) + 1.)
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
                axes3[i][2].set_ylim([-peak_weight, peak_weight * 1.1])
                axes3[i][0].hlines(ymax, xmin=mean_induction_start_loc, xmax=mean_induction_stop_loc)
                axes3[i][1].hlines(ymax, xmin=mean_induction_start_loc, xmax=mean_induction_stop_loc)
                axes3[i][2].hlines(peak_weight * 1.05, xmin=mean_induction_start_loc, xmax=mean_induction_stop_loc)
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

    return ramp_amp, ramp_width, local_peak_shift, min_val


def plot_BTSP_model_fit_summary(ramp_amp, ramp_width, peak_shift, min_val, export=False, output_dir=None,
                                         label=None):
    """

    :param ramp_amp: dict
    :param ramp_width: dict
    :param peak_shift: dict
    :param min_val: dict
    :param export: bool
    :param output_dir: str (dir)
    :param label: str
    """
    fig, axes = plt.figure(figsize=(11, 7)), []
    gs0 = gridspec.GridSpec(3, 4, wspace=0.55, hspace=0.5, left=0.075, right=0.975, top=0.95, bottom=0.075)
    axes = []
    tick_locs = [np.arange(0., 16., 3.), np.arange(30., 151., 30.), np.arange(-120., 121., 60.), np.arange(-2., 9., 2.)]
    for col, (parameter, label, tick_loc) in \
            enumerate(zip([ramp_amp, ramp_width, peak_shift, min_val],
                          ['Peak amplitude (mV)', 'Width (cm)', 'Peak shift (cm)', 'Minimum amplitude (mV)'],
                          tick_locs)):
        this_axis = fig.add_subplot(gs0[2, col])
        axes.append(this_axis)
        target_vals, model_vals = {}, {}
        for color, induction_key in zip(['darkgrey', 'r'], ['1', '2']):
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
            this_axis.legend(loc='best', frameon=False, framealpha=0.5, fontsize=mpl.rcParams['font.size'])
        this_axis.set_xlim(np.min(tick_loc), np.max(tick_loc))
        this_axis.set_ylim(np.min(tick_loc), np.max(tick_loc))
        this_axis.set_xticks(tick_loc)
        this_axis.set_yticks(tick_loc)
        this_axis.plot(tick_loc, tick_loc, c='darkgrey', alpha=0.5, ls='--')
        this_axis.set_xlabel('Experiment')
        this_axis.set_ylabel('Model')
    clean_axes(axes)
    if export:
        fig_path = '%s/%s_BTSP_model_fit_summary.svg' % (output_dir, label)
        fig.savefig(fig_path, format='svg')
    plt.show()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
