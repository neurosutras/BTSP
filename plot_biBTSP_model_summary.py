__author__ = 'milsteina'
from nested.optimize_utils import *
import matplotlib as mpl
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
              default='data/20201112_biBTSP_WD_D_90cm_exported_model_output.hdf5')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export", is_flag=True)
@click.option("--show-traces", is_flag=True)
@click.option("--label", type=str, default=None)
@click.option("--exported-data-key", type=str, default='0')
def main(model_file_path, output_dir, export, show_traces, label, exported_data_key):
    """

    :param model_file_path: str (path)
    :param output_dir: str (dir)
    :param export: bool
    :param show_traces: bool
    :param label: str
    :param exported_data_key: str
    """
    date_stamp = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    if label is None:
        label = date_stamp
    if not os.path.isfile(model_file_path):
        raise IOError('Invalid model_file_path: %s' % model_file_path)
    if export and not os.path.isdir(output_dir):
        raise IOError('Invalid output_dir: %s' % output_dir)

    ramp_amp, ramp_width, peak_shift, min_val, ramp_mse_dict, ramp_sse_dict, delta_exp_ramp, delta_model_ramp = \
        process_biBTSP_model_results(model_file_path, show_traces, export, output_dir, label, exported_data_key)

    plot_biBTSP_model_fit_summary(ramp_amp, ramp_width, peak_shift, min_val, ramp_mse_dict, ramp_sse_dict, export,
                                  output_dir, label)
    context.update(locals())


def process_biBTSP_model_results(file_path, show=False, export=False, output_dir=None, label=None,
                                 exported_data_key=None):
    """

    :param file_path: str (path)
    :param show: bool
    :param export: bool
    :param output_dir: str (dir)
    :param label: str
    :param exported_data_key: str
    :return: tuple of dict
    """
    ramp_amp = defaultdict(lambda: defaultdict(dict))
    ramp_width = defaultdict(lambda: defaultdict(dict))
    local_peak_shift = defaultdict(lambda: defaultdict(dict))
    min_val = defaultdict(lambda: defaultdict(dict))
    ramp_mse_dict = defaultdict(list)
    ramp_sse_dict = defaultdict(list)
    max_delta_ramp_val = defaultdict(lambda: defaultdict(list))
    min_delta_ramp_val = defaultdict(lambda: defaultdict(list))
    with h5py.File(file_path, 'r') as f:
        shared_context_key = 'shared_context'
        group = f[shared_context_key]
        peak_locs = group['peak_locs'][:]
        binned_x = group['binned_x'][:]
        param_names = group['param_names'][:]
        source = get_h5py_group(f, [exported_data_key, 'exported_data'])
        for cell_key in source:
            description = 'model_ramp_features'
            group = next(iter(source[cell_key].values()))[description]
            param_array = group['param_array'][:]
            param_dict = param_array_to_dict(param_array, param_names)
            if 'peak_delta_weight' in param_dict:
                peak_delta_weight = param_dict['peak_delta_weight']
            else:
                peak_delta_weight = 1.

            for induction_key in source[cell_key]:
                description = 'model_ramp_features'
                group = source[cell_key][induction_key][description]
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
            for i, induction_key in enumerate(list(source[cell_key].keys())):
                description = 'model_ramp_features'
                group = source[cell_key][induction_key][description]
                model_weights = group['model_weights'][:]
                initial_weights = group['initial_weights'][:]
                delta_weights = np.subtract(model_weights, initial_weights)
                peak_delta_weight = max([max(delta_weights), abs(min(delta_weights)), peak_delta_weight])
                initial_model_ramp = group['initial_model_ramp'][:]
                target_ramp = group['target_ramp'][:]
                model_ramp = group['model_ramp'][:]

                this_mse_array = np.square(np.subtract(target_ramp, model_ramp))
                ramp_mse_dict[induction_key].append(np.mean(this_mse_array))
                ramp_sse_dict[induction_key].append(np.sum(this_mse_array))

                ymin = min(ymin, np.min(model_ramp) - 1., np.min(target_ramp) - 1.)
                ymax = max(ymax, np.max(model_ramp) + 1., np.max(target_ramp) + 1.)
                if 'initial_exp_ramp' in group:
                    initial_exp_ramp = group['initial_exp_ramp'][:]
                    axes3[i][0].plot(binned_x, initial_exp_ramp, label='Before', c='darkgrey')
                    ymin = min(ymin, np.min(initial_exp_ramp) - 1.)
                    ymax = max(ymax, np.max(initial_exp_ramp) + 1.)
                else:
                    initial_exp_ramp = np.zeros_like(target_ramp)
                delta_model_ramp = np.subtract(model_ramp, initial_exp_ramp)
                delta_exp_ramp = np.subtract(target_ramp, initial_exp_ramp)
                max_delta_ramp_val[induction_key]['target'].append(np.max(delta_exp_ramp))
                max_delta_ramp_val[induction_key]['model'].append(np.max(delta_model_ramp))
                min_delta_ramp_val[induction_key]['target'].append(np.min(delta_exp_ramp))
                min_delta_ramp_val[induction_key]['model'].append(np.min(delta_model_ramp))

                axes3[i][0].plot(binned_x, target_ramp, label='After', c='r')
                axes3[i][0].set_title('Induction %s\nExperiment Vm:' % (induction_key),
                                      fontsize=mpl.rcParams['font.size'])
                axes3[i][1].plot(binned_x, initial_model_ramp, label='Before', c='darkgrey')
                axes3[i][1].plot(binned_x, model_ramp, label='After', c='k')
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

            for i, induction_key in enumerate(list(source[cell_key].keys())):
                description = 'model_ramp_features'
                group = source[cell_key][induction_key][description]
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
                fig_path = '%s/%s_biBTSP_cell_%s_ramp_traces.svg' % (output_dir, label, cell_key)
                fig3.savefig(fig_path, format='svg')
    if show:
        plt.show()
    else:
        plt.close('all')

    return ramp_amp, ramp_width, local_peak_shift, min_val, ramp_mse_dict, ramp_sse_dict, max_delta_ramp_val, \
           min_delta_ramp_val


def plot_biBTSP_model_fit_summary(ramp_amp, ramp_width, peak_shift, min_val, ramp_mse_dict, ramp_sse_dict,
                                  export=False, output_dir=None, label=None):
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
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.))

    tick_locs = [np.arange(0., 16., 3.), np.arange(60., 181., 30.), np.arange(-60., 61., 30.), np.arange(-2., 9., 2.)]
    for i, (parameter, param_label, tick_loc) in \
            enumerate(zip([ramp_amp, ramp_width, peak_shift, min_val],
                          ['Peak amplitude (mV)', 'Width (cm)', 'Peak shift (cm)', 'Minimum amplitude (mV)'],
                          tick_locs)):
        this_axis = axes[i]
        target_vals, model_vals = {}, {}
        for color, induction_key in zip(['c', 'r'], ['1', '2']):
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
        this_axis.set_title(param_label, fontsize=mpl.rcParams['font.size'], y=1.05)
        if i == 0:
            this_axis.legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, handletextpad=0.5,
                             fontsize=mpl.rcParams['font.size'])
        this_axis.set_xlim(np.min(tick_loc), np.max(tick_loc))
        this_axis.set_ylim(np.min(tick_loc), np.max(tick_loc))
        this_axis.set_xticks(tick_loc)
        this_axis.set_yticks(tick_loc)
        this_axis.plot(tick_loc, tick_loc, c='darkgrey', alpha=0.5, ls='--')
        this_axis.set_xlabel('Experiment')
        this_axis.set_ylabel('Model')

    clean_axes(axes)
    fig.subplots_adjust(left=0.05, wspace=0.675, right=0.975, bottom=0.2, top=0.75)
    fig.suptitle('%s: ' % label, x=0.01, y=0.975, ha='left', fontsize=mpl.rcParams['font.size'])
    # fig.show()
    # fig.suptitle('%s: ' % label, fontsize=mpl.rcParams['font.size'], ha='left', x=0.)
    if export:
        fig_path = '%s/%s_biBTSP_model_fit_summary.svg' % (output_dir, label)
        fig.savefig(fig_path, format='svg')

    fig2, axes2 = plt.subplots(2, 2, figsize=(8.25, 6))
    for i, (xlabel, vals) in enumerate(zip(['Mean squared error', 'Sum of squared error'],
                                           [ramp_mse_dict, ramp_sse_dict])):
        this_axis = axes2[0][i]
        max_val = 0.
        for color, induction_key in zip(['darkgrey', 'k'], ['1', '2']):
            vals[induction_key].sort()
            max_val = max(max_val, np.max(vals[induction_key]))
            n = len(vals[induction_key])
            this_axis.plot(vals[induction_key], np.add(np.arange(n), 1.) / float(n),
                           label='Induction: %s' % induction_key, c=color)
        this_axis.set_ylim(0., 1.05)
        this_axis.set_ylabel('Cum. fraction')
        this_axis.set_xlim(0., math.ceil(max_val))
        int_max_val = int(math.ceil(max_val))
        int_delta_val = max(1, int_max_val // 6)
        # this_axis.set_xticks(np.arange(0, int_max_val+1, int_delta_val))
        this_axis.set_xlabel(xlabel)
        this_axis.set_title('Model ramp residual error', fontsize=mpl.rcParams['font.size'])
        this_axis.legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, handletextpad=0.5,
                         fontsize=mpl.rcParams['font.size'])
    clean_axes(axes2)
    fig2.subplots_adjust(left=0.125, hspace=0.5, wspace=0.6, right=0.925)
    fig2.suptitle('%s: ' % label, x=0.05, y=0.95, ha='left', fontsize=mpl.rcParams['font.size'])

    plt.show()


def plot_compare_models_mse_by_induction(model_file_path_dict):
    """

    :param model_file_path_dict: {label (str): file_path (str; path)}
    """
    all_mse = dict()
    max_val = defaultdict(lambda: 0.)

    for model, model_file_path in model_file_path_dict.items():
        ramp_amp, ramp_width, peak_shift, min_val, all_mse[model], discard_ramp_mse_array, discard_ramp_sse_array, \
        max_delta_ramp_val, min_delta_ramp_val = process_biBTSP_model_results(model_file_path)
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


def plot_compare_models_boxplot(model_file_path_list, label_list, exported_data_key_list=None, ymax=5., control=None):
    """

    :param model_file_path_list: list of str (path)
    :param label_list: list of str
    :param exported_data_key_list: list of str
    :param ymax: float
    :param control: str (labeled data to act as control for statistical comparisons)
    :return dict
    """
    if exported_data_key_list is None:
        exported_data_key_list = ['0'] * len(label_list)
    mse_data = []
    sse_data = []
    mse_dict = {}
    for model_file_path, label, exported_data_key in zip(model_file_path_list, label_list, exported_data_key_list):
        ramp_amp, ramp_width, peak_shift, min_val, this_mse_dict, this_sse_dict, max_delta_ramp_val, \
        min_delta_ramp_val = process_biBTSP_model_results(model_file_path, exported_data_key=exported_data_key)
        this_sse_list = []
        this_mse_list = []
        for induction_key in ['2', '3']:  # this_sse_dict:
            this_sse_list.extend(this_sse_dict[induction_key])
            this_mse_list.extend(this_mse_dict[induction_key])
        sse_data.append(this_sse_list)
        mse_data.append(this_mse_list)
        mse_dict[label] = this_mse_list

    fig, axes = plt.subplots(figsize=(7., 2.9))
    axes.boxplot(mse_data, showfliers=False, medianprops=dict(color='k'), widths=0.7)
    axes.set_xticklabels(label_list)
    axes.set_ylabel('Mean squared error')
    axes.set_ylim((min(0, axes.get_ylim()[0]), max(axes.get_ylim()[1], ymax)))
    clean_axes(axes)
    fig.suptitle('Model ramp residual error', y=0.95, x=0.05, ha='left', fontsize=mpl.rcParams['font.size'])
    fig.subplots_adjust(top=0.85, hspace=0.2, wspace=0.6, bottom=0.15)
    fig.show()

    if control is not None:
        from scipy.stats import friedmanchisquare
        from scipy.stats import wilcoxon
        stats, p = friedmanchisquare(*list(mse_dict.values()))
        print('Friedman test: p = %.5f' % p)
        if p < 0.05:
            for label in [label for label in label_list if label != control]:
                stat, p = wilcoxon(mse_dict[control], mse_dict[label])
                p *= len(label_list) - 1
                print('Wilcoxon signed rank: %s vs %s; p = %.5f' % (control, label, p))

    return mse_dict


def plot_compare_model_delta_ramp_extrema(model_file_path_list, label_list, exported_data_key_list=None, control=None):
    """

    :param model_file_path_list: list of str (path)
    :param label_list: list of str
    :param exported_data_key_list: list of str
    :param control: str (labeled data to act as control for statistical comparisons)
    :return dict
    """
    if exported_data_key_list is None:
        exported_data_key_list = ['0'] * len(label_list)
    max_delta_ramp_data = []
    min_delta_ramp_data = []
    max_delta_ramp_dict = {}
    min_delta_ramp_dict = {}
    for model_file_path, label, exported_data_key in zip(model_file_path_list, label_list, exported_data_key_list):
        ramp_amp, ramp_width, peak_shift, min_val, this_mse_dict, this_sse_dict, max_delta_ramp_val, \
        min_delta_ramp_val = process_biBTSP_model_results(model_file_path, exported_data_key=exported_data_key)
        if 'Exp' not in max_delta_ramp_dict:
            this_max_delta_ramp_list = []
            this_min_delta_ramp_list = []
            for induction_key in ['2', '3']:
                this_max_delta_ramp_list.extend(max_delta_ramp_val[induction_key]['target'])
                this_min_delta_ramp_list.extend(min_delta_ramp_val[induction_key]['target'])
            max_delta_ramp_data.append(this_max_delta_ramp_list)
            min_delta_ramp_data.append(this_min_delta_ramp_list)
            max_delta_ramp_dict['Exp'] = this_max_delta_ramp_list
            min_delta_ramp_dict['Exp'] = this_min_delta_ramp_list

        this_max_delta_ramp_list = []
        this_min_delta_ramp_list = []
        for induction_key in ['2', '3']:
            this_max_delta_ramp_list.extend(max_delta_ramp_val[induction_key]['model'])
            this_min_delta_ramp_list.extend(min_delta_ramp_val[induction_key]['model'])
        max_delta_ramp_data.append(this_max_delta_ramp_list)
        min_delta_ramp_data.append(this_min_delta_ramp_list)
        max_delta_ramp_dict[label] = this_max_delta_ramp_list
        min_delta_ramp_dict[label] = this_min_delta_ramp_list

    working_label_list = ['Exp'] + label_list
    fig, axes = plt.subplots(2, figsize=(12, 7))
    axes[0].boxplot(max_delta_ramp_data, showfliers=False, medianprops=dict(color='k'), widths=0.7)
    axes[0].set_xticklabels(working_label_list)
    axes[0].set_ylabel(r'Max $\Delta$Vm (mV)')
    axes[0].set_ylim((min(0., axes[0].get_ylim()[0]), axes[0].get_ylim()[1]))
    axes[1].boxplot(min_delta_ramp_data, showfliers=False, medianprops=dict(color='k'), widths=0.7)
    axes[1].set_xticklabels(working_label_list)
    axes[1].set_ylabel(r'Min $\Delta$Vm (mV)')
    axes[1].set_ylim((axes[1].get_ylim()[0], max(0., axes[1].get_ylim()[1])))
    clean_axes(axes)
    # fig.suptitle('Model ramp residual error', y=0.95, x=0.05, ha='left', fontsize=mpl.rcParams['font.size'])
    fig.subplots_adjust(hspace=0.35, wspace=0.6, bottom=0.15)
    fig.show()

    from scipy.stats import friedmanchisquare
    from scipy.stats import wilcoxon

    num_comparisons = len(label_list)
    if control is not None and len(label_list) > 1:
        num_comparisons += len(label_list) - 1

    print('Max delta ramp:')
    stats, p = friedmanchisquare(*max_delta_ramp_data)
    print('Friedman test: p = %.5f' % p)
    if p < 0.05:
        for label in label_list:
            stat, p = wilcoxon(max_delta_ramp_dict['Exp'], max_delta_ramp_dict[label])
            p *= num_comparisons
            print('Wilcoxon signed rank: %s vs %s; p = %.5f' % ('Exp', label, p))
        if control is not None:
            for label in [label for label in label_list if label != control]:
                stat, p = wilcoxon(max_delta_ramp_dict[control], max_delta_ramp_dict[label])
                p *= num_comparisons
                print('Wilcoxon signed rank: %s vs %s; p = %.5f' % (control, label, p))

    print('Min delta ramp:')
    stats, p = friedmanchisquare(*min_delta_ramp_data)
    print('Friedman test: p = %.5f' % p)
    if p < 0.05:
        for label in label_list:
            stat, p = wilcoxon(min_delta_ramp_dict['Exp'], min_delta_ramp_dict[label])
            p *= num_comparisons
            print('Wilcoxon signed rank: %s vs %s; p = %.5f' % ('Exp', label, p))
        if control is not None:
            for label in [label for label in label_list if label != control]:
                stat, p = wilcoxon(min_delta_ramp_dict[control], min_delta_ramp_dict[label])
                p *= num_comparisons
                print('Wilcoxon signed rank: %s vs %s; p = %.5f' % (control, label, p))

    return max_delta_ramp_dict, min_delta_ramp_dict


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
