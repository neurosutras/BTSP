from biBTSP_utils import *
from nested.optimize_utils import *
import matplotlib as mpl
import click

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size'] = 11.
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['text.usetex'] = False
mpl.rcParams['axes.titlepad'] = 2.
mpl.rcParams['mathtext.default'] = 'regular'

context = Context()


@click.command()
@click.option("--data-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='data/20201123_biBTSP_data.hdf5')
@click.option("--tmax", type=float, default=4.)
@click.option("--truncate", type=float, default=2.5)
@click.option("--debug", is_flag=True)
@click.option("--cell", type=int, multiple=True, default=[1])
def main(data_file_path, tmax, truncate, debug, cell):
    """

    :param data_file_path: str (path)
    :param tmax: float
    :param truncate: float
    :param debug: bool
    :param cell: list of int
    """
    cell_keys = [str(cell_id) for cell_id in cell[:3]]
    with h5py.File(data_file_path, 'r') as f:
        for cell_key in cell_keys:
            if cell_key not in f['data'] or '2' not in f['data'][cell_key]:
                raise KeyError('plot_biBTSP_data_example_figure: problem loading data for provided cell_id: %s' %
                               cell_key)
        binned_x = f['defaults']['binned_x'][:]
        track_length = f['defaults'].attrs['track_length']

    reference_delta_t = np.linspace(-tmax* 1000., tmax * 1000., 100)

    peak_ramp_amp, total_induction_dur, group_indexes, exp_ramp, delta_exp_ramp, exp_ramp_raw, delta_exp_ramp_raw, \
    mean_induction_loc, interp_exp_ramp, interp_delta_exp_ramp, interp_delta_exp_ramp_raw, min_induction_t, \
    clean_min_induction_t, clean_induction_t_indexes, initial_induction_delta_vm = \
        get_biBTSP_data_analysis_results(data_file_path, reference_delta_t, debug=debug, truncate=truncate)

    context.update(locals())

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.25))  #, constrained_layout=True)

    ordered_interval_group_labels = ['>4 s', '2-4 s', '<=2 s']
    ordered_interval_group_colors = [('k', 'grey'), ('red', 'red'), ('green', 'green')]
    interval_group_keys = defaultdict(list)
    interp_delta_exp_raw_by_interval_group = defaultdict(list)
    exclude = [('44', '2'), ('7', '2')]
    for cell_key in exp_ramp:
        for induction_key in exp_ramp[cell_key]:
            if (cell_key, induction_key) in exclude:
                continue
            if induction_key == '1':
                min_inter_field_interval = np.nan
                interval_group_label = '>4 s'
            else:
                initial_peak_index = np.argmax(exp_ramp[cell_key][induction_key]['before'])
                min_inter_field_interval = abs(min_induction_t[cell_key][induction_key][initial_peak_index]) / 1000.
                if min_inter_field_interval <= 2.:
                    interval_group_label = '<=2 s'
                elif 2. < min_inter_field_interval <= 4.:
                    interval_group_label = '2-4 s'
                else:
                    interval_group_label = '>4 s'
            interval_group_keys[interval_group_label].append((cell_key, induction_key))
            interp_delta_exp_raw_by_interval_group[interval_group_label].append(
                interp_delta_exp_ramp_raw[cell_key][induction_key])
            print('cell: %s, induction: %s, inter-field interval: %.1f' %
                  (cell_key, induction_key, min_inter_field_interval))
    for col, interval_group_label in enumerate(ordered_interval_group_labels):
        mean_interp_delta_exp_ramp = np.nanmean(interp_delta_exp_raw_by_interval_group[interval_group_label], axis=0)
        std_interp_delta_exp_ramp = np.nanstd(interp_delta_exp_raw_by_interval_group[interval_group_label], axis=0) / \
                                    np.sqrt(len(interval_group_keys[interval_group_label]))
        for trace in interp_delta_exp_raw_by_interval_group[interval_group_label]:
            axes[2][col].plot(reference_delta_t / 1000., trace, c='grey', alpha=0.25)
        axes[2][col].plot(reference_delta_t / 1000., mean_interp_delta_exp_ramp,
                          c=ordered_interval_group_colors[col][0],
                          label='%s (n=%i)' % (interval_group_label, len(interval_group_keys[interval_group_label])))
        axes[2][col].fill_between(reference_delta_t / 1000.,
                               np.add(mean_interp_delta_exp_ramp, std_interp_delta_exp_ramp),
                               np.subtract(mean_interp_delta_exp_ramp, std_interp_delta_exp_ramp),
                               color=ordered_interval_group_colors[col][1], alpha=0.25, linewidth=0)
        axes[2][col].plot(reference_delta_t / 1000., np.zeros_like(reference_delta_t), c='darkgrey', alpha=0.75,
                          zorder=1, linestyle='--')
        axes[2][col].set_ylim((-5., 10.))
        axes[2][col].set_xlim((-tmax, tmax))
        axes[2][col].legend(loc='best', frameon=False, framealpha=0.5, fontsize=mpl.rcParams['font.size'],
                                    scatterpoints=1, handlelength=1., handletextpad=0.5)
    for key in interval_group_keys:
        print('%s: %i' % (key, len(interval_group_keys[key])))

    induction_key = '2'
    for col, cell_key in enumerate(cell_keys):
        axes[0][col].plot(binned_x, exp_ramp_raw[cell_key][induction_key]['before'], c='darkgrey',
                          label='Before Induction 2')
        axes[0][col].plot(binned_x, exp_ramp_raw[cell_key][induction_key]['after'], c='k', label='After Induction 2')
        axes[1][col].plot(reference_delta_t / 1000.,
                          interp_delta_exp_ramp_raw[cell_key][induction_key], c='k')
        axes[1][col].plot(reference_delta_t / 1000., np.zeros_like(reference_delta_t), c='darkgrey', alpha=0.75,
                          zorder=1, linestyle='--')

        this_ylim = axes[0][col].get_ylim()
        this_ymax = this_ylim[1] * 1.1
        axes[0][col].set_ylim(this_ylim[0], this_ymax)
        if col == 0:
            axes[0][col].legend(loc=(0.4, 1.05), frameon=False, framealpha=0.5, handlelength=1., handletextpad=0.5)
        axes[0][col].set_ylabel('Ramp amplitude (mV)')
        axes[0][col].set_xlabel('Position (cm)')
        axes[0][col].set_xticks(np.arange(0., track_length, 45.))
        axes[1][col].set_ylabel('Change in ramp\namplitude (mV)')
        axes[1][col].set_xlabel('Time relative to plateau onset (s)')
        axes[1][col].set_xlim(-tmax, tmax)
        axes[1][col].set_xticks([-4., -2., 0., 2., 4.])
        axes[1][col].set_yticks(np.arange(-10., 16., 5.))
        axes[0][col].scatter([mean_induction_loc[cell_key][induction_key]], [this_ymax * 0.95], color='k', marker=1)

        initial_peak_index = np.argmax(exp_ramp[cell_key][induction_key]['before'])
        min_inter_field_interval = abs(min_induction_t[cell_key][induction_key][initial_peak_index]) / 1000.

        annotation = 'Cell: %s\n' % cell_key
        annotation += r'${\Delta t}$=%.1f s' % (min_inter_field_interval)
        axes[0][col].annotate(annotation, linespacing=1.8,
                              fontsize=mpl.rcParams['font.size'], xy=(0., 1.1), xycoords='axes fraction')
        print('Cell: %s' % cell_key)
        print('Min inter-field interval: %.1f s' % (min_inter_field_interval))

        if col == 0:
            axes[0][col].legend(loc=(0.2, 0.95), frameon=False, framealpha=0., fontsize=mpl.rcParams['font.size'],
                                scatterpoints=1, handlelength=1., handletextpad=0.5)
    context.update(locals())

    clean_axes(axes)
    fig.subplots_adjust(hspace=0.6, wspace=0.66, left=0.085, right=0.945, top=0.9, bottom=0.11)
    fig.show()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
