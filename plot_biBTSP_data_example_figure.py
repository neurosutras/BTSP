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
              default='data/20190711_biBTSP_data_calibrated_input.hdf5')
@click.option("--model-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='data/20190812_biBTSP_SRL_B_90cm_all_cells_merged_exported_model_output.hdf5')
@click.option("--tmax", type=float, default=4.)
@click.option("--truncate", type=bool, default=False)
@click.option("--debug", is_flag=True)
@click.option("--cell", type=int, multiple=True, default=[1])
def main(data_file_path, model_file_path, tmax, truncate, debug, cell):
    """

    :param data_file_path: str (path)
    :param model_file_path: str (path)
    :param tmax: float
    :param truncate: bool
    :param debug: bool
    :param cell: list of int
    """
    if not os.path.isfile(data_file_path):
        raise IOError('plot_biBTSP_data_example_figure: invalid data_file_path: %s' % data_file_path)
    if not os.path.isfile(model_file_path):
        raise IOError('plot_biBTSP_data_example_figure: invalid model_file_path: %s' % model_file_path)
    cell_keys = [str(cell_id) for cell_id in cell[:3]]
    with h5py.File(data_file_path, 'r') as f:
        with h5py.File(model_file_path, 'r') as g:
            for cell_key in cell_keys:
                if cell_key not in g['exported_data'] or '2' not in g['exported_data'][cell_key]:
                    raise KeyError('plot_biBTSP_data_example_figure: problem loading data for provided cell_id: %s' %
                                   cell_key)
                if cell_key not in f['data'] or '2' not in f['data'][cell_key]:
                    raise KeyError('plot_biBTSP_data_example_figure: problem loading data for provided cell_id: %s' %
                                   cell_key)
        binned_x = f['defaults']['binned_x'][:]
        dt = f['defaults'].attrs['dt']
        track_length = f['defaults'].attrs['track_length']
    binned_extra_x = np.linspace(0., track_length, 101)
    extended_binned_x = np.concatenate([binned_x - track_length, binned_x, binned_x + track_length])
    reference_delta_t = np.linspace(-5000., 5000., 100)

    peak_ramp_amp, total_induction_dur, initial_induction_delta_vm, group_indexes, exp_ramp, extended_exp_ramp, \
    delta_exp_ramp, mean_induction_loc, extended_min_delta_t, extended_delta_exp_ramp, interp_initial_exp_ramp, \
    interp_delta_exp_ramp, interp_final_exp_ramp = \
        get_biBTSP_analysis_results(data_file_path, binned_x, binned_extra_x, extended_binned_x, reference_delta_t,
                                    track_length, dt, debug, truncate=truncate)

    context.update(locals())

    fig, axes = plt.subplots(3, 3, figsize=(12, 8.25))

    induction_key = '2'

    with h5py.File(data_file_path, 'r') as f:
        for col, cell_key in enumerate(cell_keys):
            mean_induction_start_loc = dict()
            mean_induction_stop_loc = dict()

            with h5py.File(model_file_path, 'r') as g:
                group = g['exported_data'][cell_key][induction_key]['model_ramp_features']
                mean_induction_start_loc[induction_key] = group.attrs['mean_induction_start_loc']
                mean_induction_stop_loc[induction_key] = group.attrs['mean_induction_stop_loc']

            axes[0][col].plot(binned_x, exp_ramp[cell_key][induction_key]['before'], c='darkgrey',
                              label='Before Induction 2')
            axes[0][col].plot(binned_x, exp_ramp[cell_key][induction_key]['after'], c='k', label='After Induction 2')
            indexes = np.where(~np.isnan(interp_delta_exp_ramp[cell_key][induction_key]))[0]
            axes[1][col].plot(reference_delta_t[indexes] / 1000.,
                              interp_delta_exp_ramp[cell_key][induction_key][indexes], c='k')
            axes[1][col].plot(reference_delta_t[indexes] / 1000., np.zeros_like(indexes), c='darkgrey', alpha=0.75,
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
            axes[1][col].set_xlim(-4., 4.)
            axes[1][col].set_xticks([-4., -2., 0., 2., 4.])
            axes[1][col].set_yticks(np.arange(-10., 16., 5.))
            # axes[1][col].hlines(this_ymax * 0.95, xmin=mean_induction_start_loc[induction_key],
            #                    xmax=mean_induction_stop_loc[induction_key], color='k')
            axes[0][col].scatter([mean_induction_start_loc[induction_key]], [this_ymax * 0.95], color='k', marker=1)

            this_complete_position = f['data'][cell_key][induction_key]['complete']['position'][:]
            this_complete_t = f['data'][cell_key][induction_key]['complete']['t'][:]
            with h5py.File(model_file_path, 'r') as g:
                induction_start_times = g['exported_data'][cell_key][induction_key]['model_ramp_features'].attrs[
                    'induction_start_times']
            induction_start_indexes = []
            for this_induction_start_time in induction_start_times:
                index = np.where(this_complete_t >= this_induction_start_time)[0][0]
                induction_start_indexes.append(index)
            pretty_position = np.array(this_complete_position % track_length)
            end_index = len(pretty_position) - 1
            initial_peak_index = np.argmax(exp_ramp[cell_key][induction_key]['before'])
            initial_peak_loc = binned_x[initial_peak_index]
            axes[0][col].scatter([initial_peak_loc], [this_ymax * 0.95], color='darkgrey', marker='_')
            initial_peak_indexes = []
            passed_peak = False
            for i in range(1, len(pretty_position)):
                if not passed_peak and pretty_position[i] >= initial_peak_loc:
                    initial_peak_indexes.append(i)
                    passed_peak = True
                if pretty_position[i] - pretty_position[i - 1] < -track_length / 2.:
                    pretty_position[i] = np.nan
                    end_index = i
                    passed_peak = False

            start_indexes = np.where(this_complete_t[initial_peak_indexes] <
                                     this_complete_t[induction_start_indexes][0])[0]
            if len(start_indexes) > 0:
                start_index = start_indexes[-1]
            else:
                start_index = 0
            end_indexes = np.where(this_complete_t[initial_peak_indexes] >
                                   this_complete_t[induction_start_indexes][-1])[0]
            if len(end_indexes) > 0:
                end_index = end_indexes[0] + 1
            else:
                end_index = len(initial_peak_indexes)

            initial_peak_times = this_complete_t[initial_peak_indexes][start_index:end_index]
            forward_intervals = []
            backward_intervals = []
            for event in this_complete_t[induction_start_indexes]:
                this_intervals = np.subtract(event, initial_peak_times)
                this_backward_interval = np.min(this_intervals[np.where(this_intervals > 0.)[0]])
                backward_intervals.append(this_backward_interval)
                this_forward_interval = -np.max(this_intervals[np.where(this_intervals < 0.)[0]])
                forward_intervals.append(this_forward_interval)
            mean_event_interval = min(np.mean(backward_intervals), np.mean(forward_intervals))
            annotation = 'Cell: %s\n' % cell_key
            annotation += r'${\Delta t}$=%.1f s' % (mean_event_interval / 1000.)
            axes[0][col].annotate(annotation, linespacing=1.8,
                                  fontsize=mpl.rcParams['font.size'], xy=(0., 1.1), xycoords='axes fraction')
            print('Cell: %s' % cell_key)
            print('Mean inter-field interval: %.1f s' % (mean_event_interval / 1000.))

            axes[2][col].plot(this_complete_t / 1000., pretty_position, c='darkgrey', zorder=1)
            xmax = max(np.max(initial_peak_times), np.max(this_complete_t[induction_start_indexes])) / 1000.
            axes[2][col].set_xlim(-5., xmax + 5.)
            axes[2][col].scatter(this_complete_t[initial_peak_indexes] / 1000.,
                                 pretty_position[initial_peak_indexes], c='darkgrey', s=40, linewidth=0, zorder=2,
                                 label='Initial peak location', edgecolor='none')
            axes[2][col].scatter(this_complete_t[induction_start_indexes] / 1000.,
                                 pretty_position[induction_start_indexes], c='k', s=40, linewidth=0, zorder=2,
                                 label='Induction 2 location', edgecolor='none')
            axes[2][col].set_yticks(np.arange(0., track_length, 60.))
            axes[2][col].set_ylabel('Position (cm)')
            axes[2][col].set_xlabel('Time (s)')
            if col == 0:
                axes[2][col].legend(loc=(0.2, 0.95), frameon=False, framealpha=0., fontsize=mpl.rcParams['font.size'],
                                    scatterpoints=1, handlelength=1., handletextpad=0.5)
    context.update(locals())

    clean_axes(axes)
    fig.subplots_adjust(hspace=0.6, wspace=0.66, left=0.085, right=0.945, top=0.9, bottom=0.11)
    fig.show()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
