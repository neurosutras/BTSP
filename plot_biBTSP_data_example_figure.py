from biBTSP_utils import *
from nested.optimize_utils import *
import matplotlib as mpl
import matplotlib.gridspec as gridspec
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
                if cell_key not in g['exported_data'] or '1' not in g['exported_data'][cell_key] or \
                        '2' not in g['exported_data'][cell_key]:
                    raise KeyError('plot_biBTSP_data_example_figure: problem loading data for provided cell_id: %s' %
                                   cell_key)
                if cell_key not in f['data'] or '1' not in f['data'][cell_key] or \
                        '2' not in f['data'][cell_key]:
                    raise KeyError('plot_biBTSP_data_example_figure: problem loading data for provided cell_id: %s' %
                                   cell_key)
        binned_x = f['defaults']['binned_x'][:]
        dt = f['defaults'].attrs['dt']
        track_length = f['defaults'].attrs['track_length']
    binned_extra_x = np.linspace(0., track_length, 101)
    extended_binned_x = np.concatenate([binned_x - track_length, binned_x, binned_x + track_length])
    reference_delta_t = np.linspace(-5000., 5000., 100)

    peak_ramp_amp, total_induction_dur, depo_soma, group_indexes, exp_ramp, extended_exp_ramp, delta_exp_ramp, \
    mean_induction_loc, extended_min_delta_t, extended_delta_exp_ramp, interp_initial_exp_ramp, \
    interp_delta_exp_ramp, interp_final_exp_ramp = \
        get_biBTSP_analysis_results(data_file_path, binned_x, binned_extra_x, extended_binned_x, reference_delta_t,
                                    track_length, dt, debug, truncate=truncate)

    context.update(locals())

    fig, axes = plt.figure(figsize=(11, 7.5)), []
    gs0 = gridspec.GridSpec(3, 7, wspace=3.9, hspace=0.8, left=0.075, right=0.95)
    for row in range(len(cell_keys)):
        this_axes_row = []
        this_axes_row.append(fig.add_subplot(gs0[row, :3]))
        this_axes_row.append(fig.add_subplot(gs0[row, 3:5]))
        this_axes_row.append(fig.add_subplot(gs0[row, 5:]))
        axes.append(this_axes_row)

    with h5py.File(data_file_path, 'r') as f:
        for row, cell_key in enumerate(cell_keys):
            mean_induction_start_loc = dict()
            mean_induction_stop_loc = dict()
            track_start_times = dict()
            track_stop_times = dict()
            for induction_key in exp_ramp[cell_key]:

                with h5py.File(model_file_path, 'r') as g:
                    group = g['exported_data'][cell_key][induction_key]['model_ramp_features']
                    mean_induction_start_loc[induction_key] = group.attrs['mean_induction_start_loc']
                    mean_induction_stop_loc[induction_key] = group.attrs['mean_induction_stop_loc']
                    track_start_times[induction_key] = group.attrs['track_start_times']
                    track_stop_times[induction_key] = group.attrs['track_stop_times']
                if induction_key == '1':
                    color = 'darkgrey'
                    label = 'Before induction 2'
                else:
                    color = 'k'
                    label = 'After induction 2'
                axes[row][1].plot(binned_x, exp_ramp[cell_key][induction_key]['after'], c=color, label=label)
                indexes = np.where(~np.isnan(interp_delta_exp_ramp[cell_key][induction_key]))[0]
                axes[row][2].plot(reference_delta_t[indexes] / 1000.,
                                  interp_delta_exp_ramp[cell_key][induction_key][indexes], c=color)

            this_ylim = axes[row][1].get_ylim()
            this_ymax = this_ylim[1] * 1.1
            axes[row][1].set_ylim(this_ylim[0], this_ymax)
            if row == 0:
                axes[row][1].legend(loc=(0.1, 0.95), frameon=False, framealpha=0.5, handlelength=1., handletextpad=0.5)
            axes[row][1].set_ylabel('Ramp amplitude (mV)')
            axes[row][1].set_xlabel('Position (cm)')
            axes[row][1].set_xticks(np.arange(0., track_length, 45.))
            axes[row][2].set_ylabel('Change in ramp\namplitude (mV)')
            axes[row][2].set_xlabel('Time relative to plateau onset (s)')
            axes[row][2].set_xlim(-4., 4.)
            axes[row][2].set_xticks([-4., -2., 0., 2., 4.])
            axes[row][2].set_yticks(np.arange(-10., 16., 5.))
            for induction_key in mean_induction_start_loc:
                if induction_key == '1':
                    color = 'darkgrey'
                else:
                    color = 'k'
                axes[row][1].hlines(this_ymax * 0.95, xmin=mean_induction_start_loc[induction_key],
                                    xmax=mean_induction_stop_loc[induction_key], color=color)
            induction_key = '2'
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

            event_times = np.append(this_complete_t[initial_peak_indexes][start_index:end_index],
                                    this_complete_t[induction_start_indexes])
            event_times = np.sort(event_times)
            event_intervals = np.diff(event_times)
            mean_event_interval = np.mean(event_intervals)

            print('Cell: %s' % cell_key)
            print('Event intervals: %s' % str(event_intervals))
            print('Mean event interval: %.1f s' % (mean_event_interval / 1000.))

            axes[row][0].plot(this_complete_t / 1000., pretty_position, c='darkgrey', zorder=1)
            xmax = max(event_times) / 1000.
            axes[row][0].set_xlim(-5., xmax + 5.)
            axes[row][0].scatter(this_complete_t[initial_peak_indexes] / 1000.,
                                 pretty_position[initial_peak_indexes], c='darkgrey', s=40, linewidth=0, zorder=2,
                                 label='Initial peak location', edgecolor='none')
            axes[row][0].scatter(this_complete_t[induction_start_indexes] / 1000.,
                                 pretty_position[induction_start_indexes], c='k', s=40, linewidth=0, zorder=2,
                                 label='Induction 2 location', edgecolor='none')
            axes[row][0].set_yticks(np.arange(0., track_length, 60.))
            axes[row][0].set_ylabel('Position (cm)')
            axes[row][0].set_xlabel('Time (s)')
            axes[row][0].set_title(r'Cell: %s' % cell_key, fontsize=mpl.rcParams['font.size'], loc='left', y=1.1)
            axes[row][0].annotate(r'$\overline {\Delta t}$=%.1f s' % (mean_event_interval / 1000.),
                                  fontsize=mpl.rcParams['font.size'], xy=(0.8, 1.1), xycoords='axes fraction')
            if row == 0:
                axes[row][0].legend(loc=(0.2, 0.95), frameon=False, framealpha=0., fontsize=mpl.rcParams['font.size'],
                                    scatterpoints=1, handlelength=1., handletextpad=0.5)
    context.update(locals())

    clean_axes(np.array(axes))
    fig.show()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
