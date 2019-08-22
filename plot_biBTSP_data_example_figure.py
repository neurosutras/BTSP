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
@click.option("--cell", type=int, multiple=True, default=[1])
def main(data_file_path, model_file_path, cell):
    """

    :param data_file_path: str (path)
    :param model_file_path: str (path)
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
    context.update(locals())

    exp_ramp = {}
    extended_exp_ramp = {}
    delta_exp_ramp = {}
    mean_induction_loc = {}
    extended_min_delta_t = {}
    extended_delta_exp_ramp = {}

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
            for induction_key in f['data'][cell_key]:
                induction_locs = f['data'][cell_key][induction_key].attrs['induction_locs']
                induction_durs = f['data'][cell_key][induction_key].attrs['induction_durs']

                if cell_key not in delta_exp_ramp:
                    exp_ramp[cell_key] = {}
                    extended_exp_ramp[cell_key] = {}
                    delta_exp_ramp[cell_key] = {}
                    mean_induction_loc[cell_key] = {}
                    extended_min_delta_t[cell_key] = {}
                    extended_delta_exp_ramp[cell_key] = {}
                if induction_key not in exp_ramp[cell_key]:
                    exp_ramp[cell_key][induction_key] = {}
                    extended_exp_ramp[cell_key][induction_key] = {}
                if induction_key == '1' and '2' in f['data'][cell_key]:
                    exp_ramp[cell_key][induction_key]['after'] = \
                        f['data'][cell_key]['2']['processed']['exp_ramp']['before'][:]
                else:
                    exp_ramp[cell_key][induction_key]['after'] = \
                        f['data'][cell_key][induction_key]['processed']['exp_ramp']['after'][:]

                if 'before' in f['data'][cell_key][induction_key]['processed']['exp_ramp']:
                    exp_ramp[cell_key][induction_key]['before'] = \
                        f['data'][cell_key][induction_key]['processed']['exp_ramp']['before'][:]
                    delta_exp_ramp[cell_key][induction_key] = np.subtract(exp_ramp[cell_key][induction_key]['after'],
                                                                          exp_ramp[cell_key][induction_key]['before'])
                else:
                    delta_exp_ramp[cell_key][induction_key], discard = \
                        subtract_baseline(exp_ramp[cell_key][induction_key]['after'])

                mean_induction_loc[cell_key][induction_key] = np.mean(induction_locs)
                extended_delta_exp_ramp[cell_key][induction_key] = \
                    np.concatenate([delta_exp_ramp[cell_key][induction_key]] * 3)
                for category in exp_ramp[cell_key][induction_key]:
                    extended_exp_ramp[cell_key][induction_key][category] = \
                        np.concatenate([exp_ramp[cell_key][induction_key][category]] * 3)
                backward_t = np.empty_like(binned_extra_x)
                backward_t[:] = np.nan
                forward_t = np.array(backward_t)
                for i in range(len(induction_locs)):
                    this_induction_loc = mean_induction_loc[cell_key][induction_key]
                    key = str(i)
                    this_position = f['data'][cell_key][induction_key]['processed']['position']['induction'][key][:]
                    this_t = f['data'][cell_key][induction_key]['processed']['t']['induction'][key][:]
                    this_induction_index = np.where(this_position >= this_induction_loc)[0][0]
                    this_induction_t = this_t[this_induction_index]
                    if i == 0 and 'pre' in f['data'][cell_key][induction_key]['raw']['position']:
                        pre_position = f['data'][cell_key][induction_key]['processed']['position']['pre']['0'][:]
                        pre_t = f['data'][cell_key][induction_key]['processed']['t']['pre']['0'][:]
                        pre_t -= len(pre_t) * dt
                        pre_t -= this_induction_t
                        backward_t, forward_t = update_min_t_arrays(binned_extra_x, pre_t, pre_position, backward_t,
                                                                    forward_t)
                    elif i > 0:
                        prev_t -= len(prev_t) * dt
                        prev_induction_t = prev_t[prev_induction_index]
                        backward_t, forward_t = update_min_t_arrays(binned_extra_x, np.subtract(prev_t, this_induction_t),
                                                                    prev_position, backward_t, forward_t)
                        backward_t, forward_t = update_min_t_arrays(binned_extra_x, np.subtract(this_t, prev_induction_t),
                                                                    this_position, backward_t, forward_t)
                    backward_t, forward_t = update_min_t_arrays(binned_extra_x, np.subtract(this_t, this_induction_t),
                                                                this_position, backward_t, forward_t)
                    if i == len(induction_locs) - 1 and 'post' in f['data'][cell_key][induction_key]['raw']['position']:
                        post_position = f['data'][cell_key][induction_key]['processed']['position']['post']['0'][:]
                        post_t = f['data'][cell_key][induction_key]['processed']['t']['post']['0'][:]
                        post_t += len(this_t) * dt
                        post_t -= this_induction_t
                        backward_t, forward_t = update_min_t_arrays(binned_extra_x, post_t, post_position, backward_t,
                                                                    forward_t)
                    prev_key = key
                    prev_induction_index = this_induction_index
                    prev_t = this_t
                    prev_position = this_position
                extended_min_delta_t[cell_key][induction_key] = \
                    merge_min_t_arrays(binned_x, binned_extra_x, extended_binned_x, mean_induction_loc[cell_key][induction_key], backward_t, forward_t)
                this_extended_delta_position = np.subtract(extended_binned_x,
                                                           mean_induction_loc[cell_key][induction_key])

                indexes = np.where((this_extended_delta_position > -track_length) &
                                   (this_extended_delta_position < track_length))[0]
                mask = ~np.isnan(extended_min_delta_t[cell_key][induction_key])
                indexes = np.where((extended_min_delta_t[cell_key][induction_key][mask] >= -5000.) &
                                   (extended_min_delta_t[cell_key][induction_key][mask] <= 5000.))[0]
                this_interp_delta_exp_ramp = np.interp(reference_delta_t,
                                                       extended_min_delta_t[cell_key][induction_key][mask][indexes],
                                                       extended_delta_exp_ramp[cell_key][induction_key][mask][indexes])

                with h5py.File(model_file_path, 'r') as g:
                    group = g['exported_data'][cell_key][induction_key]['model_ramp_features']
                    mean_induction_start_loc[induction_key] = group.attrs['mean_induction_start_loc']
                    mean_induction_stop_loc[induction_key] = group.attrs['mean_induction_stop_loc']
                if induction_key == '1':
                    color = 'darkgrey'
                    label = 'After induction 1'
                else:
                    color = 'k'
                    label = 'After induction 2'
                axes[row][1].plot(binned_x, exp_ramp[cell_key][induction_key]['after'], c=color, label=label)
                axes[row][2].plot(reference_delta_t / 1000., this_interp_delta_exp_ramp, c=color)

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
            for i in range(1, len(pretty_position)):
                if pretty_position[i] - pretty_position[i - 1] < -track_length / 2.:
                    pretty_position[i] = np.nan
                    end_index = i
            axes[row][0].plot(this_complete_t / 1000., pretty_position, c='darkgrey')
            xmax = this_complete_t[end_index] / 1000.
            axes[row][0].set_xlim(-5., xmax + 5.)
            axes[row][0].scatter(this_complete_t[induction_start_indexes] / 1000.,
                                 pretty_position[induction_start_indexes], c='k', s=40, linewidth=0, zorder=1)
            axes[row][0].set_yticks(np.arange(0., track_length, 60.))
            axes[row][0].set_ylabel('Position (cm)')
            axes[row][0].set_xlabel('Time (s)')
            axes[row][0].set_title('Cell: %s' % cell_key, fontsize=mpl.rcParams['font.size'], loc='left', y=1.1)
    context.update(locals())

    clean_axes(np.array(axes))
    fig.show()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
