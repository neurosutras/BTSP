from BTSP_utils import *
from nested.optimize_utils import *
from scipy.stats import pearsonr
from matplotlib import cm
import matplotlib.lines as mlines
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import click
from scipy.ndimage.filters import gaussian_filter


mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size'] = 11.
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['text.usetex'] = False
mpl.rcParams['axes.titlepad'] = 2.
mpl.rcParams['mathtext.default'] = 'regular'

context = Context()


def set_axes_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def update_min_t_arrays(context, t, x, backward_t, forward_t):
    """

    :param t:
    :param x:
    :param backward_t:
    :param forward_t:
    :return array, array
    """
    binned_extra_x = context.binned_extra_x
    binned_t = np.interp(binned_extra_x, x, t)
    temp_backward_t = np.empty_like(binned_extra_x)
    temp_backward_t[:] = np.nan
    temp_forward_t = np.array(temp_backward_t)
    backward_indexes = np.where(binned_t <= 0.)[0]
    forward_indexes = np.where(binned_t >= 0.)[0]
    temp_backward_t[backward_indexes] = binned_t[backward_indexes]
    temp_forward_t[forward_indexes] = binned_t[forward_indexes]
    backward_t = np.nanmax([backward_t, temp_backward_t], axis=0)
    forward_t = np.nanmin([forward_t, temp_forward_t], axis=0)
    return backward_t, forward_t


def merge_min_t_arrays(context, induction_loc, backward_t, forward_t, debug=False):
    """

    :param context: :class:'Context'
    :param induction_loc:
    :param backward_t:
    :param forward_t:
    :param debug: bool
    :return: array
    """
    binned_x = context.binned_x
    binned_extra_x = context.binned_extra_x
    extended_binned_x = context.extended_binned_x
    merged_min_t = np.empty_like(binned_extra_x)
    merged_min_t[:] = np.nan
    extended_min_t = np.empty_like(extended_binned_x)
    extended_min_t[:] = np.nan
    before = np.where(binned_extra_x < induction_loc)[0]
    if np.any(before):
        merged_min_t[before] = backward_t[before]
        extended_min_t[np.add(before[:-1], 2 * len(binned_x))] = forward_t[before[:-1]]
    else:
        merged_min_t[0] = backward_t[0]
        if debug:
            print 'merge_min_t_arrays: no before indexes'
    after = np.where(binned_extra_x >= induction_loc)[0]
    if np.any(after):
        merged_min_t[after] = forward_t[after]
        extended_min_t[after[1:-1]] = backward_t[after[1:-1]]
    else:
        if debug:
            print 'merge_min_t_arrays: no after indexes'
    if debug:
        for i in xrange(len(merged_min_t)):
            val = merged_min_t[i]
            if np.isnan(val):
                print 'merge_min_t_arrays: nan in merged_min_t at index: %i' % i
                break
        fig4, axes4 = plt.subplots(1)
        axes4.plot(binned_extra_x, backward_t, binned_extra_x, forward_t)
        axes4.plot(binned_extra_x, merged_min_t, label='Merged')
        axes4.legend(loc='best', frameon=False, framealpha=0.5)
        fig4.show()

        print 'merge_min_t_arrays: val at backward_t[0]: %.2f; val at forward_t[-1]: %.2f' % \
              (backward_t[0], forward_t[-1])
    extended_min_t[len(binned_x):2 * len(binned_x)] = merged_min_t[:-1]
    return extended_min_t


@click.command()
@click.option("--data-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='data/20180411_BTSP2_CA1_data.hdf5')
@click.option("--model-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='data/20180511_BTSP_all_cells_PopulationAnnealing_optimization_merged_exported_output.hdf5')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--cell-id", type=int, default=1)
@click.option("--export", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--label", type=str, default=None)
def main(data_file_path, model_file_path, output_dir, cell_id, export, debug, label):
    """

    :param data_file_path: str (path)
    :param model_file_path: str (path)
    :param output_dir: str (dir)
    :param cell_id: int
    :param export: bool
    :param label: str
    """
    date_stamp = datetime.datetime.today().strftime('%Y%m%d_%H%M')
    if label is None:
        label = date_stamp
    else:
        label = '%s_%s' % (date_stamp, label)
    if not os.path.isfile(data_file_path):
        raise IOError('plot_BTSP_data_analysis: invalid data_file_path: %s' % data_file_path)
    if not os.path.isfile(model_file_path):
        raise IOError('plot_BTSP_data_analysis: invalid model_file_path: %s' % model_file_path)
    if export and not os.path.isdir(output_dir):
        raise IOError('plot_BTSP_data_analysis: invalid output_dir: %s' % output_dir)
    target_cell_key = str(cell_id)
    with h5py.File(model_file_path, 'r') as g:
        if target_cell_key not in g['exported_data'] or '1' not in g['exported_data'][target_cell_key] or \
                '2' not in g['exported_data'][target_cell_key]:
            raise KeyError('plot_BTSP_data_analysis: problem loading data for provided cell_id: %i' % cell_id)
    with h5py.File(data_file_path, 'r') as f:
        if target_cell_key not in f['data'] or '1' not in f['data'][target_cell_key] or \
                '2' not in f['data'][target_cell_key]:
            raise KeyError('plot_BTSP_data_analysis: problem loading data for provided cell_id: %i' % cell_id)
        binned_x = f['defaults']['binned_x'][:]
        dt = f['defaults'].attrs['dt']
        track_length = f['defaults'].attrs['track_length']
    binned_extra_x = np.linspace(0., track_length, 101)
    extended_binned_x = np.concatenate([binned_x - track_length, binned_x, binned_x + track_length])
    reference_delta_t = np.linspace(-5000., 5000., 100)
    context.update(locals())

    delta_ramp_amp1 = []
    peak_ramp_amp = []
    total_induction_dur = []
    spont_indexes = []
    exp1_indexes = []
    exp2_indexes = []

    exp_ramp = {}
    extended_exp_ramp = {}
    delta_exp_ramp = {}
    mean_induction_loc = {}
    extended_min_delta_t = {}
    extended_delta_exp_ramp = {}
    pop_mean_min_delta_t = {}
    pop_mean_delta_exp_ramp = {}

    fig, axes = plt.figure(figsize=(11.5, 7.5)), []
    gs0 = gridspec.GridSpec(3, 4, wspace=0.65, hspace=0.75, left=0.075, right=0.975, top=0.95, bottom=0.1)
    subaxes = []
    gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[1, :2], wspace=0.65, hspace=0.75)

    for col in xrange(4):
        axes.append(fig.add_subplot(gs0[2, col]))
    for row in xrange(2):
        subaxes.append(fig.add_subplot(gs1[row]))
    for col in xrange(2,4):
        subaxes.append(fig.add_subplot(gs0[1, col]))

    with h5py.File(data_file_path, 'r') as f:
        for cell_key in f['data']:
            for induction_key in f['data'][cell_key]:
                induction_locs = f['data'][cell_key][induction_key].attrs['induction_locs']
                induction_durs = f['data'][cell_key][induction_key].attrs['induction_durs']
                if induction_key == '1':
                    if f['data'][cell_key].attrs['spont']:
                        spont_indexes.append(len(total_induction_dur))
                        group = 'spont'
                    else:
                        exp1_indexes.append(len(total_induction_dur))
                        group = 'exp1'
                else:
                    exp2_indexes.append(len(total_induction_dur))
                    group = 'exp2'
                total_induction_dur.append(np.sum(induction_durs))
                if group not in pop_mean_min_delta_t:
                    pop_mean_min_delta_t[group] = []
                    pop_mean_delta_exp_ramp[group] = []

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
                # if induction_key == '1':
                delta_ramp_amp1.append(np.max(delta_exp_ramp[cell_key][induction_key]))
                peak_ramp_amp.append(np.max(exp_ramp[cell_key][induction_key]['after']))
                mean_induction_loc[cell_key][induction_key] = np.mean(induction_locs)
                extended_delta_exp_ramp[cell_key][induction_key] = \
                    np.concatenate([delta_exp_ramp[cell_key][induction_key]] * 3)
                for category in exp_ramp[cell_key][induction_key]:
                    extended_exp_ramp[cell_key][induction_key][category] = \
                        np.concatenate([exp_ramp[cell_key][induction_key][category]] * 3)
                if debug:
                    fig1, axes1 = plt.subplots(1)
                backward_t = np.empty_like(binned_extra_x)
                backward_t[:] = np.nan
                forward_t = np.array(backward_t)
                for i in xrange(len(induction_locs)):
                    this_induction_loc = mean_induction_loc[cell_key][induction_key]
                    key = str(i)
                    # this_induction_loc = induction_locs[i]
                    this_position = f['data'][cell_key][induction_key]['processed']['position']['induction'][key][:]
                    this_t = f['data'][cell_key][induction_key]['processed']['t']['induction'][key][:]
                    this_induction_index = np.where(this_position >= this_induction_loc)[0][0]
                    this_induction_t = this_t[this_induction_index]
                    if i == 0 and 'pre' in f['data'][cell_key][induction_key]['raw']['position']:
                        pre_position = f['data'][cell_key][induction_key]['processed']['position']['pre']['0'][:]
                        pre_t = f['data'][cell_key][induction_key]['processed']['t']['pre']['0'][:]
                        pre_t -= len(pre_t) * dt
                        pre_t -= this_induction_t
                        backward_t, forward_t = update_min_t_arrays(context, pre_t, pre_position, backward_t, forward_t)
                        if debug:
                            axes1.plot(np.subtract(pre_position,
                                                   track_length + mean_induction_loc[cell_key][induction_key]), pre_t,
                                       label='Lap: Pre')
                    elif i > 0:
                        prev_t -= len(prev_t) * dt
                        prev_induction_t = prev_t[prev_induction_index]
                        backward_t, forward_t = update_min_t_arrays(context, np.subtract(prev_t, this_induction_t),
                                                                    prev_position, backward_t, forward_t)
                        if debug:
                            axes1.plot(np.subtract(prev_position,
                                                   track_length + mean_induction_loc[cell_key][induction_key]),
                                       np.subtract(prev_t, this_induction_t), label='Lap: %s (Prev)' % prev_key)

                        backward_t, forward_t = update_min_t_arrays(context, np.subtract(this_t, prev_induction_t),
                                                                    this_position, backward_t, forward_t)
                        if debug:
                            axes1.plot(np.subtract(this_position,
                                                   mean_induction_loc[cell_key][induction_key] - track_length),
                                       np.subtract(this_t, prev_induction_t), label='Lap: %s (Next)' % key)
                    backward_t, forward_t = update_min_t_arrays(context, np.subtract(this_t, this_induction_t),
                                                                this_position, backward_t, forward_t)
                    if debug:
                        axes1.plot(np.subtract(this_position, mean_induction_loc[cell_key][induction_key]),
                                 np.subtract(this_t, this_induction_t), label='Lap: %s (Current)' % key)
                    if i == len(induction_locs) - 1 and 'post' in f['data'][cell_key][induction_key]['raw']['position']:
                        post_position = f['data'][cell_key][induction_key]['processed']['position']['post']['0'][:]
                        post_t = f['data'][cell_key][induction_key]['processed']['t']['post']['0'][:]
                        post_t += len(this_t) * dt
                        post_t -= this_induction_t
                        backward_t, forward_t = update_min_t_arrays(context, post_t, post_position, backward_t,
                                                                    forward_t)
                        if debug:
                            axes1.plot(np.subtract(post_position,
                                                   mean_induction_loc[cell_key][induction_key] - track_length), post_t,
                                       label='Lap: Post')
                    prev_key = key
                    prev_induction_index = this_induction_index
                    prev_t = this_t
                    prev_position = this_position
                extended_min_delta_t[cell_key][induction_key] = \
                    merge_min_t_arrays(context, mean_induction_loc[cell_key][induction_key], backward_t, forward_t)
                this_extended_delta_position = np.subtract(extended_binned_x,
                                                           mean_induction_loc[cell_key][induction_key])
                if debug:
                    axes1.plot(this_extended_delta_position, extended_min_delta_t[cell_key][induction_key], c='k',
                               label='Min Interval')
                    fig1.suptitle('Cell: %s; Induction: %s' % (cell_key, induction_key),
                                  fontsize=mpl.rcParams['font.size'])
                    box = axes1.get_position()
                    axes1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    axes1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, framealpha=0.5)
                    axes1.set_xlabel('Position relative to plateau onset (cm)')
                    axes1.set_ylabel('Time relative to plateau onset (ms)')
                    clean_axes(axes1)
                    fig1.show()
                indexes = np.where((this_extended_delta_position > -track_length) &
                                   (this_extended_delta_position < track_length))[0]
                if induction_key == '2':
                    axes[0].plot(extended_min_delta_t[cell_key][induction_key][indexes] / 1000.,
                                 extended_delta_exp_ramp[cell_key][induction_key][indexes], c='grey', alpha=0.25)
                mask = ~np.isnan(extended_min_delta_t[cell_key][induction_key])
                indexes = np.where((extended_min_delta_t[cell_key][induction_key][mask] >= -5000.) &
                                   (extended_min_delta_t[cell_key][induction_key][mask] <= 5000.))[0]
                this_interp_delta_exp_ramp = np.interp(reference_delta_t,
                                                       extended_min_delta_t[cell_key][induction_key][mask][indexes],
                                                       extended_delta_exp_ramp[cell_key][induction_key][mask][indexes])
                if cell_key == target_cell_key:
                    if induction_key == '1':
                        color = 'k'
                        label = 'After induction 1'
                    else:
                        color = 'c'
                        label = 'After induction 2'
                    subaxes[2].plot(binned_x, exp_ramp[cell_key][induction_key]['after'], c=color, label=label)
                    subaxes[3].plot(reference_delta_t / 1000., this_interp_delta_exp_ramp, c=color)
                pop_mean_delta_exp_ramp[group].append(this_interp_delta_exp_ramp)
            if cell_key == target_cell_key:
                mean_induction_start_locs = []
                mean_induction_stop_locs = []
                this_ylim = subaxes[2].get_ylim()
                this_ymax = this_ylim[1] * 1.1
                subaxes[2].set_ylim(this_ylim[0], this_ymax)
                subaxes[2].legend(loc=(0.1, 0.95), frameon=False, framealpha=0.5, handlelength=1., handletextpad=0.5)
                for row in xrange(2):
                    induction_key = str(row + 1)
                    this_complete_position = f['data'][cell_key][induction_key]['complete']['position'][:]
                    this_complete_t = f['data'][cell_key][induction_key]['complete']['t'][:]
                    with h5py.File(model_file_path, 'r') as g:
                        group = g['exported_data'][cell_key][induction_key]['model_ramp_features']
                        induction_start_times = group.attrs['induction_start_times']
                        mean_induction_start_locs.append(group.attrs['mean_induction_start_loc'])
                        mean_induction_stop_locs.append(group.attrs['mean_induction_stop_loc'])
                    induction_start_indexes = []
                    for this_induction_start_time in induction_start_times:
                        index = np.where(this_complete_t >= this_induction_start_time)[0][0]
                        induction_start_indexes.append(index)
                    pretty_position = np.array(this_complete_position % track_length)
                    for i in xrange(1, len(pretty_position)):
                        if pretty_position[i] - pretty_position[i - 1] < -track_length / 2.:
                            pretty_position[i] = np.nan
                    subaxes[row].plot(this_complete_t / 1000., pretty_position, c='darkgrey')
                    xmax = math.ceil(this_complete_t[induction_start_indexes[-1]] / 10000.) * 10.
                    subaxes[row].set_xlim(0., xmax)
                    if induction_key == '1':
                        color = 'k'
                    else:
                        color = 'c'
                    subaxes[row].scatter(this_complete_t[induction_start_indexes] / 1000.,
                                         pretty_position[induction_start_indexes], c=color, s=40, linewidth=0, zorder=1)
                    subaxes[row].set_yticks(np.arange(0., track_length, 60.))
                    subaxes[2].hlines(this_ymax * 0.95, xmin=mean_induction_start_locs[row],
                                      xmax=mean_induction_stop_locs[row], color=color)
    handles = [mlines.Line2D([0], [0], linestyle='none', mfc=color, mew=0, marker='o', ms=math.sqrt(40.))
               for color in ['k', 'c']]
    labels = ['Induction 1', 'Induction 2']
    subaxes[0].legend(handles=handles, labels=labels, loc='best', frameon=False, framealpha=0.5, handletextpad=0.3,
                      handlelength=1.)
    subaxes[0].set_ylabel('Position (cm)')
    subaxes[1].set_xlabel('Time (s)')
    subaxes[2].set_ylabel('Ramp amplitude (mV)')
    subaxes[2].set_xlabel('Position (cm)')
    subaxes[2].set_xticks(np.arange(0., track_length, 45.))
    subaxes[3].set_ylabel('Change in\nramp amplitude (mV)')
    subaxes[3].set_xlabel('Time relative to plateau onset (s)')
    subaxes[3].set_xlim(-4., 4.)
    subaxes[3].set_xticks([-4., -2., 0., 2., 4.])
    subaxes[3].set_yticks(np.arange(-10., 16., 5.))
    clean_axes(subaxes)

    axes[0].set_title('After induction 2', fontsize=mpl.rcParams['font.size'])
    axes[0].set_xlim(-5., 5.)
    axes[0].plot(reference_delta_t / 1000., np.mean(pop_mean_delta_exp_ramp['exp2'], axis=0), c='c', linewidth=1.5)
    axes[0].set_ylabel('Change in\nramp amplitude (mV)')
    axes[0].set_xlabel('Time relative to plateau onset (s)')
    axes[0].set_yticks(np.arange(-10., 16., 5.))
    axes[0].set_xticks(np.arange(-4., 5., 2.))

    total_induction_dur = np.array(total_induction_dur)
    delta_ramp_amp1 = np.array(delta_ramp_amp1)
    peak_ramp_amp = np.array(peak_ramp_amp)

    axes[1].scatter(total_induction_dur[exp1_indexes+spont_indexes], peak_ramp_amp[exp1_indexes+spont_indexes],
                    c='darkgrey', s=40., alpha=0.5, linewidth=0)
    axes[1].scatter(total_induction_dur[exp2_indexes], peak_ramp_amp[exp2_indexes], c='c', s=40., alpha=0.5,
                    linewidth=0)
    axes[1].set_ylabel('Peak ramp\namplitude (mV)')
    axes[1].set_xlabel('Total accumulated\nplateau duration (ms)')
    ylim = [0., math.ceil(np.max(peak_ramp_amp) + 3.)]
    xlim = [0., (math.ceil(np.max(total_induction_dur) / 100.) + 1.) * 100.]
    result = np.polyfit(total_induction_dur, peak_ramp_amp, 1)
    fit = np.vectorize(lambda x: result[0] * x + result[1])
    fit_xlim = [(math.floor(np.min(total_induction_dur) / 100.) - 1.) * 100.,
            (math.ceil(np.max(total_induction_dur) / 100.) + 1.) * 100.]
    axes[1].plot(fit_xlim, fit(fit_xlim), c='grey', alpha=0.5, zorder=0, linestyle='--')
    axes[1].set_ylim(ylim)
    axes[1].set_xlim(xlim)
    handles = [mlines.Line2D([0], [0], linestyle='none', mfc=color, mew=0, alpha=0.5, marker='o', ms=math.sqrt(40.))
               for color in ['darkgrey', 'c']]
    labels = ['After induction 1', 'After induction 2']
    axes[1].legend(handles=handles, labels=labels, loc=(0.05, 0.95), frameon=False, framealpha=0.5, handletextpad=0.3,
                   handlelength=1.)
    r_val, p_val = pearsonr(total_induction_dur, peak_ramp_amp)
    axes[1].annotate('R$^{2}$ = %.3f; p %s %.3f' %
                     (r_val**2., '>' if p_val > 0.05 else '<', p_val if p_val > 0.001 else 0.001), xy=(0.03, 0.025),
                     xycoords='axes fraction')

    delta_amp_at_peak2 = []
    amp1_at_peak2 = []
    delta_amp_at_peak1 = []
    amp1_at_peak1 = []
    delta_amp1_forward = []
    delta_amp2_forward = []
    delta_amp1_backward = []
    delta_amp2_backward = []
    forward_delta_t_at_peak1 = []
    backward_delta_t_at_peak1 = []
    delta_x_at_peak1 = []

    with h5py.File(data_file_path, 'r') as f:
        with h5py.File(model_file_path, 'r') as g:
            for cell_key in (cell_key for cell_key in f['data'] if '2' in f['data'][cell_key]):
                induction_key = '2'
                this_extended_delta_position = \
                    np.subtract(extended_binned_x, mean_induction_loc[cell_key][induction_key])
                indexes = np.where((this_extended_delta_position > -track_length) &
                                   (this_extended_delta_position < track_length))[0]
                if cell_key not in g['exported_data'] or induction_key not in g['exported_data'][cell_key]:
                    raise KeyError('Problem encountered loading data from model_file_path: %s for cell: %s, induction: '
                                   '%s' % (model_file_path, cell_key, induction_key))
                peak2_loc = \
                    g['exported_data'][cell_key][induction_key]['model_ramp_features'].attrs['target_local_peak_shift']
                peak2_index = np.where(this_extended_delta_position[indexes] >= peak2_loc)[0][0]
                delta_amp_at_peak2.append(extended_delta_exp_ramp[cell_key][induction_key][indexes][peak2_index])
                amp1_at_peak2.append(extended_exp_ramp[cell_key][induction_key]['before'][indexes][peak2_index])
                peak1_pre_index = np.argmax(extended_exp_ramp[cell_key][induction_key]['before'][indexes][:peak2_index])
                peak1_post_index = \
                    np.argmax(extended_exp_ramp[cell_key][induction_key]['before'][indexes][peak2_index:]) + peak2_index
                if abs(extended_min_delta_t[cell_key][induction_key][indexes][peak1_pre_index]) > \
                        abs(extended_min_delta_t[cell_key][induction_key][indexes][peak1_post_index]):
                    peak1_index = peak1_post_index
                    delta_amp2_forward.append(extended_delta_exp_ramp[cell_key][induction_key][indexes][peak2_index])
                    delta_amp1_forward.append(extended_delta_exp_ramp[cell_key][induction_key][indexes][peak1_index])
                    forward_delta_t_at_peak1.append(
                        extended_min_delta_t[cell_key][induction_key][indexes][peak1_index])
                else:
                    peak1_index = peak1_pre_index
                    delta_amp2_backward.append(extended_delta_exp_ramp[cell_key][induction_key][indexes][peak2_index])
                    delta_amp1_backward.append(extended_delta_exp_ramp[cell_key][induction_key][indexes][peak1_index])
                    backward_delta_t_at_peak1.append(
                        extended_min_delta_t[cell_key][induction_key][indexes][peak1_index])
                this_delta_x = abs(this_extended_delta_position[indexes][peak1_index])
                delta_x_at_peak1.append(this_delta_x)
                amp1_at_peak1.append(extended_exp_ramp[cell_key][induction_key]['before'][indexes][peak1_index])
                delta_amp_at_peak1.append(extended_delta_exp_ramp[cell_key][induction_key][indexes][peak1_index])

    if debug:
        fig6, axes6 = plt.subplots(1)
        axes6.plot(extended_min_delta_t[cell_key][induction_key][indexes] / 1000.,
                   extended_delta_exp_ramp[cell_key][induction_key][indexes], c='c')
        axes6.plot(extended_min_delta_t[cell_key][induction_key][indexes] / 1000.,
                   extended_exp_ramp[cell_key][induction_key]['before'][indexes], c='grey')
        axes6.plot(extended_min_delta_t[cell_key][induction_key][indexes] / 1000.,
                   extended_exp_ramp[cell_key][induction_key]['after'][indexes], c='k')
        axes6.scatter([extended_min_delta_t[cell_key][induction_key][indexes][peak1_index] / 1000.,
                       extended_min_delta_t[cell_key][induction_key][indexes][peak2_index] / 1000.],
                      [extended_exp_ramp[cell_key][induction_key]['before'][indexes][peak1_index],
                       extended_exp_ramp[cell_key][induction_key]['after'][indexes][peak2_index]])
        axes6.set_xlim(-5., 5.)
        fig6.suptitle('Induction: %s; Cell: %s' % (induction_key, cell_key),
                      fontsize=mpl.rcParams['font.size'])
        fig6.show()

    xvals = np.abs(delta_x_at_peak1)
    xmin = np.min(xvals)
    xmax = np.max(xvals)
    axes[2].scatter(xvals, delta_amp_at_peak2, c='c', s=40, alpha=0.5, linewidth=0)
    axes[2].scatter(xvals, delta_amp_at_peak1, c='darkgrey', s=40, alpha=0.5, linewidth=0)
    axes[2].set_ylabel('Change in\nramp amplitude (mV)')
    axes[2].set_xlabel('Distance from initial peak\nto induction 2 (cm)')
    this_ylim = [math.floor(min(delta_amp_at_peak2 + delta_amp_at_peak1)) - 4.,
                 math.ceil(max(delta_amp_at_peak2 + delta_amp_at_peak1)) + 4.]
    axes[2].set_ylim(this_ylim)
    axes[2].set_xticks(np.arange(0., 125., 30.))
    axes[2].set_yticks(np.arange(-10., 16., 5.))
    onset2_result = np.polyfit(xvals, delta_amp_at_peak2, 1)
    onset2_fit = np.vectorize(lambda x: onset2_result[0] * x + onset2_result[1])
    peak1_result = np.polyfit(xvals, delta_amp_at_peak1, 1)
    peak1_fit = np.vectorize(lambda x: peak1_result[0] * x + peak1_result[1])
    onset2_xlim = [math.floor(min(xvals)) - 1., math.ceil(max(xvals)) + 1.]
    peak1_xlim = [math.floor(min(xvals)) - 1., math.ceil(max(xvals)) + 1.]
    axes[2].plot(onset2_xlim, onset2_fit(onset2_xlim), c='c', alpha=0.5, zorder=0, linestyle='--')
    axes[2].plot(peak1_xlim, peak1_fit(peak1_xlim), c='grey', alpha=0.5, zorder=0, linestyle='--')
    axes[2].set_aspect('auto', adjustable='box')
    handles = [mlines.Line2D([0], [0], linestyle='none', mfc=color, mew=0, alpha=0.5, marker='o', ms=math.sqrt(40.))
               for color in ['darkgrey', 'c']]
    labels = ['Initial peak position', 'Translocated peak position']
    axes[2].legend(handles=handles, labels=labels, loc=(-0.1, 0.95), frameon=False, framealpha=0.5, handletextpad=0.3,
                    handlelength=1.)
    r_val, p_val = pearsonr(xvals, delta_amp_at_peak2)
    axes[2].annotate('R$^{2}$ = %.3f;\np %s %.3f' %
                     (r_val ** 2., '>' if p_val > 0.05 else '<', p_val if p_val > 0.001 else 0.001), xy=(0.03, 0.7),
                     xycoords='axes fraction')
    r_val, p_val = pearsonr(xvals, delta_amp_at_peak1)
    axes[2].annotate('R$^{2}$ = %.3f; p %s %.3f' %
                     (r_val ** 2., '>' if p_val > 0.05 else '<', p_val if p_val > 0.001 else 0.001), xy=(0.03, 0.025),
                     xycoords='axes fraction')
    
    axes[3].scatter(amp1_at_peak2, delta_amp_at_peak2, c='c', s=40, alpha=0.5, linewidth=0)
    axes[3].scatter(amp1_at_peak1, delta_amp_at_peak1, c='darkgrey', s=40, alpha=0.5, linewidth=0)
    axes[3].set_ylabel('Change in\nramp amplitude (mV)')
    axes[3].set_xlabel('Initial ramp amplitude (mV)')
    this_ylim = [math.floor(min(delta_amp_at_peak2 + delta_amp_at_peak1)) - 4.,
                 math.ceil(max(delta_amp_at_peak2 + delta_amp_at_peak1)) + 4.]
    this_xlim = [math.floor(min(amp1_at_peak2 + amp1_at_peak1)) - 1.,
                 math.ceil(max(amp1_at_peak2 + amp1_at_peak1)) + 1.]
    axes[3].set_ylim(this_ylim)
    axes[3].set_xlim(this_xlim)
    axes[3].set_yticks(np.arange(-10., 16., 5.))
    onset2_result = np.polyfit(amp1_at_peak2, delta_amp_at_peak2, 1)
    onset2_fit = np.vectorize(lambda x: onset2_result[0] * x + onset2_result[1])
    peak1_result = np.polyfit(amp1_at_peak1, delta_amp_at_peak1, 1)
    peak1_fit = np.vectorize(lambda x: peak1_result[0] * x + peak1_result[1])
    onset2_xlim = [math.floor(min(amp1_at_peak2)) - 1., math.ceil(max(amp1_at_peak2)) + 1.]
    peak1_xlim = [math.floor(min(amp1_at_peak1)) - 1., math.ceil(max(amp1_at_peak1)) + 1.]
    axes[3].plot(onset2_xlim, onset2_fit(onset2_xlim), c='c', alpha=0.5, zorder=0, linestyle='--')
    axes[3].plot(peak1_xlim, peak1_fit(peak1_xlim), c='grey', alpha=0.5, zorder=0, linestyle='--')
    r_val, p_val = pearsonr(amp1_at_peak2, delta_amp_at_peak2)
    axes[3].annotate('R$^{2}$ = %.3f; p %s %.3f' %
                     (r_val**2., '>' if p_val > 0.05 else '<', p_val if p_val > 0.001 else 0.001), xy=(0.15, 0.85),
                     xycoords='axes fraction')
    r_val, p_val = pearsonr(amp1_at_peak1, delta_amp_at_peak1)
    axes[3].annotate('R$^{2}$ = %.3f; p %s %.3f' % 
                     (r_val**2., '>' if p_val > 0.05 else '<', p_val if p_val > 0.001 else 0.001), xy=(0.03, 0.025),
                     xycoords='axes fraction')
    clean_axes(axes)
    fig.show()

    fig8, axes8 = plt.figure(figsize=(11, 7)), []
    gs8 = gridspec.GridSpec(3, 4, wspace=0.55, hspace=0.5, left=0.075, right=0.975, top=0.95, bottom=0.075)
    axes8.append(fig8.add_subplot(gs8[0,0]))

    axes8[0].set_ylabel('Change in\nramp amplitude (mV)')
    axes8[0].set_xlabel('Time from initial peak\nto induction 2 (s)')
    ymin = math.floor(min(delta_amp2_forward + delta_amp1_forward + delta_amp2_backward + delta_amp1_backward))
    ymax = math.ceil(max(delta_amp2_forward + delta_amp1_forward + delta_amp2_backward + delta_amp1_backward))
    axes8[0].set_ylim([ymin - 4., ymax + 4.])
    axes8[0].set_yticks(np.arange(-10., 16., 5.))
    axes8[0].set_xticks(np.arange(-5., 11., 5.))
    xvals = np.array(forward_delta_t_at_peak1) / 1000.
    xmin = np.min(xvals)
    xmax = np.max(xvals)
    axes8[0].scatter(xvals, delta_amp2_forward, c='c', s=40, alpha=0.5, linewidth=0)
    axes8[0].scatter(xvals, delta_amp1_forward, c='darkgrey', s=40, alpha=0.5, linewidth=0)
    onset2_result = np.polyfit(xvals, delta_amp2_forward, 1)
    onset2_fit = np.vectorize(lambda x: onset2_result[0] * x + onset2_result[1])
    peak1_result = np.polyfit(xvals, delta_amp1_forward, 1)
    peak1_fit = np.vectorize(lambda x: peak1_result[0] * x + peak1_result[1])
    fit_xlim = [math.floor(min(xvals)) - 1., math.ceil(max(xvals)) + 1.]
    axes8[0].plot(fit_xlim, onset2_fit(fit_xlim), c='c', alpha=0.5, zorder=0, linestyle='--')
    axes8[0].plot(fit_xlim, peak1_fit(fit_xlim), c='grey', alpha=0.5, zorder=0, linestyle='--')
    handles = [mlines.Line2D([0], [0], linestyle='none', mfc=color, mew=0, alpha=0.5, marker='o', ms=math.sqrt(40.))
               for color in ['darkgrey', 'c']]
    labels = ['Initial peak position', 'Translocated peak position']
    axes8[0].legend(handles=handles, labels=labels, loc=(0.05, 0.9), frameon=False, framealpha=0.5, handletextpad=0.3,
                    handlelength=1.)
    r_val, p_val = pearsonr(xvals, delta_amp2_forward)
    print 'Peak 2 forward: R$^{2}$ = %.3f; p %s %.3f' % \
          (r_val ** 2., '>' if p_val > 0.05 else '<', p_val if p_val > 0.001 else 0.001)
    r_val, p_val = pearsonr(xvals, delta_amp1_forward)
    print 'Peak 1 forward: R$^{2}$ = %.3f; p %s %.3f' % \
          (r_val ** 2., '>' if p_val > 0.05 else '<', p_val if p_val > 0.001 else 0.001)

    xvals = np.array(backward_delta_t_at_peak1) / 1000.
    xmin = np.min(xvals)
    xmax = np.max(xvals)
    axes8[0].scatter(xvals, delta_amp2_backward, c='c', s=40, alpha=0.5, linewidth=0)
    axes8[0].scatter(xvals, delta_amp1_backward, c='darkgrey', s=40, alpha=0.5, linewidth=0)
    onset2_result = np.polyfit(xvals, delta_amp2_backward, 1)
    onset2_fit = np.vectorize(lambda x: onset2_result[0] * x + onset2_result[1])
    peak1_result = np.polyfit(xvals, delta_amp1_backward, 1)
    peak1_fit = np.vectorize(lambda x: peak1_result[0] * x + peak1_result[1])
    fit_xlim = [math.floor(min(xvals)) - 1., math.ceil(max(xvals)) + 1.]
    axes8[0].plot(fit_xlim, onset2_fit(fit_xlim), c='c', alpha=0.5, zorder=0, linestyle='--')
    axes8[0].plot(fit_xlim, peak1_fit(fit_xlim), c='grey', alpha=0.5, zorder=0, linestyle='--')
    r_val, p_val = pearsonr(xvals, delta_amp2_backward)
    print 'Peak 2 backward: R$^{2}$ = %.3f; p %s %.3f' % \
          (r_val ** 2., '>' if p_val > 0.05 else '<', p_val if p_val > 0.001 else 0.001)
    r_val, p_val = pearsonr(xvals, delta_amp1_backward)
    print 'Peak 1 backward: R$^{2}$ = %.3f; p %s %.3f' % \
          (r_val ** 2., '>' if p_val > 0.05 else '<', p_val if p_val > 0.001 else 0.001)
    axes8[0].set_aspect('auto', adjustable='box')
    clean_axes(axes8)
    fig8.show()

    context.update(locals())

    flat_min_t = []
    flat_delta_ramp = []
    flat_initial_ramp = []

    induction_key = '2'
    for cell_key in context.extended_delta_exp_ramp:
        if induction_key in context.extended_delta_exp_ramp[cell_key]:
            this_min_t = np.array(context.extended_min_delta_t[cell_key][induction_key])
            this_delta_ramp = np.array(context.extended_delta_exp_ramp[cell_key][induction_key])
            this_initial_ramp = np.array(context.extended_exp_ramp[cell_key][induction_key]['before'])
            valid_indexes = np.where(~np.isnan(this_min_t))
            indexes = np.where((this_min_t[valid_indexes] >= -5000) & (this_min_t[valid_indexes] <= 5000.))[0]
            flat_min_t.extend(this_min_t[valid_indexes][indexes])
            flat_delta_ramp.extend(this_delta_ramp[valid_indexes][indexes])
            flat_initial_ramp.extend(this_initial_ramp[valid_indexes][indexes])
    flat_min_t = np.divide(flat_min_t, 1000.)

    points = np.array([flat_min_t, flat_initial_ramp]).transpose()
    data = np.array(flat_delta_ramp)

    tmax = 5.
    interp_tmax = math.ceil(tmax * 1.2)
    deltat = 0.05
    min_t = np.arange(-interp_tmax, interp_tmax + deltat / 2., deltat)
    ymax = max(np.max(flat_initial_ramp), -np.min(flat_initial_ramp))

    interp_ymax = math.ceil(ymax * 1.2)
    deltay = 0.1
    initial_ramp = np.arange(-ymax, ymax + deltay / 2., deltay)

    vmax = max(np.max(flat_delta_ramp), -np.min(flat_delta_ramp))

    X, Y = np.meshgrid(min_t, initial_ramp)
    nn_interp_f = scipy.interpolate.NearestNDInterpolator(points, data, rescale=True)
    nn_interp_data = nn_interp_f(X, Y)

    context.update(locals())

    sigmax, sigmay = 10, 10
    smoothed_nn_interp_data = gaussian_filter(nn_interp_data, [sigmax, sigmay], mode='constant')
    fig, axes = plt.subplots()
    axes.scatter(flat_min_t, flat_initial_ramp, c=flat_delta_ramp, cmap='RdBu_r', s=5., vmax=vmax,
                 vmin=-vmax, zorder=1) #, alpha=0.5)
    cax = axes.pcolor(X, Y, smoothed_nn_interp_data, cmap='RdBu_r', vmax=vmax, vmin=-vmax, zorder=0)
    axes.set_ylabel('Initial ramp amplitude (mV)')
    axes.set_xlabel('Time relative to plateau onset (s)')
    axes.set_ylim(0., ymax)
    axes.set_xlim(-tmax, tmax)
    cbar = fig.colorbar(cax)
    cbar.set_label('Change in ramp amplitude', rotation=270., labelpad=15.)
    clean_axes(axes)

    fig.show()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)