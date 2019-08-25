from biBTSP_utils import *
from nested.optimize_utils import *
import matplotlib.lines as mlines
import matplotlib as mpl
import click
from scipy.stats import pearsonr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from collections import defaultdict

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
@click.option("--vmax", type=float, default=12.66762)
@click.option("--tmax", type=float, default=5.)
@click.option("--debug", is_flag=True)
@click.option("--target-induction", type=int, default=2)
def main(data_file_path, model_file_path, vmax, tmax, debug, target_induction):
    """

    :param data_file_path: str (path)
    :param model_file_path: str (path)
    :param vmax: float
    :param tmax: float
    :param debug: bool
    :param target_induction: int
    """
    if not os.path.isfile(data_file_path):
        raise IOError('plot_biBTSP_data_summary_figure: invalid data_file_path: %s' % data_file_path)
    if not os.path.isfile(model_file_path):
        raise IOError('plot_biBTSP_data_summary_figure: invalid model_file_path: %s' % model_file_path)
    with h5py.File(data_file_path, 'r') as f:
        binned_x = f['defaults']['binned_x'][:]
        dt = f['defaults'].attrs['dt']
        track_length = f['defaults'].attrs['track_length']
    binned_extra_x = np.linspace(0., track_length, 101)
    extended_binned_x = np.concatenate([binned_x - track_length, binned_x, binned_x + track_length])
    reference_delta_t = np.linspace(-5000., 5000., 100)
    context.update(locals())

    peak_ramp_amp = []
    total_induction_dur = []
    spont_indexes = []
    exp1_indexes = []
    exp2_indexes = []

    exp_ramp = defaultdict(lambda: defaultdict(dict))
    extended_exp_ramp = defaultdict(lambda: defaultdict(dict))
    delta_exp_ramp = defaultdict(dict)
    mean_induction_loc = defaultdict(dict)
    extended_min_delta_t = defaultdict(dict)
    extended_delta_exp_ramp = defaultdict(dict)
    interp_initial_exp_ramp = defaultdict(dict)
    interp_delta_exp_ramp = defaultdict(dict)
    interp_final_exp_ramp = defaultdict(dict)

    fig, axesgrid = plt.subplots(2, 2, figsize=(8.25, 6))
    axes = []
    for row in range(2):
        for col in range(2):
            axes.append(axesgrid[col][row])

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
                    exp_ramp[cell_key][induction_key]['before'] = \
                        np.zeros_like(exp_ramp[cell_key][induction_key]['after'])
                    delta_exp_ramp[cell_key][induction_key], discard = \
                        subtract_baseline(exp_ramp[cell_key][induction_key]['after'])

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
                        if debug:
                            axes1.plot(np.subtract(pre_position,
                                                   track_length + mean_induction_loc[cell_key][induction_key]), pre_t,
                                       label='Lap: Pre')
                    elif i > 0:
                        prev_t -= len(prev_t) * dt
                        prev_induction_t = prev_t[prev_induction_index]
                        backward_t, forward_t = update_min_t_arrays(binned_extra_x,
                                                                    np.subtract(prev_t, this_induction_t),
                                                                    prev_position, backward_t, forward_t)
                        if debug:
                            axes1.plot(np.subtract(prev_position,
                                                   track_length + mean_induction_loc[cell_key][induction_key]),
                                       np.subtract(prev_t, this_induction_t), label='Lap: %s (Prev)' % prev_key)

                        backward_t, forward_t = update_min_t_arrays(binned_extra_x,
                                                                    np.subtract(this_t, prev_induction_t),
                                                                    this_position, backward_t, forward_t)
                        if debug:
                            axes1.plot(np.subtract(this_position,
                                                   mean_induction_loc[cell_key][induction_key] - track_length),
                                       np.subtract(this_t, prev_induction_t), label='Lap: %s (Next)' % key)
                    backward_t, forward_t = update_min_t_arrays(binned_extra_x, np.subtract(this_t, this_induction_t),
                                                                this_position, backward_t, forward_t)
                    if debug:
                        axes1.plot(np.subtract(this_position, mean_induction_loc[cell_key][induction_key]),
                                   np.subtract(this_t, this_induction_t), label='Lap: %s (Current)' % key)
                    if i == len(induction_locs) - 1 and 'post' in f['data'][cell_key][induction_key]['raw']['position']:
                        post_position = f['data'][cell_key][induction_key]['processed']['position']['post']['0'][:]
                        post_t = f['data'][cell_key][induction_key]['processed']['t']['post']['0'][:]
                        post_t += len(this_t) * dt
                        post_t -= this_induction_t
                        backward_t, forward_t = update_min_t_arrays(binned_extra_x, post_t, post_position, backward_t,
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
                    merge_min_t_arrays(binned_x, binned_extra_x, extended_binned_x, mean_induction_loc[cell_key][induction_key], backward_t, forward_t)
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

                mask = ~np.isnan(extended_min_delta_t[cell_key][induction_key])
                indexes = np.where((extended_min_delta_t[cell_key][induction_key][mask] >= -5000.) &
                                   (extended_min_delta_t[cell_key][induction_key][mask] <= 5000.))[0]
                this_interp_delta_exp_ramp = np.interp(reference_delta_t,
                                                       extended_min_delta_t[cell_key][induction_key][mask][indexes],
                                                       extended_delta_exp_ramp[cell_key][induction_key][mask][indexes])

                interp_delta_exp_ramp[cell_key][induction_key] = this_interp_delta_exp_ramp

                interp_initial_exp_ramp[cell_key][induction_key] = \
                    np.interp(reference_delta_t, extended_min_delta_t[cell_key][induction_key][mask][indexes],
                              extended_exp_ramp[cell_key][induction_key]['before'][mask][indexes])
                interp_final_exp_ramp[cell_key][induction_key] = \
                    np.interp(reference_delta_t, extended_min_delta_t[cell_key][induction_key][mask][indexes],
                              extended_exp_ramp[cell_key][induction_key]['after'][mask][indexes])

    total_induction_dur = np.array(total_induction_dur)
    peak_ramp_amp = np.array(peak_ramp_amp)

    axes[0].scatter(total_induction_dur[exp1_indexes + spont_indexes], peak_ramp_amp[exp1_indexes + spont_indexes],
                    c='darkgrey', s=40., alpha=0.5, linewidth=0)
    axes[0].scatter(total_induction_dur[exp2_indexes], peak_ramp_amp[exp2_indexes], c='k', s=40., alpha=0.5,
                    linewidth=0)
    axes[0].set_ylabel('Peak ramp\namplitude (mV)')
    axes[0].set_xlabel('Total accumulated\nplateau duration (ms)')
    ylim = [0., math.ceil(np.max(peak_ramp_amp) + 3.)]
    xlim = [0., (math.ceil(np.max(total_induction_dur) / 100.) + 1.) * 100.]
    result = np.polyfit(total_induction_dur, peak_ramp_amp, 1)
    fit = np.vectorize(lambda x: result[0] * x + result[1])
    fit_xlim = [(math.floor(np.min(total_induction_dur) / 100.) - 1.) * 100.,
                (math.ceil(np.max(total_induction_dur) / 100.) + 1.) * 100.]
    axes[0].plot(fit_xlim, fit(fit_xlim), c='grey', alpha=0.5, zorder=0, linestyle='--')
    axes[0].set_ylim(ylim)
    axes[0].set_xlim(xlim)
    handles = [mlines.Line2D([0], [0], linestyle='none', mfc=color, mew=0, alpha=0.5, marker='o', ms=math.sqrt(40.))
               for color in ['darkgrey', 'k']]
    labels = ['After induction 1', 'After induction 2']
    axes[0].legend(handles=handles, labels=labels, loc=(0.05, 0.95), frameon=False, framealpha=0.5, handletextpad=0.3,
                   handlelength=1.)
    r_val, p_val = pearsonr(total_induction_dur, peak_ramp_amp)
    axes[0].annotate('R$^{2}$ = %.3f; p %s %.3f' %
                     (r_val ** 2., '>' if p_val > 0.05 else '<', p_val if p_val > 0.001 else 0.001), xy=(0.03, 0.025),
                     xycoords='axes fraction')

    with h5py.File(data_file_path, 'r') as f:
        with h5py.File(model_file_path, 'r') as g:
            for cell_key in (cell_key for cell_key in f['data'] if '2' in f['data'][cell_key]):
                induction_key = '2'
                this_extended_delta_position = \
                    np.subtract(extended_binned_x, mean_induction_loc[cell_key][induction_key])
                indexes = np.where((this_extended_delta_position > -track_length) &
                                   (this_extended_delta_position < track_length))[0]
                if cell_key not in g['exported_data'] or induction_key not in g['exported_data'][cell_key]:
                    raise RuntimeError('plot_biBTSP_data_summary_figure: problem encountered loading data from '
                                       'model_file_path: %s for cell: %s, induction: %s' %
                                       (model_file_path, cell_key, induction_key))
                peak2_loc = \
                    g['exported_data'][cell_key][induction_key]['model_ramp_features'].attrs['target_local_peak_shift']
                peak2_index = np.where(this_extended_delta_position[indexes] >= peak2_loc)[0][0]
                peak1_pre_index = np.argmax(extended_exp_ramp[cell_key][induction_key]['before'][indexes][:peak2_index])
                peak1_post_index = \
                    np.argmax(extended_exp_ramp[cell_key][induction_key]['before'][indexes][peak2_index:]) + peak2_index
                if abs(extended_min_delta_t[cell_key][induction_key][indexes][peak1_pre_index]) > \
                        abs(extended_min_delta_t[cell_key][induction_key][indexes][peak1_post_index]):
                    peak1_index = peak1_post_index
                else:
                    peak1_index = peak1_pre_index

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

    reference_delta_t = np.divide(reference_delta_t, 1000.)

    flat_min_t = []
    flat_delta_ramp = []
    flat_initial_ramp = []
    flat_final_ramp = []

    induction_key = str(target_induction)
    for cell_key in interp_delta_exp_ramp:
        # for induction_key in interp_delta_exp_ramp[cell_key]:
        if induction_key in interp_delta_exp_ramp[cell_key]:
            flat_min_t.extend(reference_delta_t)
            flat_delta_ramp.extend(interp_delta_exp_ramp[cell_key][induction_key])
            flat_initial_ramp.extend(interp_initial_exp_ramp[cell_key][induction_key])
            flat_final_ramp.extend(interp_final_exp_ramp[cell_key][induction_key])

    mean_final_ramp = np.mean([interp_final_exp_ramp[cell_key][induction_key] for
                               cell_key in interp_final_exp_ramp
                               if induction_key in interp_final_exp_ramp[cell_key]], axis=0)

    points = np.array([flat_min_t, flat_initial_ramp]).transpose()
    data = np.array(flat_delta_ramp)

    ymax = np.max(flat_initial_ramp)
    ymin = min(0., np.min(flat_initial_ramp))

    this_vmax = max(abs(np.max(flat_delta_ramp)), abs(np.min(flat_delta_ramp)))
    if this_vmax > vmax:
        print('New max detected for 3D data color legend: vmax: %.5f' % this_vmax)
        sys.stdout.flush()
        vmax = this_vmax

    lines_cmap = 'jet'
    interp_cmap = 'bwr_r'

    axes[1].set_xlim(-tmax, tmax)
    for cell_key in (cell_key for cell_key in interp_delta_exp_ramp
                     if induction_key in interp_delta_exp_ramp[cell_key]):
        lc = colorline(reference_delta_t, interp_delta_exp_ramp[cell_key][induction_key],
                       interp_initial_exp_ramp[cell_key][induction_key], vmin=ymin, vmax=ymax, cmap=lines_cmap)
        cax = axes[1].add_collection(lc)
    cbar = plt.colorbar(cax, ax=axes[1])
    cbar.set_label('Initial ramp\namplitude (mV)', rotation=270., labelpad=23.)
    axes[1].set_ylabel('Change in ramp\namplitude (mV)')
    axes[1].set_xlabel('Time relative to plateau onset (s)')
    axes[1].set_yticks(np.arange(-10., 16., 5.))
    axes[1].set_xticks(np.arange(-4., 5., 2.))

    if target_induction != 1:
        axes[2].scatter(flat_initial_ramp, flat_delta_ramp, c='grey', s=5., alpha=0.5, linewidth=0.)
        fit_params = np.polyfit(flat_initial_ramp, flat_delta_ramp, 1)
        fit_f = np.vectorize(lambda x: fit_params[0] * x + fit_params[1])
        xlim = [0., math.ceil(np.max(flat_initial_ramp))]
        axes[2].plot(xlim, fit_f(xlim), c='k', alpha=0.75, zorder=1, linestyle='--')
        r_val, p_val = pearsonr(flat_initial_ramp, flat_delta_ramp)
        axes[2].annotate('R$^{2}$ = %.3f;\np %s %.3f' %
                         (r_val ** 2., '>' if p_val > 0.05 else '<', p_val if p_val > 0.001 else 0.001), xy=(0.6, 0.7),
                         xycoords='axes fraction')
        axes[2].set_xlim(xlim)
        axes[2].set_xlabel('Initial ramp amplitude (mV)')
        axes[2].set_ylabel('Change in ramp\namplitude (mV)')

        res = 50
        t_range = np.linspace(np.min(flat_min_t), np.max(flat_min_t), res)
        initial_ramp_range = np.linspace(np.min(flat_initial_ramp), np.max(flat_initial_ramp), res)

        t_grid, initial_ramp_grid = np.meshgrid(t_range, initial_ramp_range)
        interp_points = np.vstack((t_grid.flatten(), initial_ramp_grid.flatten())).T

        kernel = RationalQuadratic(1., length_scale_bounds=(1e-10, 10.))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=3., n_restarts_optimizer=20, normalize_y=True)
        start_time = time.time()
        print('Starting Gaussian Process Regression with %i samples' % len(data))
        gp.fit(points, data)
        interp_data = gp.predict(interp_points).reshape(-1, res)
        print('Gaussian Process Regression took %.1f s' % (time.time() - start_time))

        # axes[3].plot(reference_delta_t, mean_final_ramp, c='k')
        cax = axes[3].pcolor(t_grid, initial_ramp_grid, interp_data, cmap=interp_cmap, vmin=-vmax, vmax=vmax, zorder=0)
        axes[3].set_ylabel('Initial ramp\namplitude (mV)')
        axes[3].set_xlabel('Time relative to plateau onset (s)')
        axes[3].set_ylim(0., ymax)
        axes[3].set_xlim(-tmax, tmax)
        axes[3].set_xticks(np.arange(-4., 5., 2.))
        cbar = plt.colorbar(cax, ax=axes[3])
        cbar.set_label('Change in ramp\namplitude (mV)', rotation=270., labelpad=23.)
    clean_axes(axes)
    fig.subplots_adjust(left=0.125, hspace=0.5, wspace=0.6, right=0.925)
    fig.show()

    context.update(locals())


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)