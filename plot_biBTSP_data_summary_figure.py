"""
To reproduce Figure 3:
python -i plot_biBTSP_data_summary_figure.py --target-induction=2

To reproduce panels from Figure 6:
python -i plot_biBTSP_data_summary_figure.py --target-induction=1 --target-induction=2
plot_ramp_prediction_from_interpolation(context.gp, 'data/20190711_biBTSP_data_calibrated_input.hdf5',
        context.binned_x, context.binned_extra_x, context.extended_binned_x, context.reference_delta_t,
        context.track_length, context.dt, context.debug, label='Control', color='k', tmax=context.tmax, induction=1,
        group='exp1')
plot_ramp_prediction_from_interpolation(context.gp, 'data/20190825_biBTSP_data_soma_DC.hdf5',
        context.binned_x, context.binned_extra_x, context.extended_binned_x, context.reference_delta_t,
        context.track_length, context.dt, context.debug, label='Depolarized', color='purple', tmax=context.tmax,
        induction=1, group='exp1')
boxplot_compare_ramp_summary_features(
        {'Control': 'data/20190812_biBTSP_SRL_B_90cm_all_cells_merged_exported_model_output.hdf5',
         'Depolarized': 'data/20190825_biBTSP_SRL_B_90cm_soma_DC_cells_exported_model_output.hdf5'},
        ['Control', 'Depolarized'], induction=1)
"""
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
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['text.usetex'] = False
mpl.rcParams['axes.titlepad'] = 2.
mpl.rcParams['mathtext.default'] = 'regular'

context = Context()


def plot_ramp_prediction_from_interpolation(regressor, data_file_path, binned_x, binned_extra_x, extended_binned_x,
                                            reference_delta_t, track_length, dt, debug, label, color='k', tmax=5.,
                                            induction=1, group='exp1', show_std=True):
    """

    :param regressor: :class:'GaussianProcessRegressor'
    :param data_file_path: str (path)
    :param binned_x: array
    :param binned_extra_x: array
    :param extended_binned_x: array
    :param reference_delta_t: array
    :param track_length: float
    :param dt: float
    :param debug: bool
    :param label: str
    :param color: str
    :param tmax: float
    :param induction: int
    :param show_std: bool
    """
    if not os.path.isfile(data_file_path):
        raise IOError('plot_ramp_prediction_from_interpolation: invalid data_file_path: %s' % data_file_path)

    peak_ramp_amp, total_induction_dur, depo_soma, group_indexes, exp_ramp, extended_exp_ramp, delta_exp_ramp, \
    mean_induction_loc, extended_min_delta_t, extended_delta_exp_ramp, interp_initial_exp_ramp, \
    interp_delta_exp_ramp, interp_final_exp_ramp = \
        get_biBTSP_analysis_results(data_file_path, binned_x, binned_extra_x, extended_binned_x, reference_delta_t,
                                    track_length, dt, debug, truncate=truncate)

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.75))

    this_axis = axes[1]
    this_axis.set_xlim(-tmax, tmax)
    induction_key = str(induction)
    ramp_list = []
    for cell_key in interp_delta_exp_ramp:
        if induction_key in interp_delta_exp_ramp[cell_key]:
            ramp_list.append(interp_delta_exp_ramp[cell_key][induction_key])
            this_axis.plot(np.divide(reference_delta_t, 1000.), interp_delta_exp_ramp[cell_key][induction_key],
                           c='darkgrey', linewidth=1., alpha=0.75)
    this_axis.plot(np.divide(reference_delta_t, 1000.), np.nanmean(ramp_list, axis=0), c=color)
    this_axis.set_title(label, fontsize=mpl.rcParams['font.size'], y=1.1)
    this_axis.set_ylabel('Change in ramp\namplitude (mV)')
    this_axis.set_xlabel('Time relative to\nplateau onset (s)')
    this_axis.set_yticks(np.arange(-4., 17., 4.))
    this_axis.set_xticks(np.arange(-4., 5., 2.))

    res = 50
    t_range = np.linspace(-tmax, tmax, res)

    this_axis = axes[2]
    this_axis.set_xlim(-tmax, tmax)
    interp_points = np.vstack((t_range, np.zeros(res))).T
    if show_std:
        voltage_independent_prediction, voltage_independent_prediction_std = \
            regressor.predict(interp_points, return_std=True)
    else:
        voltage_independent_prediction = regressor.predict(interp_points)
    this_axis.plot(t_range, voltage_independent_prediction, c='c', label='Voltage-independent')
    if show_std:
        this_axis.fill_between(t_range, np.add(voltage_independent_prediction, voltage_independent_prediction_std),
                               np.subtract(voltage_independent_prediction, voltage_independent_prediction_std),
                               color='c', alpha=0.25, linewidth=0)
    if group in depo_soma and len(depo_soma[group]) > 0 and group in group_indexes and \
            len(group_indexes[group]) == len(depo_soma[group]):
        vals = np.array(depo_soma[group])
    else:
        vals = np.ones(len(group_indexes[group])) * 10.
    ramp_list = []
    std_list = []
    for val in vals:
        interp_points = np.vstack((t_range, np.ones(res) * val)).T
        if show_std:
            voltage_dependent_prediction, voltage_dependent_prediction_std = \
                regressor.predict(interp_points, return_std=True)
            std_list.append(voltage_dependent_prediction_std)
        else:
            voltage_dependent_prediction = regressor.predict(interp_points)
        ramp_list.append(voltage_dependent_prediction)
        # this_axis.plot(t_range, voltage_dependent_prediction, c='darkgrey', linewidth=1., alpha=0.75)
    voltage_dependent_prediction = np.mean(ramp_list, axis=0)
    this_axis.plot(t_range, voltage_dependent_prediction, c='r', label='Voltage-dependent')
    if show_std:
        voltage_dependent_prediction_std = np.mean(std_list, axis=0)
        this_axis.fill_between(t_range, np.add(voltage_dependent_prediction, voltage_dependent_prediction_std),
                               np.subtract(voltage_dependent_prediction, voltage_dependent_prediction_std),
                               color='r', alpha=0.25, linewidth=0)
    this_axis.legend(loc='best', frameon=False, framealpha=0.5, fontsize=mpl.rcParams['font.size'], handletextpad=0.3,
                     handlelength=1.)
    this_axis.set_title('Kernel prediction', fontsize=mpl.rcParams['font.size'], y=1.1)
    this_axis.set_ylabel('Change in ramp\namplitude (mV)')
    this_axis.set_xlabel('Time relative to\nplateau onset (s)')
    this_axis.set_yticks(np.arange(-4., 17., 4.))
    this_axis.set_xticks(np.arange(-4., 5., 2.))
    clean_axes(axes)
    fig.subplots_adjust(left=0.1, wspace=0.6, right=0.975, bottom=0.25, top=0.85)
    fig.show()


def boxplot_compare_ramp_summary_features(model_file_path_dict, ordered_labels, induction=1):
    ramp_amp = defaultdict(list)
    ramp_width = defaultdict(list)
    local_peak_shift = defaultdict(list)

    induction_key = str(induction)
    exported_data_key = 'exported_data'
    description = 'model_ramp_features'
    for label, file_path in viewitems(model_file_path_dict):
        with h5py.File(file_path, 'r') as f:
            for cell_key in f[exported_data_key]:
                if induction_key in f[exported_data_key][cell_key]:
                    group = f[exported_data_key][cell_key][induction_key][description]
                    ramp_amp[label].append(group.attrs['target_ramp_amp'])
                    ramp_width[label].append(group.attrs['target_ramp_width'])
                    local_peak_shift[label].append(group.attrs['target_local_peak_shift'])

    fig, axes = plt.subplots(1, 3, figsize=(8, 3.25))
    colors = ['k', 'purple', 'c', 'r']
    for i, label in enumerate(ordered_labels):
        c = colors[i]
        axes[0].boxplot([ramp_amp[label]], positions=[i + 1], widths=[0.5], patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor='None', color=c), capprops=dict(color=c), whiskerprops=dict(color=c),
                        medianprops=dict(color=c))
        axes[1].boxplot([ramp_width[label]], positions=[i + 1], widths=[0.5], patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor='None', color=c), capprops=dict(color=c), whiskerprops=dict(color=c),
                        medianprops=dict(color=c))
        axes[2].boxplot([local_peak_shift[label]], positions=[i + 1], widths=[0.5], patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor='None', color=c), capprops=dict(color=c), whiskerprops=dict(color=c),
                        medianprops=dict(color=c))
    axes[0].set_title('Peak amplitude (mV)', fontsize=mpl.rcParams['font.size'], y=1.1)
    axes[0].set_xlim(0.25, float(len(ordered_labels)) + 0.75)
    axes[0].set_xticks(range(1, len(ordered_labels) + 1))
    axes[0].set_xticklabels(ordered_labels, rotation=45., ha='right')
    ylim = axes[0].get_ylim()
    axes[0].set_ylim(min(0., ylim[0]), max(16., ylim[1]))

    axes[1].set_title('Width (cm)', fontsize=mpl.rcParams['font.size'], y=1.1)
    axes[1].set_xlim(0.25, float(len(ordered_labels)) + 0.75)
    axes[1].set_xticks(range(1, len(ordered_labels) + 1))
    axes[1].set_xticklabels(ordered_labels, rotation=45., ha='right')
    ylim = axes[1].get_ylim()
    axes[1].set_ylim(min(45., ylim[0]), max(135., ylim[1]))

    axes[2].set_title('Peak shift (cm)', fontsize=mpl.rcParams['font.size'], y=1.1)
    axes[2].set_xlim(0.25, float(len(ordered_labels)) + 0.75)
    axes[2].set_xticks(range(1, len(ordered_labels) + 1))
    axes[2].set_xticklabels(ordered_labels, rotation=45., ha='right')
    ylim = axes[2].get_ylim()
    axes[2].set_ylim(min(-10., ylim[0]), max(0., ylim[1]))

    clean_axes(axes)
    fig.subplots_adjust(left=0.1, wspace=0.6, right=0.975, bottom=0.275, top=0.85)
    fig.show()


@click.command()
@click.option("--data-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='data/20190711_biBTSP_data_calibrated_input.hdf5')
@click.option("--vmax", type=float, default=12.97868)
@click.option("--tmax", type=float, default=5.)
@click.option("--truncate", type=bool, default=True)
@click.option("--debug", is_flag=True)
@click.option("--target-induction", type=int, multiple=True, default=[2])
@click.option("--font-size", type=float, default=11.)
def main(data_file_path, vmax, tmax, truncate, debug, target_induction, font_size):
    """

    :param data_file_path: str (path)
    :param vmax: float
    :param tmax: float
    :param truncate: bool
    :param debug: bool
    :param target_induction: tuple of int
    :param font_size: float
    """
    if not os.path.isfile(data_file_path):
        raise IOError('plot_biBTSP_data_summary_figure: invalid data_file_path: %s' % data_file_path)
    with h5py.File(data_file_path, 'r') as f:
        binned_x = f['defaults']['binned_x'][:]
        dt = f['defaults'].attrs['dt']
        track_length = f['defaults'].attrs['track_length']
    binned_extra_x = np.linspace(0., track_length, 101)
    extended_binned_x = np.concatenate([binned_x - track_length, binned_x, binned_x + track_length])
    reference_delta_t = np.linspace(-1000. * tmax, 1000. * tmax, 100)

    mpl.rcParams['font.size'] = font_size

    peak_ramp_amp, total_induction_dur, depo_soma, group_indexes, exp_ramp, extended_exp_ramp, delta_exp_ramp, \
        mean_induction_loc, extended_min_delta_t, extended_delta_exp_ramp, interp_initial_exp_ramp, \
        interp_delta_exp_ramp, interp_final_exp_ramp = \
        get_biBTSP_analysis_results(data_file_path, binned_x, binned_extra_x, extended_binned_x, reference_delta_t,
                                    track_length, dt, debug, truncate=truncate)

    context.update(locals())

    fig, axesgrid = plt.subplots(2, 2, figsize=(8.25, 6))
    axes = []
    for row in range(2):
        for col in range(2):
            axes.append(axesgrid[col][row])

    axes[0].scatter(total_induction_dur[group_indexes['exp1'] + group_indexes['spont']],
                    peak_ramp_amp[group_indexes['exp1'] + group_indexes['spont']], c='darkgrey', s=40., alpha=0.5,
                    linewidth=0)
    axes[0].scatter(total_induction_dur[group_indexes['exp2']], peak_ramp_amp[group_indexes['exp2']], c='k', s=40.,
                    alpha=0.5, linewidth=0)
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

    flat_min_t = []
    flat_delta_ramp = []
    flat_initial_ramp = []
    flat_final_ramp = []
    target_flat_delta_ramp = []
    target_flat_initial_ramp = []

    for cell_key in interp_delta_exp_ramp:
        for induction in target_induction:
            induction_key = str(induction)
            if induction_key in interp_delta_exp_ramp[cell_key]:
                if debug:
                    print('Including cell: %s, induction: %s' % (cell_key, induction_key))
                indexes = np.where(~np.isnan(interp_delta_exp_ramp[cell_key][induction_key]))[0]
                flat_min_t.extend(np.divide(reference_delta_t[indexes], 1000.))
                flat_delta_ramp.extend(interp_delta_exp_ramp[cell_key][induction_key][indexes])
                target_flat_delta_ramp.extend(interp_delta_exp_ramp[cell_key][induction_key][indexes])
                flat_initial_ramp.extend(interp_initial_exp_ramp[cell_key][induction_key][indexes])
                target_flat_initial_ramp.extend(interp_initial_exp_ramp[cell_key][induction_key][indexes])
                flat_final_ramp.extend(interp_final_exp_ramp[cell_key][induction_key][indexes])
        if 2 in target_induction and 1 not in target_induction and '2' in interp_delta_exp_ramp[cell_key] and \
                '1' in interp_delta_exp_ramp[cell_key]:
            induction_key = '1'
            if debug:
                print('Including cell: %s, induction: %s' % (cell_key, induction_key))
            indexes = np.where(~np.isnan(interp_delta_exp_ramp[cell_key][induction_key]))[0]
            flat_min_t.extend(np.divide(reference_delta_t[indexes], 1000.))
            flat_delta_ramp.extend(interp_delta_exp_ramp[cell_key][induction_key][indexes])
            flat_initial_ramp.extend(interp_initial_exp_ramp[cell_key][induction_key][indexes])
            flat_final_ramp.extend(interp_final_exp_ramp[cell_key][induction_key][indexes])

    ymax = np.max(target_flat_initial_ramp)
    ymin = min(0., np.min(target_flat_initial_ramp))

    lines_cmap = 'jet'
    interp_cmap = 'bwr_r'

    axes[1].set_xlim(-tmax, tmax)
    for cell_key in interp_delta_exp_ramp:
        for induction in target_induction:
            induction_key = str(induction)
            if induction_key in interp_delta_exp_ramp[cell_key]:
                indexes = np.where(~np.isnan(interp_delta_exp_ramp[cell_key][induction_key]))[0]
                lc = colorline(np.divide(reference_delta_t[indexes], 1000.),
                               interp_delta_exp_ramp[cell_key][induction_key][indexes],
                               interp_initial_exp_ramp[cell_key][induction_key][indexes],
                               vmin=ymin, vmax=ymax, cmap=lines_cmap)
                cax = axes[1].add_collection(lc)
    cbar = plt.colorbar(cax, ax=axes[1])
    cbar.set_label('Initial ramp\namplitude (mV)', rotation=270., labelpad=23.)
    axes[1].set_ylabel('Change in ramp\namplitude (mV)')
    axes[1].set_xlabel('Time relative to plateau onset (s)')
    axes[1].set_yticks(np.arange(-10., 16., 5.))
    axes[1].set_xticks(np.arange(-4., 5., 2.))

    if np.any(np.array(target_induction) != 1):
        axes[2].scatter(target_flat_initial_ramp, target_flat_delta_ramp, c='grey', s=5., alpha=0.5, linewidth=0.)
        fit_params = np.polyfit(target_flat_initial_ramp, target_flat_delta_ramp, 1)
        fit_f = np.vectorize(lambda x: fit_params[0] * x + fit_params[1])
        xlim = [0., math.ceil(np.max(target_flat_initial_ramp))]
        axes[2].plot(xlim, fit_f(xlim), c='k', alpha=0.75, zorder=1, linestyle='--')
        r_val, p_val = pearsonr(target_flat_initial_ramp, target_flat_delta_ramp)
        axes[2].annotate('R$^{2}$ = %.3f;\np %s %.3f' %
                         (r_val ** 2., '>' if p_val > 0.05 else '<', p_val if p_val > 0.001 else 0.001), xy=(0.6, 0.7),
                         xycoords='axes fraction')
        axes[2].set_xlim(xlim)
        axes[2].set_xlabel('Initial ramp amplitude (mV)')
        axes[2].set_ylabel('Change in ramp\namplitude (mV)')

        points = np.array([flat_min_t, flat_initial_ramp]).transpose()
        data = np.array(flat_delta_ramp)

        this_vmax = max(abs(np.max(flat_delta_ramp)), abs(np.min(flat_delta_ramp)))
        if this_vmax > vmax:
            print('New max detected for 3D data color legend: vmax: %.5f' % this_vmax)
            sys.stdout.flush()
            vmax = this_vmax

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