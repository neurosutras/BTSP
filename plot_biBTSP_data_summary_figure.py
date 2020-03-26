"""
To reproduce panels from Figure 1:
python -i plot_biBTSP_data_summary_figure.py --target-induction=2

To reproduce panels from Figure 3:
python -i plot_biBTSP_data_summary_figure.py --target-induction=1 --target-induction=2

plot_ramp_prediction_from_interpolation(context.gp,
    ['data/20200316_biBTSP_data_DC_soma_depo.h5', 'data/20200316_biBTSP_data_DC_soma_hyper.h5'],
    ['Depolarized', 'Hyperpolarized'], context.binned_x, context.binned_extra_x, context.extended_binned_x,
    context.reference_delta_t, context.track_length, context.dt, tmax=context.tmax, colors=['k', 'k'],
    induction_list=[1, 2], group_list=['exp1', 'exp2'], v_dep_show_std_list=['samples', 'samples'],
    v_indep_show_std_list=['kernel', 'samples'], show_both_predictions_list=[True, True],
    display_tmin_list=[-5., -1.], debug=False, truncate=True)

To reproduce panels from Figure S1:
plot_ramp_prediction_from_interpolation(context.gp,
    ['data/20190711_biBTSP_data_calibrated_input.hdf5'],
    ['Control'], context.binned_x, context.binned_extra_x, context.extended_binned_x,
    context.reference_delta_t, context.track_length, context.dt, tmax=context.tmax, colors=['k'],
    induction_list=[2], group_list=['exp2'], v_dep_show_std_list=['samples'],
    v_indep_show_std_list=['samples'], show_both_predictions_list=[False],
    debug=False, truncate=True)

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


def plot_ramp_prediction_from_interpolation(regressor, data_file_path_list, labels, binned_x, binned_extra_x,
                                            extended_binned_x, reference_delta_t, track_length, dt, tmax=5.,
                                            colors=['k'], induction_list=[1], group_list=['exp1'],
                                            v_dep_show_std_list=['samples'], v_indep_show_std_list=['samples'],
                                            show_both_predictions_list=[True], display_tmin_list=None, debug=False,
                                            truncate=True):
    """

    :param regressor: :class:'GaussianProcessRegressor'
    :param data_file_path_list: list of str (path)
    :param labels: list of str
    :param binned_x: array
    :param binned_extra_x: array
    :param extended_binned_x: array
    :param reference_delta_t: array
    :param track_length: float
    :param dt: float
    :param tmax: float
    :param colors: list of str
    :param induction_list: list of int
    :param group_list: list of str
    :param v_dep_show_std_list: list of str in ['samples', 'kernel'] or None
    :param v_indep_show_std_list: list of str in ['samples', 'kernel'] or None
    :param show_both_predictions_list: list of bool
    :param display_tmin_list: list of float or None
    :param debug: bool
    :param truncate: bool
    """
    fig, axes = plt.subplots(2, 3, figsize=(9., 6.75))
    for row, data_file_path in enumerate(data_file_path_list):
        if not os.path.isfile(data_file_path):
            raise IOError('plot_ramp_prediction_from_interpolation: invalid data_file_path: %s' % data_file_path)
        label = labels[row]
        color = colors[row]
        induction = induction_list[row]
        group = group_list[row]
        v_dep_show_std = v_dep_show_std_list[row]
        v_indep_show_std = v_indep_show_std_list[row]
        show_both_predictions = show_both_predictions_list[row]
        if display_tmin_list is None:
            display_tmin = -tmax
        else:
            display_tmin = display_tmin_list[row]
        initial_induction_delta_vm, initial_exp_ramp, delta_exp_ramp = \
            get_biBTSP_DC_soma_analysis_results(data_file_path, binned_x, binned_extra_x, extended_binned_x,
                                                reference_delta_t, track_length, dt, induction=induction,
                                                group=group, debug=debug, truncate=truncate)

        this_axis = axes[row][0]
        this_axis.set_xlim(display_tmin, tmax)
        ramp_list = []
        for this_delta_exp_ramp in delta_exp_ramp:
            ramp_list.append(this_delta_exp_ramp)
            this_axis.plot(np.divide(reference_delta_t, 1000.), this_delta_exp_ramp,
                           c='grey', linewidth=1., alpha=0.5)
        this_axis.plot(np.divide(reference_delta_t, 1000.), np.nanmean(ramp_list, axis=0), c=color)
        this_axis.set_title(label, fontsize=mpl.rcParams['font.size'], y=1.1)
        this_axis.set_ylabel('Change in ramp\namplitude (mV)')
        this_axis.set_xlabel('Time relative to\nplateau onset (s)')
        this_axis.set_yticks(np.arange(-10., 16., 5.))
        this_axis.set_ylim([-10., 16.])
        if display_tmin <= -4.:
            this_axis.set_xticks(np.arange(-4., tmax, 2.))
        elif display_tmin <= -1.:
            this_axis.set_xticks(np.arange(-1., tmax, 1.))
        this_xlim = this_axis.get_xlim()
        this_axis.plot([this_xlim[0], this_xlim[1]], [0., 0.], '--', c='darkgrey', alpha=0.75)

        this_axis = axes[row][1]
        this_axis.set_xlim(display_tmin, tmax)
        if not len(initial_induction_delta_vm) > 0:
            raise RuntimeError(
                'plot_ramp_prediction_from_interpolation: problem loading initial_induction_delta_vm for '
                'DC_soma experiments in group: %s' % group)
        flat_actual = []
        flat_v_dep_prediction = []
        flat_v_indep_prediction = []
        if len(initial_induction_delta_vm) > 1:
            v_dep_prediction_list = []
            v_dep_prediction_std_list = []
            v_indep_prediction_list = []
            v_indep_prediction_std_list = []
            for i in range(len(initial_induction_delta_vm)):
                this_initial_induction_delta_vm = initial_induction_delta_vm[i]
                this_initial_exp_ramp = initial_exp_ramp[i]
                indexes = np.where(~np.isnan(this_initial_exp_ramp))[0]
                flat_actual.extend(delta_exp_ramp[i][indexes])
                v_dep_points = np.vstack((reference_delta_t[indexes] / 1000.,
                                          this_initial_induction_delta_vm[indexes])).T
                v_indep_points = np.vstack((reference_delta_t[indexes] / 1000.,
                                            this_initial_exp_ramp[indexes])).T
                this_v_dep_prediction, this_v_dep_prediction_std = regressor.predict(v_dep_points, return_std=True)
                this_v_indep_prediction, this_v_indep_prediction_std = \
                    regressor.predict(v_indep_points, return_std=True)
                flat_v_dep_prediction.extend(this_v_dep_prediction)
                flat_v_indep_prediction.extend(this_v_indep_prediction)
                this_v_dep_prediction = \
                    np.interp(reference_delta_t, reference_delta_t[indexes], this_v_dep_prediction)
                bad_indexes = np.where(np.isnan(this_initial_exp_ramp))[0]
                this_v_dep_prediction[bad_indexes] = np.nan
                this_v_indep_prediction = \
                    np.interp(reference_delta_t, reference_delta_t[indexes], this_v_indep_prediction)
                this_v_indep_prediction[bad_indexes] = np.nan
                v_dep_prediction_list.append(this_v_dep_prediction)
                v_indep_prediction_list.append(this_v_indep_prediction)
                if v_dep_show_std == 'kernel':
                    this_v_dep_prediction_std = \
                        np.interp(reference_delta_t, reference_delta_t[indexes], this_v_dep_prediction_std)
                    v_dep_prediction_std_list.append(this_v_dep_prediction_std)
                if v_indep_show_std == 'kernel':
                    this_v_indep_prediction_std = \
                        np.interp(reference_delta_t, reference_delta_t[indexes], this_v_indep_prediction_std)
                    v_indep_prediction_std_list.append(this_v_indep_prediction_std)
            v_dep_prediction = np.nanmean(v_dep_prediction_list, axis=0)
            v_indep_prediction = np.nanmean(v_indep_prediction_list, axis=0)
            if v_dep_show_std == 'kernel':
                v_dep_prediction_std = np.nanmax(v_dep_prediction_std_list, axis=0)
            elif v_dep_show_std == 'samples':
                v_dep_prediction_std = np.nanstd(v_dep_prediction_list, axis=0)
            elif v_dep_show_std is None:
                v_dep_prediction_std = None
            else:
                raise RuntimeError('plot_ramp_prediction_from_interpolation: v_dep_show_std parameter not recognized: '
                                   '%s' % (v_dep_show_std))
            if v_indep_show_std == 'kernel':
                v_indep_prediction_std = np.nanmax(v_indep_prediction_std_list, axis=0)
            elif v_indep_show_std == 'samples':
                v_indep_prediction_std = np.nanstd(v_indep_prediction_list, axis=0)
            elif v_indep_show_std is None:
                v_indep_prediction_std = None
            else:
                raise RuntimeError('plot_ramp_prediction_from_interpolation: v_indep_show_std parameter not '
                                   'recognized: %s' % (v_indep_show_std))
        else:
            this_initial_induction_delta_vm = initial_induction_delta_vm[0]
            this_initial_exp_ramp = initial_exp_ramp[0]
            indexes = np.where(~np.isnan(this_initial_exp_ramp))[0]
            flat_actual.extend(delta_exp_ramp[i][indexes])
            v_dep_points = np.vstack((reference_delta_t[indexes] / 1000.,
                                      this_initial_induction_delta_vm[indexes])).T
            v_indep_points = np.vstack(
                (reference_delta_t[indexes] / 1000., this_initial_exp_ramp[indexes])).T
            v_dep_prediction, v_dep_prediction_std = \
                regressor.predict(v_dep_points, return_std=True)
            flat_v_dep_prediction.extend(v_dep_prediction)
            v_dep_prediction = \
                np.interp(reference_delta_t, reference_delta_t[indexes], v_dep_prediction)
            bad_indexes = np.where(np.isnan(this_initial_exp_ramp))[0]
            v_dep_prediction[bad_indexes] = np.nan
            v_dep_prediction_std = \
                np.interp(reference_delta_t, reference_delta_t[indexes], v_dep_prediction_std)
            v_dep_prediction_std[bad_indexes] = np.nan
            v_indep_prediction, v_indep_prediction_std = \
                regressor.predict(v_indep_points, return_std=True)
            flat_v_indep_prediction.extend(v_indep_prediction)
            v_indep_prediction = \
                np.interp(reference_delta_t, reference_delta_t[indexes], v_indep_prediction)
            v_indep_prediction[bad_indexes] = np.nan
            v_indep_prediction_std = \
                np.interp(reference_delta_t, reference_delta_t[indexes], v_indep_prediction_std)
            v_indep_prediction_std[bad_indexes] = np.nan
            if v_dep_show_std is None:
                v_dep_prediction_std = None
            elif v_dep_show_std not in ['kernel', 'samples']:
                raise RuntimeError('plot_ramp_prediction_from_interpolation: v_dep_show_std parameter not recognized: '
                                   '%s' % (v_dep_show_std))
            if v_indep_show_std is None:
                v_indep_prediction_std = None
            elif v_indep_show_std not in ['kernel', 'samples']:
                raise RuntimeError('plot_ramp_prediction_from_interpolation: v_indep_show_std parameter not '
                                   'recognized: %s' % (v_dep_show_std))

        if show_both_predictions:
            this_axis.plot(reference_delta_t / 1000., v_indep_prediction, c='c', label='Voltage-independent')
            if v_indep_prediction_std is not None:
                this_axis.fill_between(reference_delta_t / 1000.,
                                       np.add(v_indep_prediction, v_indep_prediction_std),
                                       np.subtract(v_indep_prediction, v_indep_prediction_std),
                                       color='c', alpha=0.25, linewidth=0)

            this_axis.plot(reference_delta_t / 1000., v_dep_prediction, c='r', label='Voltage-dependent')
            if v_dep_prediction_std is not None:
                this_axis.fill_between(reference_delta_t / 1000.,
                                       np.add(v_dep_prediction, v_dep_prediction_std),
                                       np.subtract(v_dep_prediction, v_dep_prediction_std),
                                       color='r', alpha=0.25, linewidth=0)
            this_axis.legend(loc='best', frameon=False, framealpha=0.5, fontsize=mpl.rcParams['font.size'],
                             handletextpad=0.3,
                             handlelength=1.)
            this_axis.set_title('Prediction', fontsize=mpl.rcParams['font.size'], y=1.1)
            this_axis.set_ylabel('Change in ramp\namplitude (mV)')
            this_axis.set_xlabel('Time relative to\nplateau onset (s)')
            this_axis.set_yticks(np.arange(-10., 16., 5.))
            this_axis.set_ylim([-10., 16.])
            if display_tmin <= -4.:
                this_axis.set_xticks(np.arange(-4., tmax, 2.))
            elif display_tmin <= -1.:
                this_axis.set_xticks(np.arange(-1., tmax, 1.))
            this_xlim = this_axis.get_xlim()
            this_axis.plot([this_xlim[0], this_xlim[1]], [0., 0.], '--', c='darkgrey', alpha=0.5)
            # this_axis.axhspan(0., 16., facecolor='c', alpha=0.15)
            # this_axis.axhspan(-6., 0., facecolor='r', alpha=0.15)

            this_axis = axes[row][2]
            this_axis.scatter(flat_actual, flat_v_dep_prediction, c='r', label='Voltage-dependent', linewidth=0,
                              alpha=0.25, s=10)
            this_axis.scatter(flat_actual, flat_v_indep_prediction, c='c', label='Voltage-independent',
                              linewidth=0,
                              alpha=0.25, s=10)
            this_axis.set_xlabel('Actual (mV)')
            this_axis.set_ylabel('Predicted (mV)')
            this_axis.set_yticks(np.arange(-10., 16., 5.))
            this_axis.set_ylim([-10., 16.])
            this_axis.set_xticks(np.arange(-10., 16., 5.))
            this_axis.set_xlim([-10., 16.])
            this_xlim = this_axis.get_xlim()
            this_axis.plot([this_xlim[0], this_xlim[1]], [this_xlim[0], this_xlim[1]], '--', c='darkgrey', alpha=0.75)
            this_axis.set_title('Change in\nramp amplitude', fontsize=mpl.rcParams['font.size'], y=1.1)

            r_val, p_val = pearsonr(flat_v_indep_prediction, flat_actual)
            this_axis.annotate('R$^{2}$ = %.3f; p %s %.3f' %
                               (r_val ** 2., '>' if p_val > 0.05 else '<', p_val if p_val > 0.001 else 0.001),
                               xy=(0.1, 0.9),
                               xycoords='axes fraction', color='c')
            r_val, p_val = pearsonr(flat_v_dep_prediction, flat_actual)
            this_axis.annotate('R$^{2}$ = %.3f; p %s %.3f' %
                               (r_val ** 2., '>' if p_val > 0.05 else '<', p_val if p_val > 0.001 else 0.001),
                               xy=(0.3, 0.025), xycoords='axes fraction', color='r')
        else:
            this_axis.plot(reference_delta_t / 1000., v_indep_prediction, c='k')
            if v_indep_prediction_std is not None:
                this_axis.fill_between(reference_delta_t / 1000.,
                                       np.add(v_indep_prediction, v_indep_prediction_std),
                                       np.subtract(v_indep_prediction, v_indep_prediction_std),
                                       color='grey', alpha=0.25, linewidth=0)

            this_axis.set_title('Prediction', fontsize=mpl.rcParams['font.size'], y=1.1)
            this_axis.set_ylabel('Change in ramp\namplitude (mV)')
            this_axis.set_xlabel('Time relative to\nplateau onset (s)')
            this_axis.set_yticks(np.arange(-10., 16., 5.))
            this_axis.set_ylim([-10., 16.])
            if display_tmin <= -4.:
                this_axis.set_xticks(np.arange(-4., tmax, 2.))
            elif display_tmin <= -1.:
                this_axis.set_xticks(np.arange(-1., tmax, 1.))
            this_xlim = this_axis.get_xlim()
            this_axis.plot([this_xlim[0], this_xlim[1]], [0., 0.], '--', c='darkgrey', alpha=0.5)

            this_axis = axes[row][2]
            this_axis.scatter(flat_actual, flat_v_indep_prediction, c='k', linewidth=0, alpha=0.25, s=10)
            this_axis.set_xlabel('Actual (mV)')
            this_axis.set_ylabel('Predicted (mV)')
            this_axis.set_yticks(np.arange(-10., 16., 5.))
            this_axis.set_ylim([-10., 16.])
            this_axis.set_xticks(np.arange(-10., 16., 5.))
            this_axis.set_xlim([-10., 16.])
            this_xlim = this_axis.get_xlim()
            this_axis.plot([this_xlim[0], this_xlim[1]], [this_xlim[0], this_xlim[1]], '--', c='darkgrey', alpha=0.75)
            this_axis.set_title('Change in\nramp amplitude', fontsize=mpl.rcParams['font.size'], y=1.1)

            r_val, p_val = pearsonr(flat_v_indep_prediction, flat_actual)
            this_axis.annotate('R$^{2}$ = %.3f; p %s %.3f' %
                               (r_val ** 2., '>' if p_val > 0.05 else '<', p_val if p_val > 0.001 else 0.001),
                               xy=(0.1, 0.9), xycoords='axes fraction', color='k')
        clean_axes(axes)
        fig.subplots_adjust(left=0.1, wspace=0.55, hspace=0.7, right=0.95, bottom=0.15, top=0.9)
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

    return ramp_amp, ramp_width, local_peak_shift


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

    peak_ramp_amp, total_induction_dur, initial_induction_delta_vm, group_indexes, exp_ramp, extended_exp_ramp, \
        delta_exp_ramp, mean_induction_loc, extended_min_delta_t, extended_delta_exp_ramp, interp_initial_exp_ramp, \
        interp_delta_exp_ramp, interp_final_exp_ramp = \
        get_biBTSP_analysis_results(data_file_path, binned_x, binned_extra_x, extended_binned_x, reference_delta_t,
                                    track_length, dt, debug, truncate=truncate)

    context.update(locals())

    fig, axesgrid = plt.subplots(3, 3, figsize=(12, 8.25))
    axes = []
    axes.append(axesgrid[1][0])
    for col in range(3):
        axes.append(axesgrid[2][col])

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
    labels = ['After Induction 1', 'After Induction 2']
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
    target_flat_min_t = []

    for cell_key in interp_delta_exp_ramp:
        for induction in target_induction:
            induction_key = str(induction)
            if induction_key in interp_delta_exp_ramp[cell_key]:
                if debug:
                    print('Including cell: %s, induction: %s' % (cell_key, induction_key))
                indexes = np.where(~np.isnan(interp_delta_exp_ramp[cell_key][induction_key]))[0]
                flat_min_t.extend(np.divide(reference_delta_t[indexes], 1000.))
                target_flat_min_t.extend(np.divide(reference_delta_t[indexes], 1000.))
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

    axes[2].set_xlim(-tmax, tmax)
    for cell_key in interp_delta_exp_ramp:
        for induction in target_induction:
            induction_key = str(induction)
            if induction_key in interp_delta_exp_ramp[cell_key]:
                indexes = np.where(~np.isnan(interp_delta_exp_ramp[cell_key][induction_key]))[0]
                lc = colorline(np.divide(reference_delta_t[indexes], 1000.),
                               interp_delta_exp_ramp[cell_key][induction_key][indexes],
                               interp_initial_exp_ramp[cell_key][induction_key][indexes],
                               vmin=ymin, vmax=ymax, cmap=lines_cmap)
                cax = axes[2].add_collection(lc)
    cbar = plt.colorbar(cax, ax=axes[2])
    cbar.set_label('Initial ramp\namplitude (mV)', rotation=270., labelpad=23.)
    axes[2].set_ylabel('Change in ramp\namplitude (mV)')
    axes[2].set_xlabel('Time relative to plateau onset (s)')
    axes[2].set_yticks(np.arange(-10., 16., 5.))
    axes[2].set_xticks(np.arange(-4., 5., 2.))

    if np.any(np.array(target_induction) != 1):
        axes[1].scatter(target_flat_initial_ramp, target_flat_delta_ramp, c='grey', s=5., alpha=0.5, linewidth=0.)
        fit_params = np.polyfit(target_flat_initial_ramp, target_flat_delta_ramp, 1)
        fit_f = np.vectorize(lambda x: fit_params[0] * x + fit_params[1])
        xlim = [0., math.ceil(np.max(target_flat_initial_ramp))]
        axes[1].plot(xlim, fit_f(xlim), c='k', alpha=0.75, zorder=1, linestyle='--')
        r_val, p_val = pearsonr(target_flat_initial_ramp, target_flat_delta_ramp)
        axes[1].annotate('R$^{2}$ = %.3f;\np %s %.3f' %
                         (r_val ** 2., '>' if p_val > 0.05 else '<', p_val if p_val > 0.001 else 0.001), xy=(0.6, 0.7),
                         xycoords='axes fraction')
        axes[1].set_xlim(xlim)
        axes[1].set_xlabel('Initial ramp amplitude (mV)')
        axes[1].set_ylabel('Change in ramp\namplitude (mV)')

        target_flat_delta_ramp = np.array(target_flat_delta_ramp)
        target_flat_initial_ramp = np.array(target_flat_initial_ramp)
        target_flat_min_t = np.array(target_flat_min_t)
        indexes = np.where(target_flat_delta_ramp <= 1.)[0]

        points = np.array([flat_min_t, flat_initial_ramp]).transpose()
        data = np.array(flat_delta_ramp)

        this_vmax = max(abs(np.max(flat_delta_ramp)), abs(np.min(flat_delta_ramp)))
        if this_vmax > vmax:
            print('New max detected for 3D data color legend: vmax: %.5f' % this_vmax)
            sys.stdout.flush()
            vmax = this_vmax
        if not debug:
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

            cax = axes[3].pcolor(t_grid, initial_ramp_grid, interp_data, cmap=interp_cmap, vmin=-vmax, vmax=vmax,
                                 zorder=0)
            axes[3].set_ylabel('Initial ramp\namplitude (mV)')
            axes[3].set_xlabel('Time relative to plateau onset (s)')
            axes[3].set_ylim(0., ymax)
            axes[3].set_xlim(-tmax, tmax)
            axes[3].set_xticks(np.arange(-4., 5., 2.))
            cbar = plt.colorbar(cax, ax=axes[3])
            cbar.set_label('Change in ramp\namplitude (mV)', rotation=270., labelpad=23.)
    clean_axes(axes)
    fig.subplots_adjust(hspace=0.6, wspace=0.66, left=0.085, right=0.945, top=0.9, bottom=0.11)
    fig.show()

    context.update(locals())


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)