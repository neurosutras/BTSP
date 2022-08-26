"""
To reproduce Figure 3F:
python -i plot_biBTSP_data_summary_figure.py

To use interpolation to predict the outcomes of the voltage perturbation experiments in Figures 4 and 5:
python -i plot_biBTSP_data_summary_figure.py --target-induction=1 --target-induction=2 --target-induction=3

plot_ramp_prediction_from_interpolation(context.gp,
    ['data/20200430_biBTSP_data_DC_soma_depo.h5', 'data/20200430_biBTSP_data_DC_soma_hyper.h5'],
    ['Depolarized\nSilent -> Place 1', 'Hyperpolarized\nPlace 1 -> Place 2'], context.reference_delta_t,
    colors=['k', 'k'], target_induction_list=[[1], [2]], show_both_predictions_list=[True, True],
    t_lim_list=[(-4., 4.), (-1., 4.)])

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


def plot_ramp_prediction_from_interpolation(regressor, data_file_path_list, labels, reference_delta_t,
                                            colors=None, target_induction_list=None, show_both_predictions_list=None,
                                            t_lim_list=None, debug=False, truncate=2.5):
    """

    :param regressor: :class:'GaussianProcessRegressor'
    :param data_file_path_list: list of str (path)
    :param labels: list of str
    :param reference_delta_t: array
    :param colors: list of str
    :param target_induction_list: list of list of int
    :param show_both_predictions_list: list of bool
    :param t_lim_list: list of tuple of float
    :param debug: bool
    :param truncate: bool
    """
    if colors is None:
        colors = ['k' for i in range(len(data_file_path_list))]
    if target_induction_list is None:
        target_induction_list = [[1] for i in range(len(data_file_path_list))]
    if show_both_predictions_list is None:
        show_both_predictions_list = [True for i in range(len(data_file_path_list))]
    if t_lim_list is None:
        t_lim_list = [(reference_delta_t[0] / 1000., reference_delta_t[-1] / 1000.)
                      for i in range(len(data_file_path_list))]

    fig, axes = plt.subplots(2, 3, figsize=(9., 6.75))
    fig2, axes2 = plt.subplots(1, 3, figsize=(11.5, 3.5))
    for row, data_file_path in enumerate(data_file_path_list):
        peak_ramp_amp, total_induction_dur, group_indexes, exp_ramp, delta_exp_ramp, exp_ramp_raw, delta_exp_ramp_raw, \
        mean_induction_loc, interp_exp_ramp, interp_delta_exp_ramp, interp_delta_exp_ramp_raw, min_induction_t, \
        clean_min_induction_t, clean_induction_t_indexes, initial_induction_delta_vm, baseline_vm = \
            get_biBTSP_data_analysis_results(data_file_path, reference_delta_t, debug=debug, truncate=truncate)

        label = labels[row]
        color = colors[row]
        target_induction = target_induction_list[row]
        show_both_predictions = show_both_predictions_list[row]
        tmin = t_lim_list[row][0]
        tmax = t_lim_list[row][1]

        this_axis = axes[row][0]
        this_axis.set_xlim(tmin, tmax)
        interp_delta_exp_ramp_list = []
        min_induction_t_list = []
        delta_exp_ramp_list = []
        initial_exp_ramp_list = []
        initial_induction_delta_vm_list = []
        for cell_key in delta_exp_ramp:
            for induction in target_induction:
                induction_key = str(induction)
                if induction_key in delta_exp_ramp[cell_key]:
                    interp_delta_exp_ramp_list.append(interp_delta_exp_ramp[cell_key][induction_key])
                    this_clean_indexes = clean_induction_t_indexes[cell_key][induction_key]
                    min_induction_t_list.append(clean_min_induction_t[cell_key][induction_key])
                    delta_exp_ramp_list.append(delta_exp_ramp[cell_key][induction_key][this_clean_indexes])
                    initial_exp_ramp_list.append(exp_ramp[cell_key][induction_key]['before'][this_clean_indexes])
                    initial_induction_delta_vm_list.append(
                        initial_induction_delta_vm[cell_key][induction_key][this_clean_indexes])
            if 2 in target_induction and 1 not in target_induction and '2' in delta_exp_ramp[cell_key] and \
                    '1' in delta_exp_ramp[cell_key]:
                induction_key = '1'
                interp_delta_exp_ramp_list.append(interp_delta_exp_ramp[cell_key][induction_key])
                this_clean_indexes = clean_induction_t_indexes[cell_key][induction_key]
                min_induction_t_list.append(clean_min_induction_t[cell_key][induction_key])
                delta_exp_ramp_list.append(delta_exp_ramp[cell_key][induction_key][this_clean_indexes])
                initial_exp_ramp_list.append(exp_ramp[cell_key][induction_key]['before'][this_clean_indexes])
                initial_induction_delta_vm_list.append(
                    initial_induction_delta_vm[cell_key][induction_key][this_clean_indexes])
        for this_interp_delta_exp_ramp in interp_delta_exp_ramp_list:
            this_axis.plot(np.divide(reference_delta_t, 1000.), this_interp_delta_exp_ramp, c='grey', linewidth=1.,
                           alpha=0.5)
        this_axis.plot(np.divide(reference_delta_t, 1000.), np.nanmean(interp_delta_exp_ramp_list, axis=0), c=color)
        this_axis.set_title(label, fontsize=mpl.rcParams['font.size'], y=1.05)
        this_axis.set_ylabel(r'$\Delta$Vm (mV)')
        this_axis.set_xlabel('Time from plateau (s)')
        this_axis.set_yticks(np.arange(-10., 16., 5.))
        this_axis.set_ylim([-10., 16.])
        if tmin <= -4.:
            this_axis.set_xticks(np.arange(-4., tmax + 0.1, 2.))
        elif tmin <= -1.:
            this_axis.set_xticks(np.arange(-1., tmax + 0.1, 1.))
        this_xlim = this_axis.get_xlim()
        this_axis.plot([this_xlim[0], this_xlim[1]], [0., 0.], '--', c='darkgrey', alpha=0.5)

        this_axis = axes[row][1]
        this_axis.set_xlim(tmin, tmax)
        flat_actual = []
        flat_v_dep_prediction = []
        flat_v_indep_prediction = []
        interp_v_dep_prediction_list = []
        interp_v_indep_prediction_list = []
        for i in range(len(initial_induction_delta_vm_list)):
            this_initial_induction_delta_vm = initial_induction_delta_vm_list[i]
            this_initial_exp_ramp = initial_exp_ramp_list[i]
            this_min_induction_t = min_induction_t_list[i]
            flat_actual.extend(delta_exp_ramp_list[i])
            v_dep_points = np.vstack((np.divide(this_min_induction_t, 1000.), this_initial_induction_delta_vm)).T
            v_indep_points = np.vstack((np.divide(this_min_induction_t, 1000.), this_initial_exp_ramp)).T
            this_v_dep_prediction = regressor.predict(v_dep_points)
            this_v_indep_prediction = regressor.predict(v_indep_points)
            flat_v_dep_prediction.extend(this_v_dep_prediction)
            flat_v_indep_prediction.extend(this_v_indep_prediction)
            bad_indexes = np.where((reference_delta_t < np.min(this_min_induction_t)) |
                                   (reference_delta_t > np.max(this_min_induction_t)))[0]
            this_interp_v_dep_prediction = \
                np.interp(reference_delta_t, this_min_induction_t, this_v_dep_prediction)
            this_interp_v_dep_prediction[bad_indexes] = np.nan
            interp_v_dep_prediction_list.append(this_interp_v_dep_prediction)
            this_interp_v_indep_prediction = \
                np.interp(reference_delta_t, this_min_induction_t, this_v_indep_prediction)
            this_interp_v_indep_prediction[bad_indexes] = np.nan
            interp_v_indep_prediction_list.append(this_interp_v_indep_prediction)
        v_dep_prediction = np.nanmean(interp_v_dep_prediction_list, axis=0)
        v_dep_prediction_std = np.nanstd(interp_v_dep_prediction_list, axis=0)
        v_indep_prediction = np.nanmean(interp_v_indep_prediction_list, axis=0)
        v_indep_prediction_std = np.nanstd(interp_v_indep_prediction_list, axis=0)

        if show_both_predictions:
            this_axis.plot(reference_delta_t / 1000., v_indep_prediction, c='c', label='Voltage-independent')
            this_axis.fill_between(reference_delta_t / 1000.,
                                   np.add(v_indep_prediction, v_indep_prediction_std),
                                   np.subtract(v_indep_prediction, v_indep_prediction_std),
                                   color='c', alpha=0.25, linewidth=0)
            this_axis.plot(reference_delta_t / 1000., v_dep_prediction, c='r', label='Voltage-dependent')
            this_axis.fill_between(reference_delta_t / 1000.,
                                   np.add(v_dep_prediction, v_dep_prediction_std),
                                   np.subtract(v_dep_prediction, v_dep_prediction_std),
                                   color='r', alpha=0.25, linewidth=0)
            this_axis.legend(loc='best', frameon=False, framealpha=0.5, fontsize=mpl.rcParams['font.size'],
                             handletextpad=0.3,
                             handlelength=1.)
            this_axis.set_title('Prediction', fontsize=mpl.rcParams['font.size'], y=1.05)
            this_axis.set_ylabel(r'$\Delta$Vm (mV)')
            this_axis.set_xlabel('Time from plateau (s)')
            this_axis.set_yticks(np.arange(-10., 16., 5.))
            this_axis.set_ylim([-10., 16.])
            if tmin <= -4.:
                this_axis.set_xticks(np.arange(-4., tmax + 0.1, 2.))
            elif tmin <= -1.:
                this_axis.set_xticks(np.arange(-1., tmax + 0.1, 1.))
            this_xlim = this_axis.get_xlim()
            this_axis.plot([this_xlim[0], this_xlim[1]], [0., 0.], '--', c='darkgrey', alpha=0.5)
            # this_axis.axhspan(0., 16., facecolor='c', alpha=0.15)
            # this_axis.axhspan(-6., 0., facecolor='r', alpha=0.15)

            this_axis = axes2[row]
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
            this_axis.plot([this_xlim[0], this_xlim[1]], [this_xlim[0], this_xlim[1]], '--', c='darkgrey', alpha=0.5)
            this_axis.set_title(r'$\Delta$Vm (mV)', fontsize=mpl.rcParams['font.size'], y=1.05)

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
            this_axis.plot(reference_delta_t / 1000., v_dep_prediction, c='purple')
            this_axis.fill_between(reference_delta_t / 1000.,
                                   np.add(v_dep_prediction, v_dep_prediction_std),
                                   np.subtract(v_dep_prediction, v_dep_prediction_std),
                                   color='purple', alpha=0.25, linewidth=0)

            this_axis.set_title('Prediction', fontsize=mpl.rcParams['font.size'], y=1.05)
            this_axis.set_ylabel(r'$\Delta$Vm (mV)')
            this_axis.set_xlabel('Time from plateau (s)')
            this_axis.set_yticks(np.arange(-10., 16., 5.))
            this_axis.set_ylim([-10., 16.])
            if tmin <= -4.:
                this_axis.set_xticks(np.arange(-4., tmax + 0.1, 2.))
            elif tmin <= -1.:
                this_axis.set_xticks(np.arange(-1., tmax + 0.1, 1.))
            this_xlim = this_axis.get_xlim()
            this_axis.plot([this_xlim[0], this_xlim[1]], [0., 0.], '--', c='darkgrey', alpha=0.5)

            this_axis = axes[row][2]
            this_axis.scatter(flat_actual, flat_v_dep_prediction, c='k', linewidth=0, alpha=0.25, s=10)
            this_axis.set_xlabel('Actual (mV)')
            this_axis.set_ylabel('Predicted (mV)')
            this_axis.set_yticks(np.arange(-10., 16., 5.))
            this_axis.set_ylim([-10., 16.])
            this_axis.set_xticks(np.arange(-10., 16., 5.))
            this_axis.set_xlim([-10., 16.])
            this_xlim = this_axis.get_xlim()
            this_axis.plot([this_xlim[0], this_xlim[1]], [this_xlim[0], this_xlim[1]], '--', c='darkgrey', alpha=0.5)
            this_axis.set_title(r'$\Delta$Vm (mV)', fontsize=mpl.rcParams['font.size'], y=1.05)

            r_val, p_val = pearsonr(flat_v_dep_prediction, flat_actual)
            this_axis.annotate('R$^{2}$ = %.3f; p %s %.3f' %
                               (r_val ** 2., '>' if p_val > 0.05 else '<', p_val if p_val > 0.001 else 0.001),
                               xy=(0.1, 0.9), xycoords='axes fraction', color='k')
        clean_axes(axes)
        fig.subplots_adjust(left=0.1, wspace=0.55, hspace=0.7, right=0.95, bottom=0.15, top=0.9)
        fig.show()
        clean_axes(axes2)
        fig2.tight_layout()
        fig2.subplots_adjust(top=0.75, hspace=0.2, wspace=0.6)
        fig2.show()


def plot_ramp_prediction_from_interpolation_alt(regressor, data_file_path_list, labels, reference_delta_t, colors=None,
                                                target_induction_list=None, t_lim_list=None, debug=False, truncate=2.5):
    """

    :param regressor: :class:'GaussianProcessRegressor'
    :param data_file_path_list: list of str (path)
    :param labels: list of str
    :param reference_delta_t: array
    :param colors: list of str
    :param target_induction_list: list of list of int
    :param t_lim_list: list of tuple of float
    :param debug: bool
    :param truncate: bool
    """
    if colors is None:
        colors = ['k' for i in range(len(data_file_path_list))]
    if target_induction_list is None:
        target_induction_list = [[1] for i in range(len(data_file_path_list))]
    if t_lim_list is None:
        t_lim_list = [(reference_delta_t[0] / 1000., reference_delta_t[-1] / 1000.)
                      for i in range(len(data_file_path_list))]

    fig, axes = plt.subplots(2, 3, figsize=(9., 6.75))
    fig2, axes2 = plt.subplots(1, 3, figsize=(11.5, 3.5))
    for row, data_file_path in enumerate(data_file_path_list):
        peak_ramp_amp, total_induction_dur, group_indexes, exp_ramp, delta_exp_ramp, exp_ramp_raw, delta_exp_ramp_raw, \
        mean_induction_loc, interp_exp_ramp, interp_delta_exp_ramp, interp_delta_exp_ramp_raw, min_induction_t, \
        clean_min_induction_t, clean_induction_t_indexes, initial_induction_delta_vm, baseline_vm = \
            get_biBTSP_data_analysis_results(data_file_path, reference_delta_t, debug=debug, truncate=truncate)

        label = labels[row]
        color = colors[row]
        target_induction = target_induction_list[row]
        tmin = t_lim_list[row][0]
        tmax = t_lim_list[row][1]

        this_axis = axes[row][0]
        this_axis.set_xlim(tmin, tmax)
        interp_delta_exp_ramp_list = []
        min_induction_t_list = []
        delta_exp_ramp_list = []
        initial_exp_ramp_list = []
        initial_induction_delta_vm_list = []
        for cell_key in delta_exp_ramp:
            for induction in target_induction:
                induction_key = str(induction)
                if induction_key in delta_exp_ramp[cell_key]:
                    interp_delta_exp_ramp_list.append(interp_delta_exp_ramp[cell_key][induction_key])
                    this_clean_indexes = clean_induction_t_indexes[cell_key][induction_key]
                    min_induction_t_list.append(clean_min_induction_t[cell_key][induction_key])
                    delta_exp_ramp_list.append(delta_exp_ramp[cell_key][induction_key][this_clean_indexes])
                    initial_exp_ramp_list.append(exp_ramp[cell_key][induction_key]['before'][this_clean_indexes])
                    initial_induction_delta_vm_list.append(
                        initial_induction_delta_vm[cell_key][induction_key][this_clean_indexes])
            if 2 in target_induction and 1 not in target_induction and '2' in delta_exp_ramp[cell_key] and \
                    '1' in delta_exp_ramp[cell_key]:
                induction_key = '1'
                interp_delta_exp_ramp_list.append(interp_delta_exp_ramp[cell_key][induction_key])
                this_clean_indexes = clean_induction_t_indexes[cell_key][induction_key]
                min_induction_t_list.append(clean_min_induction_t[cell_key][induction_key])
                delta_exp_ramp_list.append(delta_exp_ramp[cell_key][induction_key][this_clean_indexes])
                initial_exp_ramp_list.append(exp_ramp[cell_key][induction_key]['before'][this_clean_indexes])
                initial_induction_delta_vm_list.append(
                    initial_induction_delta_vm[cell_key][induction_key][this_clean_indexes])

        delta_exp_ramp_mean = np.nanmean(interp_delta_exp_ramp_list, axis=0)
        delta_exp_ramp_std = np.nanstd(interp_delta_exp_ramp_list, axis=0)
        this_axis.plot(np.divide(reference_delta_t, 1000.), delta_exp_ramp_mean, c=color, label='Actual')
        this_axis.fill_between(reference_delta_t / 1000.,
                               np.add(delta_exp_ramp_mean, delta_exp_ramp_std),
                               np.subtract(delta_exp_ramp_mean, delta_exp_ramp_std),
                               color='grey', alpha=0.25, linewidth=0)
        this_axis.set_title(label, fontsize=mpl.rcParams['font.size'], y=1.05)
        this_axis.set_ylabel(r'$\Delta$Vm (mV)')
        this_axis.set_xlabel('Time from plateau (s)')
        this_axis.set_yticks(np.arange(-10., 16., 5.))
        this_axis.set_ylim([-10., 16.])
        if tmin <= -4.:
            this_axis.set_xticks(np.arange(-4., tmax + 0.1, 2.))
        elif tmin <= -1.:
            this_axis.set_xticks(np.arange(-1., tmax + 0.1, 1.))
        this_xlim = this_axis.get_xlim()
        this_axis.plot([this_xlim[0], this_xlim[1]], [0., 0.], '--', c='darkgrey', alpha=0.5)

        flat_actual = []
        flat_v_dep_prediction = []
        interp_v_dep_prediction_list = []
        for i in range(len(initial_induction_delta_vm_list)):
            this_initial_induction_delta_vm = initial_induction_delta_vm_list[i]
            this_min_induction_t = min_induction_t_list[i]
            flat_actual.extend(delta_exp_ramp_list[i])
            v_dep_points = np.vstack((np.divide(this_min_induction_t, 1000.), this_initial_induction_delta_vm)).T
            this_v_dep_prediction = regressor.predict(v_dep_points)
            flat_v_dep_prediction.extend(this_v_dep_prediction)
            bad_indexes = np.where((reference_delta_t < np.min(this_min_induction_t)) |
                                   (reference_delta_t > np.max(this_min_induction_t)))[0]
            this_interp_v_dep_prediction = \
                np.interp(reference_delta_t, this_min_induction_t, this_v_dep_prediction)
            this_interp_v_dep_prediction[bad_indexes] = np.nan
            interp_v_dep_prediction_list.append(this_interp_v_dep_prediction)
        v_dep_prediction = np.nanmean(interp_v_dep_prediction_list, axis=0)
        v_dep_prediction_std = np.nanstd(interp_v_dep_prediction_list, axis=0)

        this_axis.plot(reference_delta_t / 1000., v_dep_prediction, c='purple', label='Predicted')
        this_axis.fill_between(reference_delta_t / 1000.,
                               np.add(v_dep_prediction, v_dep_prediction_std),
                               np.subtract(v_dep_prediction, v_dep_prediction_std),
                               color='purple', alpha=0.25, linewidth=0)
        this_axis.legend(loc='best', frameon=False, framealpha=0., handlelength=1)

        this_axis = axes[row][2]
        this_axis.scatter(flat_actual, flat_v_dep_prediction, c='k', linewidth=0, alpha=0.25, s=10)
        this_axis.set_xlabel('Actual (mV)')
        this_axis.set_ylabel('Predicted (mV)')
        this_axis.set_yticks(np.arange(-10., 16., 5.))
        this_axis.set_ylim([-10., 16.])
        this_axis.set_xticks(np.arange(-10., 16., 5.))
        this_axis.set_xlim([-10., 16.])
        this_xlim = this_axis.get_xlim()
        this_axis.plot([this_xlim[0], this_xlim[1]], [this_xlim[0], this_xlim[1]], '--', c='darkgrey', alpha=0.75)
        this_axis.set_title(r'$\Delta$Vm (mV)', fontsize=mpl.rcParams['font.size'], y=1.05)

        r_val, p_val = pearsonr(flat_v_dep_prediction, flat_actual)
        this_axis.annotate('R$^{2}$ = %.3f; p %s %.3f' %
                           (r_val ** 2., '>' if p_val > 0.05 else '<', p_val if p_val > 0.001 else 0.001),
                           xy=(0.1, 0.9), xycoords='axes fraction', color='k')
        clean_axes(axes)
        fig.subplots_adjust(left=0.1, wspace=0.55, hspace=0.7, right=0.95, bottom=0.15, top=0.9)
        fig.show()
        clean_axes(axes2)
        fig2.tight_layout()
        fig2.subplots_adjust(top=0.75, hspace=0.2, wspace=0.6)
        fig2.show()


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
              default='data/20201123_biBTSP_data.hdf5')
@click.option("--vmax", type=float, default=12.97868)
@click.option("--tmax", type=float, default=5.)
@click.option("--truncate", type=float, default=2.5)
@click.option("--debug", is_flag=True)
@click.option("--target-induction", '-i', type=int, multiple=True, default=[1, 2, 3])
@click.option("--font-size", type=float, default=12.)
def main(data_file_path, vmax, tmax, truncate, debug, target_induction, font_size):
    """

    :param data_file_path: str (path)
    :param vmax: float
    :param tmax: float
    :param truncate: float
    :param debug: bool
    :param target_induction: tuple of int
    :param font_size: float
    """
    mpl.rcParams['font.size'] = font_size

    reference_delta_t = np.linspace(-1000. * tmax, 1000. * tmax, 100)

    peak_ramp_amp, total_induction_dur, group_indexes, exp_ramp, delta_exp_ramp, exp_ramp_raw, delta_exp_ramp_raw, \
    mean_induction_loc, interp_exp_ramp, interp_delta_exp_ramp, interp_delta_exp_ramp_raw, min_induction_t, \
    clean_min_induction_t, clean_induction_t_indexes, initial_induction_delta_vm, baseline_vm = \
        get_biBTSP_data_analysis_results(data_file_path, reference_delta_t, debug=debug, truncate=truncate)

    context.update(locals())

    fig1, axes1 = plt.subplots(1, 3, figsize=(8.5, 3.))
    fig2, axes2 = plt.subplots(1, 3, figsize=(11.5, 3.5))
    fig3, axes3 = plt.subplots(1, 3, figsize=(8.5, 2.5))

    flat_min_t = []
    flat_delta_ramp = []
    flat_initial_ramp = []
    target_flat_delta_ramp = []
    target_flat_initial_ramp = []
    target_flat_min_t = []

    for cell_key in interp_delta_exp_ramp:
        for induction in target_induction:
            induction_key = str(induction)
            if induction_key in interp_delta_exp_ramp[cell_key]:
                if debug:
                    print('Including cell: %s, induction: %s' % (cell_key, induction_key))
                this_clean_indexes = clean_induction_t_indexes[cell_key][induction_key]
                flat_min_t.extend(np.divide(clean_min_induction_t[cell_key][induction_key], 1000.))
                target_flat_min_t.extend(np.divide(clean_min_induction_t[cell_key][induction_key], 1000.))
                flat_delta_ramp.extend(delta_exp_ramp[cell_key][induction_key][this_clean_indexes])
                target_flat_delta_ramp.extend(delta_exp_ramp[cell_key][induction_key][this_clean_indexes])
                flat_initial_ramp.extend(exp_ramp[cell_key][induction_key]['before'][this_clean_indexes])
                target_flat_initial_ramp.extend(exp_ramp[cell_key][induction_key]['before'][this_clean_indexes])
        if 2 in target_induction and 1 not in target_induction and '2' in interp_delta_exp_ramp[cell_key] and \
                '1' in interp_delta_exp_ramp[cell_key]:
            induction_key = '1'
            if debug:
                print('Including cell: %s, induction: %s' % (cell_key, induction_key))
            this_clean_indexes = clean_induction_t_indexes[cell_key][induction_key]
            flat_min_t.extend(np.divide(clean_min_induction_t[cell_key][induction_key], 1000.))
            flat_delta_ramp.extend(delta_exp_ramp[cell_key][induction_key][this_clean_indexes])
            flat_initial_ramp.extend(exp_ramp[cell_key][induction_key]['before'][this_clean_indexes])

    ymax = np.max(target_flat_initial_ramp)
    ymin = min(0., np.min(target_flat_initial_ramp))

    lines_cmap = 'jet'
    interp_cmap = 'bwr'

    this_axis = axes3[2]
    this_axis.set_xlim(-tmax, tmax)
    for cell_key in interp_delta_exp_ramp:
        for induction in target_induction:
            induction_key = str(induction)
            if induction_key in interp_delta_exp_ramp[cell_key]:
                lc = colorline(np.divide(reference_delta_t, 1000.),
                               interp_delta_exp_ramp[cell_key][induction_key],
                               interp_exp_ramp[cell_key][induction_key]['before'],
                               vmin=ymin, vmax=ymax, cmap=lines_cmap)
                cax = this_axis.add_collection(lc)
    cbar = plt.colorbar(cax, ax=this_axis)
    cbar.set_label('Initial Vm ramp\namplitude (mV)', rotation=270., labelpad=23.)
    this_axis.set_ylabel(r'$\Delta$Vm (mV)')
    this_axis.set_xlabel('Time from plateau (s)')
    this_axis.set_yticks(np.arange(-10., 16., 5.))
    this_axis.set_xticks(np.arange(-4., 5., 2.))

    this_axis = axes1[1]
    this_axis.scatter(total_induction_dur[group_indexes['exp1'] + group_indexes['spont']],
                      peak_ramp_amp[group_indexes['exp1'] + group_indexes['spont']], c='darkgrey', s=40., alpha=0.5,
                      linewidth=0)
    this_axis.scatter(total_induction_dur[group_indexes['exp2'] + group_indexes['exp3']],
                      peak_ramp_amp[group_indexes['exp2'] + group_indexes['exp3']], c='k', s=40.,
                      alpha=0.5, linewidth=0)
    this_axis.set_ylabel('Peak Vm ramp\namplitude (mV)')
    this_axis.set_xlabel('Total accumulated\nplateau duration (ms)')
    ylim = [0., math.ceil(np.max(peak_ramp_amp) + 3.)]
    xlim = [0., (math.ceil(np.max(total_induction_dur) / 100.) + 1.) * 100.]
    result = np.polyfit(total_induction_dur, peak_ramp_amp, 1)
    fit = np.vectorize(lambda x: result[0] * x + result[1])
    fit_xlim = [(math.floor(np.min(total_induction_dur) / 100.) - 1.) * 100.,
                (math.ceil(np.max(total_induction_dur) / 100.) + 1.) * 100.]
    this_axis.plot(fit_xlim, fit(fit_xlim), c='grey', alpha=0.5, zorder=0, linestyle='--')
    this_axis.set_ylim(ylim)
    this_axis.set_xlim(xlim)
    this_axis.set_xticks(np.arange(0., xlim[1], 1000.))
    handles = [mlines.Line2D([0], [0], linestyle='none', mfc=color, mew=0, alpha=0.5, marker='o', ms=math.sqrt(40.))
               for color in ['darkgrey', 'k']]
    labels = ['Silent -> Place 1', 'Place 1 -> Place 2']
    this_axis.legend(handles=handles, labels=labels, loc=(0.05, 0.95), frameon=False, framealpha=0.5, handletextpad=0.3,
                     handlelength=1.)
    r_val, p_val = pearsonr(total_induction_dur, peak_ramp_amp)
    this_axis.annotate('R$^{2}$ = %.3f; p %s %.3f' %
                       (r_val ** 2., '>' if p_val > 0.05 else '<', p_val if p_val > 0.001 else 0.001), xy=(0.03, 0.025),
                       xycoords='axes fraction')
    # cbar = plt.colorbar(cax, ax=this_axis)
    # cbar.ax.set_visible(False)

    if np.any(np.array(target_induction) != 1):
        this_axis = axes1[0]
        this_axis.scatter(target_flat_initial_ramp, target_flat_delta_ramp, c='grey', s=5., alpha=0.5, linewidth=0.)
        fit_params = np.polyfit(target_flat_initial_ramp, target_flat_delta_ramp, 1)
        fit_f = np.vectorize(lambda x: fit_params[0] * x + fit_params[1])
        xlim = [0., math.ceil(np.max(target_flat_initial_ramp))]
        this_axis.plot(xlim, fit_f(xlim), c='k', alpha=0.75, zorder=1, linestyle='--')
        r_val, p_val = pearsonr(target_flat_initial_ramp, target_flat_delta_ramp)
        this_axis.annotate('R$^{2}$ = %.3f;\np %s %.3f' %
                         (r_val ** 2., '>' if p_val > 0.05 else '<', p_val if p_val > 0.001 else 0.001), xy=(0.6, 0.7),
                         xycoords='axes fraction')
        this_axis.set_xlim(xlim)
        this_axis.set_xlabel('Initial Vm ramp\namplitude (mV)')
        this_axis.set_ylabel(r'$\Delta$Vm (mV)')

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
            res = len(reference_delta_t)
            t_range = np.divide(reference_delta_t, 1000.)
            initial_ramp_range = np.linspace(np.min(flat_initial_ramp), np.max(flat_initial_ramp), res)

            t_grid, initial_ramp_grid = np.meshgrid(t_range, initial_ramp_range)
            interp_points = np.vstack((t_grid.flatten(), initial_ramp_grid.flatten())).T

            kernel = RationalQuadratic(1., length_scale_bounds=(1e-10, 10.))
            gp = GaussianProcessRegressor(kernel=kernel, alpha=3., n_restarts_optimizer=20, normalize_y=True)
            start_time = time.time()
            print('Starting Gaussian Process Regression with %i samples' % len(data))
            gp.fit(points, data)
            print('Gaussian Process Regression took %.1f s' % (time.time() - start_time))
            current_time = time.time()
            interp_data = gp.predict(interp_points).reshape(-1, res)
            print('Gaussian Process Interpolation took %.1f s' % (time.time() - current_time))
            for this_axis in [axes2[0], axes3[0]]:
                cax = this_axis.pcolormesh(t_grid, initial_ramp_grid, interp_data, cmap=interp_cmap, vmin=-vmax,
                                           vmax=vmax, zorder=0, edgecolors='face', rasterized=True)
                this_axis.set_ylabel('Initial Vm ramp\namplitude (mV)')
                this_axis.set_xlabel('Time from plateau (s)')
                this_axis.set_ylim(0., ymax)
                this_axis.set_xlim(-tmax, tmax)
                this_axis.set_xticks(np.arange(-4., 5., 2.))
                cbar = plt.colorbar(cax, ax=this_axis)
                cbar.set_label(r'$\Delta$Vm (mV)', rotation=270., labelpad=15.)

    clean_axes(axes1)
    # fig.set_constrained_layout_pads(wspace=0.08, hspace=0.12)
    fig1.subplots_adjust(hspace=0.6, wspace=0.66, left=0.085, right=0.945, top=0.825, bottom=0.225)
    fig1.show()

    clean_axes(axes2)
    # fig2.suptitle('Experimental data', y=0.95, x=0.05, ha='left', fontsize=mpl.rcParams['font.size'])  # y=0.95,
    fig2.tight_layout()
    fig2.subplots_adjust(top=0.75, hspace=0.2, wspace=0.6)
    fig2.show()

    clean_axes(axes3)
    fig3.suptitle('Experimental data', y=0.95, x=0.05, ha='left', fontsize=mpl.rcParams['font.size'])  # y=0.95,
    fig3.tight_layout()
    fig3.subplots_adjust(top=0.8, hspace=0.2, wspace=0.6, bottom=0.2)
    fig3.show()

    context.update(locals())


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)