from biBTSP_utils import *
from nested.optimize_utils import *
import matplotlib as mpl
import click
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from scipy.stats import pearsonr
from collections import defaultdict

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['text.usetex'] = False
mpl.rcParams['axes.titlepad'] = 2.
mpl.rcParams['mathtext.default'] = 'regular'

context = Context()


@click.command()
@click.option("--data-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='data/20200427_biBTSP_data.hdf5')
@click.option("--model-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='data/20200429_biBTSP_WD_E_90cm_exported_model_output.hdf5')
@click.option("--vmax", type=float, default=12.97868)
@click.option("--tmax", type=float, default=5.)
@click.option("--truncate", type=float, default=2.5)
@click.option("--debug", is_flag=True)
@click.option("--label", type=str, default=None)
@click.option("--target-induction", type=int, multiple=True, default=[2])
@click.option("--exported-data-key", type=str, default=None)
@click.option("--font-size", type=float, default=12.)
def main(data_file_path, model_file_path, vmax, tmax, truncate, debug, label, target_induction, exported_data_key,
         font_size):
    """

    :param data_file_path: str (path)
    :param model_file_path: str (path)
    :param vmax: float
    :param tmax: float
    :param truncate: float
    :param debug: bool
    :param label: str
    :param target_induction: tuple of int
    :param exported_data_key: str
    :param font_size: float
    """
    mpl.rcParams['font.size'] = font_size

    if label is None:
        label = ''
    if not os.path.isfile(model_file_path):
        raise IOError('plot_biBTSP_data_summary_figure: invalid model_file_path: %s' % model_file_path)

    reference_delta_t = np.linspace(-1000. * tmax, 1000. * tmax, 100)

    peak_ramp_amp, total_induction_dur, group_indexes, exp_ramp, delta_exp_ramp, exp_ramp_raw, delta_exp_ramp_raw, \
    mean_induction_loc, interp_exp_ramp, interp_delta_exp_ramp, interp_delta_exp_ramp_raw, min_induction_t, \
    clean_min_induction_t, clean_induction_t_indexes, initial_induction_delta_vm = \
        get_biBTSP_data_analysis_results(data_file_path, reference_delta_t, debug=debug, truncate=truncate)

    delta_model_ramp, interp_delta_model_ramp = \
        get_biBTSP_model_analysis_results(model_file_path, reference_delta_t, exp_ramp, clean_min_induction_t,
                                          clean_induction_t_indexes, exported_data_key=exported_data_key, debug=debug)

    context.update(locals())

    flat_min_t = []
    flat_delta_model_ramp = []
    flat_initial_ramp = []
    flat_delta_exp_ramp = []
    target_flat_delta_model_ramp = []
    target_flat_initial_ramp = []
    target_flat_min_t = []

    for cell_key in delta_model_ramp:
        for induction in target_induction:
            induction_key = str(induction)
            if induction_key in delta_model_ramp[cell_key]:
                if debug:
                    print('Including cell: %s, induction: %s' % (cell_key, induction_key))
                this_clean_indexes = clean_induction_t_indexes[cell_key][induction_key]
                flat_min_t.extend(np.divide(clean_min_induction_t[cell_key][induction_key], 1000.))
                target_flat_min_t.extend(np.divide(clean_min_induction_t[cell_key][induction_key], 1000.))
                flat_delta_model_ramp.extend(delta_model_ramp[cell_key][induction_key][this_clean_indexes])
                target_flat_delta_model_ramp.extend(delta_model_ramp[cell_key][induction_key][this_clean_indexes])
                flat_initial_ramp.extend(exp_ramp[cell_key][induction_key]['before'][this_clean_indexes])
                target_flat_initial_ramp.extend(exp_ramp[cell_key][induction_key]['before'][this_clean_indexes])
                flat_delta_exp_ramp.extend(delta_exp_ramp[cell_key][induction_key][this_clean_indexes])
        if 2 in target_induction and 1 not in target_induction and '2' in delta_model_ramp[cell_key] and \
                '1' in delta_model_ramp[cell_key]:
            induction_key = '1'
            if debug:
                print('Including cell: %s, induction: %s' % (cell_key, induction_key))
            this_clean_indexes = clean_induction_t_indexes[cell_key][induction_key]
            flat_min_t.extend(np.divide(clean_min_induction_t[cell_key][induction_key], 1000.))
            flat_delta_model_ramp.extend(delta_model_ramp[cell_key][induction_key][this_clean_indexes])
            flat_initial_ramp.extend(exp_ramp[cell_key][induction_key]['before'][this_clean_indexes])
            flat_delta_exp_ramp.extend(delta_exp_ramp[cell_key][induction_key][this_clean_indexes])

    ymax = np.max(target_flat_initial_ramp)
    ymin = min(0., np.min(target_flat_initial_ramp))

    lines_cmap = 'jet'
    interp_cmap = 'bwr'

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))  #, constrained_layout=True)
    # axes = [axesgrid[2][0], axesgrid[2][1], axesgrid[2][2]]

    axes[0].set_xlim(-tmax, tmax)
    for cell_key in interp_delta_model_ramp:
        for induction in target_induction:
            induction_key = str(induction)
            if induction_key in interp_delta_model_ramp[cell_key]:
                lc = colorline(np.divide(reference_delta_t, 1000.),
                               interp_delta_model_ramp[cell_key][induction_key],
                               interp_exp_ramp[cell_key][induction_key]['before'],
                               vmin=ymin, vmax=ymax, cmap=lines_cmap)
                cax = axes[0].add_collection(lc)
    cbar = plt.colorbar(cax, ax=axes[0])
    cbar.set_label('Initial ramp\namplitude (mV)', rotation=270., labelpad=23.)
    axes[0].set_ylabel('Change in ramp\namplitude (mV)')
    axes[0].set_xlabel('Time relative to plateau onset (s)')
    axes[0].set_yticks(np.arange(-10., 16., 5.))
    axes[0].set_xticks(np.arange(-4., 5., 2.))

    if np.any(np.array(target_induction) != 1):
        points = np.array([flat_min_t, flat_initial_ramp]).transpose()
        data = np.array(flat_delta_model_ramp)

        this_vmax = max(abs(np.max(flat_delta_model_ramp)), abs(np.min(flat_delta_model_ramp)))
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
            cax = axes[1].pcolor(t_grid, initial_ramp_grid, interp_data, cmap=interp_cmap, vmin=-vmax, vmax=vmax,
                                 zorder=0)
            axes[1].set_ylabel('Initial ramp\namplitude (mV)')
            axes[1].set_xlabel('Time relative to plateau onset (s)')
            axes[1].set_ylim(0., ymax)
            axes[1].set_xlim(-tmax, tmax)
            axes[1].set_xticks(np.arange(-4., 5., 2.))
            cbar = plt.colorbar(cax, ax=axes[1])
            cbar.set_label('Change in ramp\namplitude (mV)', rotation=270., labelpad=23.)

    context.update(locals())

    this_axis = axes[2]
    this_axis.scatter(flat_delta_exp_ramp, flat_delta_model_ramp, c='k', linewidth=0, alpha=0.25, s=10)
    this_axis.set_xlabel('Actual (mV)')
    this_axis.set_ylabel('Predicted (mV)')
    this_axis.set_yticks(np.arange(-10., 16., 5.))
    this_axis.set_ylim([-10., 16.])
    this_axis.set_xticks(np.arange(-10., 16., 5.))
    this_axis.set_xlim([-10., 16.])
    this_xlim = this_axis.get_xlim()
    this_axis.plot([this_xlim[0], this_xlim[1]], [this_xlim[0], this_xlim[1]], '--', c='darkgrey', alpha=0.75)
    this_axis.set_title('Change in\nramp amplitude', fontsize=mpl.rcParams['font.size'], y=1.1)
    cbar = plt.colorbar(cax, ax=this_axis)
    cbar.ax.set_visible(False)

    r_val, p_val = pearsonr(flat_delta_model_ramp, flat_delta_exp_ramp)
    this_axis.annotate('R$^{2}$ = %.3f; p %s %.3f' %
                       (r_val ** 2., '>' if p_val > 0.05 else '<', p_val if p_val > 0.001 else 0.001),
                       xy=(0.1, 0.9), xycoords='axes fraction', color='k')

    clean_axes(axes)
    fig.suptitle(label, y=0.95, x=0.05, ha='left', fontsize=mpl.rcParams['font.size'])  # y=0.95,
    # fig.set_constrained_layout_pads(wspace=0.08, hspace=0.12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.75, hspace=0.2, wspace=0.6)
    fig.show()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)