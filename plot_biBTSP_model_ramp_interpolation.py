from biBTSP_utils import *
from nested.optimize_utils import *
import matplotlib as mpl
import click
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from collections import defaultdict

mpl.rcParams['svg.fonttype'] = 'none'
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
@click.option("--vmax", type=float, default=12.97868)
@click.option("--tmax", type=float, default=5.)
@click.option("--truncate", type=bool, default=True)
@click.option("--debug", is_flag=True)
@click.option("--label", type=str, default=None)
@click.option("--target-induction", type=int, multiple=True, default=[2])
@click.option("--font-size", type=float, default=11.)
def main(data_file_path, model_file_path, vmax, tmax, truncate, debug, label, target_induction, font_size):
    """

    :param data_file_path: str (path)
    :param model_file_path: str (path)
    :param vmax: float
    :param tmax: float
    :param truncate: bool
    :param debug: bool
    :param label: str
    :param target_induction: tuple of int
    :param font_size: float
    """
    if label is None:
        label = ''
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
    reference_delta_t = np.linspace(-1000. * tmax, 1000. * tmax, 100)

    mpl.rcParams['font.size'] = font_size

    peak_ramp_amp, total_induction_dur, depo_soma, group_indexes, exp_ramp, extended_exp_ramp, delta_exp_ramp, \
    mean_induction_loc, extended_min_delta_t, extended_delta_exp_ramp, interp_initial_exp_ramp, \
    interp_delta_exp_ramp, interp_final_exp_ramp = \
        get_biBTSP_analysis_results(data_file_path, binned_x, binned_extra_x, extended_binned_x, reference_delta_t,
                                    track_length, dt, debug, truncate=truncate)

    context.update(locals())

    model_ramp = defaultdict(lambda: defaultdict(dict))
    extended_model_ramp = defaultdict(lambda: defaultdict(dict))
    delta_model_ramp = defaultdict(dict)
    extended_delta_model_ramp = defaultdict(dict)
    interp_delta_model_ramp = defaultdict(dict)
    interp_final_model_ramp = defaultdict(dict)

    with h5py.File(model_file_path, 'r') as f:
        for cell_key in f['exported_data']:
            for induction_key in f['exported_data'][cell_key]:
                model_ramp[cell_key][induction_key]['after'] = \
                    f['exported_data'][cell_key][induction_key]['model_ramp_features']['model_ramp'][:]
                delta_model_ramp[cell_key][induction_key] = np.subtract(model_ramp[cell_key][induction_key]['after'],
                                                                        exp_ramp[cell_key][induction_key]['before'])
                extended_delta_model_ramp[cell_key][induction_key] = \
                    np.concatenate([delta_model_ramp[cell_key][induction_key]] * 3)
                for category in model_ramp[cell_key][induction_key]:
                    extended_model_ramp[cell_key][induction_key][category] = \
                        np.concatenate([model_ramp[cell_key][induction_key][category]] * 3)

                mask = ~np.isnan(extended_min_delta_t[cell_key][induction_key])
                bad_indexes = np.where(np.isnan(interp_delta_exp_ramp[cell_key][induction_key]))[0]

                interp_delta_model_ramp[cell_key][induction_key] = \
                    np.interp(reference_delta_t, extended_min_delta_t[cell_key][induction_key][mask],
                              extended_delta_model_ramp[cell_key][induction_key][mask])

                interp_final_model_ramp[cell_key][induction_key] = \
                    np.interp(reference_delta_t, extended_min_delta_t[cell_key][induction_key][mask],
                              extended_model_ramp[cell_key][induction_key]['after'][mask])

                if len(bad_indexes) > 0:
                    interp_delta_model_ramp[cell_key][induction_key][bad_indexes] = np.nan
                    interp_final_model_ramp[cell_key][induction_key][bad_indexes] = np.nan

    flat_min_t = []
    flat_delta_ramp = []
    flat_initial_ramp = []
    flat_final_ramp = []
    target_flat_delta_ramp = []
    target_flat_initial_ramp = []

    for cell_key in interp_delta_model_ramp:
        for induction in target_induction:
            induction_key = str(induction)
            if induction_key in interp_delta_model_ramp[cell_key]:
                indexes = np.where(~np.isnan(interp_delta_model_ramp[cell_key][induction_key]))[0]
                flat_min_t.extend(np.divide(reference_delta_t[indexes], 1000.))
                flat_delta_ramp.extend(interp_delta_model_ramp[cell_key][induction_key][indexes])
                target_flat_delta_ramp.extend(interp_delta_model_ramp[cell_key][induction_key][indexes])
                flat_initial_ramp.extend(interp_initial_exp_ramp[cell_key][induction_key][indexes])
                target_flat_initial_ramp.extend(interp_initial_exp_ramp[cell_key][induction_key][indexes])
                flat_final_ramp.extend(interp_final_model_ramp[cell_key][induction_key][indexes])
        if 2 in target_induction and 1 not in target_induction and '2' in interp_delta_exp_ramp[cell_key] and \
                '1' in interp_delta_model_ramp[cell_key]:
            induction_key = '1'
            indexes = np.where(~np.isnan(interp_delta_model_ramp[cell_key][induction_key]))[0]
            flat_min_t.extend(np.divide(reference_delta_t[indexes], 1000.))
            flat_delta_ramp.extend(interp_delta_model_ramp[cell_key][induction_key][indexes])
            flat_initial_ramp.extend(interp_initial_exp_ramp[cell_key][induction_key][indexes])
            flat_final_ramp.extend(interp_final_model_ramp[cell_key][induction_key][indexes])

    ymax = np.max(target_flat_initial_ramp)
    ymin = min(0., np.min(target_flat_initial_ramp))

    lines_cmap = 'jet'
    interp_cmap = 'bwr_r'

    context.update(locals())

    fig, axesgrid = plt.subplots(2, 2, figsize=(8.25, 6))
    axes = []
    for row in range(2):
        for col in range(2):
            axes.append(axesgrid[col][row])

    axes[1].set_xlim(-tmax, tmax)
    for cell_key in interp_delta_model_ramp:
        for induction in target_induction:
            induction_key = str(induction)
            if induction_key in interp_delta_model_ramp[cell_key]:
                indexes = np.where(~np.isnan(interp_delta_model_ramp[cell_key][induction_key]))[0]
                lc = colorline(np.divide(reference_delta_t[indexes], 1000.),
                               interp_delta_model_ramp[cell_key][induction_key][indexes],
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
    fig.suptitle(label, x=0.05, y=0.95, ha='left', fontsize=mpl.rcParams['font.size'])
    fig.show()

    context.update(locals())


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)