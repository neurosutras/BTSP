__author__ = 'Aaron D. Milstein'
from biBTSP_utils import *
import click
from nested.parallel import *
from nested.optimize_utils import *

"""
Magee lab CA1 BTSP2 place field data should have already been exported to .hdf5 file with a standard format.
This method appends the results of least square approximation to estimate the initial weights for each place field in
the dataset. During optimization, these weights may be further modified to accomodate a change in the value of the peak
weight parameters.

data:
    cell_id:
        induction:
            processed:
                LSA_ramp:
                    before (if available)
                    after
                LSA_weights:
                    before (zeros by default)
                    after
"""

context = Context()


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--min-delta-weight", type=float, default=-0.2)
@click.option("--peak-delta-weight", type=float, default=3.)
@click.option("--input-field-width", type=float, default=90.)
@click.option("--calibration-ramp-width", type=float, default=108.)
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--data-file-name", type=str, default='20190710_biBTSP_data_calibrated_input.hdf5')
@click.option("--export-file-name", type=str, default=None)
@click.option("--export", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--plot", type=int, default=1)
@click.option("--interactive", is_flag=True)
@click.option("--verbose", type=int, default=1)
@click.pass_context
def main(cli, min_delta_weight, peak_delta_weight, input_field_width, calibration_ramp_width, data_dir, data_file_name,
         export_file_name, export, debug, plot, interactive, verbose):
    """
    :param cli: contains unrecognized args as list of str
    :param min_delta_weight: float
    :param peak_delta_weight: float
    :param input_field_width: float
    :param calibration_ramp_width: float
    :param data_dir: str (path)
    :param data_file_name: str (path)
    :param export_file_name: str (path)
    :param export: bool
    :param debug: bool
    :param plot: bool
    :param interactive: bool
    :param verbose: int
    """
    context.update(locals())
    kwargs = get_unknown_click_arg_dict(cli.args)
    context.disp = verbose > 0

    context.interface = get_parallel_interface(source_file=__file__, source_package=__package__, **kwargs)
    context.interface.start(disp=context.disp)
    context.interface.ensure_controller()
    config_parallel_interface(__file__, output_dir=data_dir, export=export, disp=context.disp,
                              interface=context.interface, min_delta_weight=min_delta_weight,
                              peak_delta_weight=peak_delta_weight, input_field_width=input_field_width,
                              calibration_ramp_width=calibration_ramp_width, data_dir=data_dir,
                              data_file_name=data_file_name, debug=debug, verbose=verbose, **kwargs)

    ramp_scaling_factor = context.interface.execute(compute_ramp_scaling_factor, plot=plot, verbose=verbose)
    args = context.interface.execute(get_cell_ids)
    group_size = len(args[0])
    sequences = args + [[ramp_scaling_factor] * group_size] + [[export] * group_size] + [[plot] * group_size]
    context.interface.map(compute_LSA_weights, *sequences)

    if export:
        if export_file_name is not None:
            export_file_path = data_dir + '/' + export_file_name
        else:
            export_file_path = data_dir + '/' + data_file_name
        collect_and_merge_temp_output(context.interface, export_file_path, verbose=context.disp)

    if plot > 0:
        context.interface.apply(plt.show)

    if not interactive:
        context.interface.stop()


def config_worker():
    """

    """
    context.data_file_path = context.data_dir + '/' + context.data_file_name
    if not os.path.isfile(context.data_file_path):
        raise IOError('init_context: invalid data_file_path: %s' % context.data_file_path)

    context.target_range = {'ramp_offset': 0.01, 'delta_min_val': 0.01, 'delta_peak_val': 0.01, 'residuals': 0.1,
                            'weights_smoothness': 0.005}
    with h5py.File(context.data_file_path, 'r') as f:
        dt = f['defaults'].attrs['dt']  # ms
        input_field_peak_rate = f['defaults'].attrs['input_field_peak_rate']  # Hz
        num_inputs = f['defaults'].attrs['num_inputs']
        track_length = f['defaults'].attrs['track_length']  # cm
        binned_dx = f['defaults'].attrs['binned_dx']  # cm
        generic_dx = f['defaults'].attrs['generic_dx']  # cm
        default_run_vel = f['defaults'].attrs['default_run_vel']  # cm/s
        generic_position_dt = f['defaults'].attrs['generic_position_dt']  # ms
        default_interp_dx = f['defaults'].attrs['default_interp_dx']  # cm
        binned_x = f['defaults']['binned_x'][:]
        generic_x = f['defaults']['generic_x'][:]
        generic_t = f['defaults']['generic_t'][:]
        default_interp_t = f['defaults']['default_interp_t'][:]
        default_interp_x = f['defaults']['default_interp_x'][:]
        extended_x = f['defaults']['extended_x'][:]
        context.input_field_width = float(context.input_field_width)
        input_rate_maps, peak_locs = \
            generate_spatial_rate_maps(binned_x, num_inputs, input_field_peak_rate, context.input_field_width,
                                       track_length)
        context.calibration_ramp_width = float(context.calibration_ramp_width)

        if 'data_keys' not in context() or context.data_keys is None:
            if 'cell_id' not in context() or context.cell_id == 'all' or context.cell_id is None:
                context.data_keys = \
                    [(int(cell_id), int(induction)) for cell_id in f['data'] for induction in ['1', '2']
                     if induction in f['data'][cell_id]]
            else:
                context.data_keys = \
                    [(int(context.cell_id), int(induction)) for induction in ['1', '2']
                     if induction in f['data'][str(context.cell_id)]]
        else:
            context.data_keys = [(int(cell_id), int(induction))
                                 for cell_id in [cell_id for cell_id in context.data_keys if str(cell_id) in f['data']]
                                 for induction in ['1', '2'] if induction in f['data'][str(cell_id)]]
        spont_cell_id_list = [int(cell_id) for cell_id in f['data'] if f['data'][cell_id].attrs['spont']]
        allow_offset_cell_ids = [int(cell_id) for cell_id in f['data'] if '2' in f['data'][cell_id] and
                                 ('1' not in f['data'][cell_id] or
                                  'before' not in f['data'][cell_id]['1']['raw']['exp_ramp'])]
    if context.verbose > 1:
        print('pid: %i; get_LSA_weights_biBTSP: processing the following data_keys: %s' %
              (os.getpid(), str(context.data_keys)))
        sys.stdout.flush()
    self_consistent_cell_ids = [cell_id for (cell_id, induction) in context.data_keys if induction == 1 and
                                (cell_id, 2) in context.data_keys]
    context.update(locals())
    context.cell_id = None
    context.induction = None


def get_cell_ids():
    """
    Return the data_keys to be mapped.
    :return: list of list
    """
    if context.debug:
        return [[list(set([data_key[0] for data_key in context.data_keys]))[0]]]
    return [list(set([data_key[0] for data_key in context.data_keys]))]


def compute_ramp_scaling_factor(plot=False, verbose=0):
    """

    :param plot: bool
    :param verbose: int
    :return: float
    """
    ramp_scaling_factor = \
        calibrate_ramp_scaling_factor(ramp_x=context.binned_x, input_x=context.binned_x,
                                      interp_x=context.default_interp_x,
                                      input_rate_maps=context.input_rate_maps, peak_locs=context.peak_locs,
                                      track_length=context.track_length, target_range=context.target_range,
                                      bounds=(0., context.peak_delta_weight),
                                      calibration_ramp_width=context.calibration_ramp_width, plot=plot > 0,
                                      verbose=verbose)
    if context.verbose > 1:
        print('get_LSA_weights_biBTSP: ramp_scaling_factor recomputed for %.1f cm field width: %.4E' %
              (context.input_field_width, ramp_scaling_factor))
        sys.stdout.flush()
    return ramp_scaling_factor


def import_data(cell_id, induction):
    """

    :param cell_id: int
    :param induction: int
    """
    cell_id = int(cell_id)
    induction = int(induction)
    if cell_id == context.cell_id and induction == context.induction:
        return
    cell_key = str(cell_id)
    induction_key = str(induction)
    with h5py.File(context.data_file_path, 'r') as f:
        if cell_key not in f['data'] or induction_key not in f['data'][cell_key]:
            raise KeyError('get_LSA_weights_biBTSP: no data found for cell_id: %s, induction: %s' %
                           (cell_key, induction_key))
        this_group = f['data'][cell_key][induction_key]
        context.induction_locs = this_group.attrs['induction_locs']
        context.induction_durs = this_group.attrs['induction_durs']
        context.exp_ramp_raw = {'after': this_group['raw']['exp_ramp']['after'][:]}
        if 'before' in this_group['raw']['exp_ramp']:
            context.exp_ramp_raw['before'] = this_group['raw']['exp_ramp']['before'][:]
        context.position = {}
        context.t = {}
        for category in this_group['processed']['position']:
            context.position[category] = []
            context.t[category] = []
            for i in range(len(this_group['processed']['position'][category])):
                lap_key = str(i)
                context.position[category].append(this_group['processed']['position'][category][lap_key][:])
                context.t[category].append(this_group['processed']['t'][category][lap_key][:])
        context.mean_position = this_group['processed']['mean_position'][:]
        context.mean_t = this_group['processed']['mean_t'][:]
        context.exp_ramp = {'after': this_group['processed']['exp_ramp']['after'][:]}
        if 'before' in this_group['processed']['exp_ramp']:
            context.exp_ramp['before'] = this_group['processed']['exp_ramp']['before'][:]

    context.mean_induction_start_loc = np.mean(context.induction_locs)
    context.mean_induction_dur = np.mean(context.induction_durs)
    mean_induction_start_index = np.where(context.mean_position >= context.mean_induction_start_loc)[0][0]
    mean_induction_stop_index = np.where(context.mean_t >= context.mean_t[mean_induction_start_index] +
                                         context.mean_induction_dur)[0][0]
    context.mean_induction_stop_loc = context.mean_position[mean_induction_stop_index]

    context.cell_id = cell_id
    context.induction = induction
    if context.verbose > 0:
        print('get_LSA_weights_biBTSP: process: %i loaded data for cell: %i, induction: %i' %
              (os.getpid(), cell_id, induction))
        sys.stdout.flush()


def compute_LSA_weights(cell_id, ramp_scaling_factor, export=False, plot=False):
    """

    :param cell_id: int
    :param ramp_scaling_factor
    :param export: bool
    :param plot: bool
    """
    context.ramp_scaling_factor = float(ramp_scaling_factor)
    cell_id = int(cell_id)
    induction_list = [data_key[1] for data_key in context.data_keys if data_key[0] == cell_id]
    prev_allow_offset = False
    prev_cell_id = None

    for induction in induction_list:
        import_data(cell_id, induction)
        LSA_ramp, delta_weights, ramp_offset = {}, {}, {}
        if induction == 1:
            if 'before' in context.exp_ramp:
                LSA_ramp['before'], delta_weights['before'], ramp_offset['before'], discard_residual_score = \
                    get_delta_weights_LSA(context.exp_ramp['before'], ramp_x=context.binned_x, input_x=context.binned_x,
                                          interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                          peak_locs=context.peak_locs, ramp_scaling_factor=context.ramp_scaling_factor,
                                          induction_start_loc=context.mean_induction_start_loc,
                                          induction_stop_loc=context.mean_induction_stop_loc,
                                          track_length=context.track_length, target_range=context.target_range,
                                          bounds=(context.min_delta_weight, context.peak_delta_weight), beta=2.,
                                          plot=plot > 1, label='Cell: %i; Induction: %i' % (cell_id, induction),
                                          verbose=context.verbose)
                if context.verbose > 0:
                    print('Process: %i; cell: %i: before induction: %i; ramp_offset: %.3f' %
                          (os.getpid(), context.cell_id, context.induction, ramp_offset['before']))
                prev_allow_offset = False
            else:
                delta_weights['before'] = np.zeros_like(context.peak_locs)
                ramp_offset['before'] = 0.
                prev_allow_offset = True
            LSA_ramp['after'], delta_weights['after'], ramp_offset['after'], discard_residual_score = \
                get_delta_weights_LSA(context.exp_ramp['after'], ramp_x=context.binned_x, input_x=context.binned_x,
                                      interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                      peak_locs=context.peak_locs, ramp_scaling_factor=context.ramp_scaling_factor,
                                      induction_start_loc=context.mean_induction_start_loc,
                                      induction_stop_loc=context.mean_induction_stop_loc,
                                      track_length=context.track_length, target_range=context.target_range,
                                      bounds=(context.min_delta_weight, context.peak_delta_weight), beta=2.,
                                      allow_offset=prev_allow_offset, plot=plot > 1,
                                      label='Cell: %i; Induction: %i' % (cell_id, induction), verbose=context.verbose)

            if context.verbose > 0:
                print('Process: %i; cell: %i: after induction: %i; ramp_offset: %.3f' %
                      (os.getpid(), context.cell_id, context.induction, ramp_offset['after']))
        else:
            if prev_cell_id is None or prev_cell_id != cell_id:
                LSA_ramp['before'], delta_weights['before'], ramp_offset['before'], discard_residual_score = \
                    get_delta_weights_LSA(context.exp_ramp['before'], ramp_x=context.binned_x, input_x=context.binned_x,
                                          interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                          peak_locs=context.peak_locs, ramp_scaling_factor=context.ramp_scaling_factor,
                                          induction_start_loc=context.mean_induction_start_loc,
                                          induction_stop_loc=context.mean_induction_stop_loc,
                                          track_length=context.track_length, target_range=context.target_range,
                                          bounds=(context.min_delta_weight, context.peak_delta_weight), beta=2.,
                                          allow_offset=True, plot=plot > 1,
                                          label='Cell: %i; Induction: %i' % (cell_id, induction),
                                          verbose=context.verbose)
            else:
                LSA_ramp['before'], delta_weights['before'], ramp_offset['before'], discard_residual_score = \
                    get_delta_weights_LSA(context.exp_ramp['before'], ramp_x=context.binned_x, input_x=context.binned_x,
                                          interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                          peak_locs=context.peak_locs, ramp_scaling_factor=context.ramp_scaling_factor,
                                          induction_start_loc=context.mean_induction_start_loc,
                                          induction_stop_loc=context.mean_induction_stop_loc,
                                          track_length=context.track_length, target_range=context.target_range,
                                          bounds=(context.min_delta_weight, context.peak_delta_weight), beta=2.,
                                          allow_offset=prev_allow_offset, plot=plot > 1,
                                          label='Cell: %i; Induction: %i' % (cell_id, induction),
                                          verbose=context.verbose)
            if context.verbose > 0:
                print('Process: %i; cell: %i: before induction: %i; ramp_offset: %.3f' %
                      (os.getpid(), context.cell_id, context.induction, ramp_offset['before']))
            LSA_ramp['after'], delta_weights['after'], ramp_offset['after'], discard_residual_score = \
                get_delta_weights_LSA(context.exp_ramp['after'], ramp_x=context.binned_x, input_x=context.binned_x,
                                      interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                      peak_locs=context.peak_locs, ramp_scaling_factor=context.ramp_scaling_factor,
                                      induction_start_loc=context.mean_induction_start_loc,
                                      induction_stop_loc=context.mean_induction_stop_loc,
                                      track_length=context.track_length, target_range=context.target_range,
                                      bounds=(context.min_delta_weight, context.peak_delta_weight), beta=2.,
                                      allow_offset=False, impose_offset=ramp_offset['before'], plot=plot > 1,
                                      label='Cell: %i; Induction: %i' % (cell_id, induction),
                                      verbose=context.verbose)
            if context.verbose > 0:
                print('Process: %i; cell: %i: after induction: %i; ramp_offset: %.3f' %
                      (os.getpid(), context.cell_id, context.induction, ramp_offset['after']))
            prev_allow_offset = False
        prev_cell_id = cell_id

        if plot > 0:
            fig, axes = plt.subplots(1, 2)
            x_start = context.mean_induction_start_loc
            x_end = context.mean_induction_stop_loc
            max_ramp = max(np.max(LSA_ramp['after']), np.max(context.exp_ramp['after']),
                           np.max(context.exp_ramp_raw['after']))
            max_weights = np.max(delta_weights['after'])
            axes[0].plot(context.binned_x, LSA_ramp['after'])
            axes[1].plot(context.peak_locs, delta_weights['after'] + 1., label='After induction')
            if 'before' in context.exp_ramp:
                axes[0].plot(context.binned_x, LSA_ramp['before'])
                axes[1].plot(context.peak_locs, delta_weights['before'] + 1., label='Before induction')
                max_weights = max(max_weights, np.max(delta_weights['before']))
            max_weights += 1
            axes[0].plot(context.binned_x, context.exp_ramp['after'])
            axes[0].plot(context.binned_x, context.exp_ramp_raw['after'])
            if 'before' in context.exp_ramp:
                axes[0].plot(context.binned_x, context.exp_ramp['before'])
                axes[0].plot(context.binned_x, context.exp_ramp_raw['before'])
                max_ramp = max(max_ramp, np.max(LSA_ramp['before']), np.max(context.exp_ramp['before']),
                               np.max(context.exp_ramp_raw['before']))
            axes[0].hlines(max_ramp * 1.1, xmin=x_start, xmax=x_end, linewidth=2, colors='k')
            axes[1].hlines(max_weights * 1.1, xmin=x_start, xmax=x_end, linewidth=2, colors='k')
            axes[0].set_ylim(-1., max_ramp * 1.2)
            axes[1].set_ylim(0.5, max_weights * 1.2)
            axes[0].set_xlabel('Position (cm)')
            axes[1].set_xlabel('Position (cm)')
            axes[0].set_ylabel('Ramp amplitude (mV)')
            axes[1].set_ylabel('Candidate synaptic weights (a.u.)')
            axes[1].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
            axes[0].set_title('Cell: %i, Induction: %i' % (context.cell_id, context.induction))
            clean_axes(axes)
            fig.tight_layout()
            fig.show()
        sys.stdout.flush()
        context.update(locals())
        if export:
            export_data(context.temp_output_path)


def export_data(export_file_path=None):
    """

    :param export_file_path: str (path)
    """
    with h5py.File(export_file_path, 'a') as f:
        if 'calibrated_input' not in f:
            f.create_group('calibrated_input')
        input_field_width_key = str(int(context.input_field_width))
        if input_field_width_key not in f['calibrated_input']:
            f['calibrated_input'].create_group(input_field_width_key)
            f['calibrated_input'][input_field_width_key].attrs['input_field_width'] = context.input_field_width
            f['calibrated_input'][input_field_width_key].attrs['ramp_scaling_factor'] = context.ramp_scaling_factor
        cell_key = str(context.cell_id)
        induction_key = str(context.induction)
        if cell_key not in f['calibrated_input'][input_field_width_key]:
            f['calibrated_input'][input_field_width_key].create_group(cell_key)
        if induction_key in f['calibrated_input'][input_field_width_key][cell_key]:
            raise RuntimeError('get_LSA_weights_biBTSP: data for input_field_width: %.1f; cell: %s; induction: %s '
                               'has already been exported to the provided export_file_path: %s' %
                               (context.input_field_width, cell_key, induction_key, export_file_path))
        this_group = f['calibrated_input'][input_field_width_key][cell_key].create_group(induction_key)
        this_group.create_group('LSA_ramp')
        this_group['LSA_ramp'].create_dataset('after', compression='gzip', data=context.LSA_ramp['after'])
        this_group['LSA_ramp']['after'].attrs['ramp_offset'] = context.ramp_offset['after']
        if 'before' in context.LSA_ramp:
            this_group['LSA_ramp'].create_dataset('before', compression='gzip', data=context.LSA_ramp['before'])
            this_group['LSA_ramp']['before'].attrs['ramp_offset'] = context.ramp_offset['before']
        this_group.create_group('LSA_weights')
        this_group['LSA_weights'].create_dataset('before', compression='gzip', data=context.delta_weights['before'])
        this_group['LSA_weights'].create_dataset('after', compression='gzip', data=context.delta_weights['after'])
    if context.verbose > 0:
        print('Exported data for cell: %i, induction: %i to %s' %
              (context.cell_id, context.induction, export_file_path))
        sys.stdout.flush()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
