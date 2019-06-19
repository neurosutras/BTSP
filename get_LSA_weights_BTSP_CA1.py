__author__ = 'Aaron D. Milstein'
from BTSP_utils_alt import *
import click


"""
Magee lab CA1 BTSP2 place field data should have already been exported to .hdf5 file with a standard format.
This method appends the results of least square approximation to estimate the initial weights for each place field in
the dataset. These weights will be used to initialize another round of approximation given potentially changing
peak weight bounds during optimization.

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


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True,))
@click.option("--min-delta-weight", type=float, default=-0.2)
@click.option("--target-weights-width", type=float, default=108.)
@click.option("--input-field-width", type=float, default=90.)
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='data')
@click.option("--data-file-name", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='data/20180411_BTSP2_CA1_data.hdf5')
@click.option("--export-file-name", type=str, default=None)
@click.option("--plot", type=int, default=1)
@click.option("--verbose", type=int, default=1)
@click.option("--export", is_flag=True)
@click.pass_context
def main(cli, min_delta_weight, target_weights_width, input_field_width, data_dir, data_file_name, export_file_name,
         plot, verbose, export):
    """
    :param cli: contains unrecognized args as list of str
    :param min_delta_weight: float
    :param target_weights_width: float
    :param input_field_width: float
    :param data_dir: str (path)
    :param data_file_name: str (path)
    :param export_file_name: str (path)
    :param plot: bool
    :param verbose: int
    :param export: bool
    """
    context.update(locals())
    kwargs = get_unknown_click_arg_dict(cli.args)
    context.update(kwargs)
    init_context()
    prev_cell_id = None
    prev_allow_offset = False

    for cell_id, induction in context.data_keys:
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
                                          bounds=(context.min_delta_weight, 3.), beta=2., plot=plot > 1,
                                          label='Cell: %i; Induction: %i' % (cell_id, induction),
                                          verbose=verbose)
                if verbose > 0:
                    print 'Process: %i; cell: %i: before induction: %i; ramp_offset: %.3f' % \
                          (os.getpid(), context.cell_id, context.induction, ramp_offset['before'])
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
                                      bounds=(context.min_delta_weight, 3.), beta=2., allow_offset=prev_allow_offset,
                                      plot=plot > 1, label='Cell: %i; Induction: %i' % (cell_id, induction),
                                      verbose=verbose)

            if verbose > 0:
                print 'Process: %i; cell: %i: after induction: %i; ramp_offset: %.3f' % \
                      (os.getpid(), context.cell_id, context.induction, ramp_offset['after'])
        else:
            if prev_cell_id is None or prev_cell_id != cell_id:
                LSA_ramp['before'], delta_weights['before'], ramp_offset['before'], discard_residual_score = \
                    get_delta_weights_LSA(context.exp_ramp['before'], ramp_x=context.binned_x, input_x=context.binned_x,
                                          interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                          peak_locs=context.peak_locs, ramp_scaling_factor=context.ramp_scaling_factor,
                                          induction_start_loc=context.mean_induction_start_loc,
                                          induction_stop_loc=context.mean_induction_stop_loc,
                                          track_length=context.track_length, target_range=context.target_range,
                                          bounds=(context.min_delta_weight, 3.), beta=2.,
                                          allow_offset=True, plot=plot > 1,
                                          label='Cell: %i; Induction: %i' % (cell_id, induction), verbose=verbose)
            else:
                LSA_ramp['before'], delta_weights['before'], ramp_offset['before'], discard_residual_score = \
                    get_delta_weights_LSA(context.exp_ramp['before'], ramp_x=context.binned_x, input_x=context.binned_x,
                                          interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                          peak_locs=context.peak_locs, ramp_scaling_factor=context.ramp_scaling_factor,
                                          induction_start_loc=context.mean_induction_start_loc,
                                          induction_stop_loc=context.mean_induction_stop_loc,
                                          track_length=context.track_length, target_range=context.target_range,
                                          bounds=(context.min_delta_weight, 3.), beta=2.,
                                          allow_offset=prev_allow_offset, plot=plot > 1,
                                          label='Cell: %i; Induction: %i' % (cell_id, induction), verbose=verbose)
            if verbose > 0:
                print 'Process: %i; cell: %i: before induction: %i; ramp_offset: %.3f' % \
                      (os.getpid(), context.cell_id, context.induction, ramp_offset['before'])
            LSA_ramp['after'], delta_weights['after'], ramp_offset['after'], discard_residual_score = \
                get_delta_weights_LSA(context.exp_ramp['after'], ramp_x=context.binned_x, input_x=context.binned_x,
                                      interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                      peak_locs=context.peak_locs, ramp_scaling_factor=context.ramp_scaling_factor,
                                      induction_start_loc=context.mean_induction_start_loc,
                                      induction_stop_loc=context.mean_induction_stop_loc,
                                      track_length=context.track_length, target_range=context.target_range,
                                      bounds=(context.min_delta_weight, 3.), beta=2.,
                                      allow_offset=False, impose_offset=ramp_offset['before'], plot=plot > 1,
                                      label='Cell: %i; Induction: %i' % (cell_id, induction), verbose=verbose)
            if verbose > 0:
                print 'Process: %i; cell: %i: after induction: %i; ramp_offset: %.3f' % \
                      (os.getpid(), context.cell_id, context.induction, ramp_offset['after'])
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
        context.update(locals())
        if export:
            export_data()


def init_context():
    """

    """
    if context.data_file_path is None or not os.path.isfile(context.data_file_path):
        raise IOError('init_context: invalid data_file_path: %s' % context.data_file_path)
    context.target_range = {'ramp_offset': 0.01, 'delta_min_val': 0.01, 'delta_peak_val': 0.01, 'residuals': 0.1,
                            'weights_smoothness': 0.01}
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
        ramp_scaling_factor = \
            get_weights_LSA_scaling_factor(ramp_x=binned_x, input_x=binned_x, interp_x=default_interp_x,
                                           input_rate_maps=input_rate_maps, peak_locs=peak_locs,
                                           target_field_width=context.target_field_width, track_length=track_length,
                                           target_range=context.target_range,
                                           bounds=(context.min_delta_weight, 3.), plot=context.plot > 0,
                                           verbose=context.verbose)[3]
        if context.verbose > 1:
            print 'optimize_BTSP_D_CA1: ramp_scaling_factor recomputed for %.1f cm field width: %.4E' % \
                  (context.input_field_width, ramp_scaling_factor)
        if 'data_keys' not in context() or context.data_keys is None:
            if 'cell_id' not in context() or context.cell_id == 'all' or context.cell_id is None:
                    context.data_keys = \
                        [(int(cell_id), int(induction)) for cell_id in f['data'] for induction in f['data'][cell_id]]
            else:
                context.data_keys = \
                    [(int(context.cell_id), int(induction)) for induction in f['data'][str(context.cell_id)]]
        else:
            context.data_keys = [(int(cell_id), int(induction))
                                 for cell_id in [cell_id for cell_id in context.data_keys if str(cell_id) in f['data']]
                                 for induction in f['data'][str(cell_id)]]
        spont_cell_id_list = [int(cell_id) for cell_id in f['data'] if f['data'][cell_id].attrs['spont']]
        allow_offset_cell_ids = [int(cell_id) for cell_id in f['data'] if '2' in f['data'][cell_id] and
                                 ('1' not in f['data'][cell_id] or
                                  'before' not in f['data'][cell_id]['1']['raw']['exp_ramp'])]
    if context.verbose > 1:
        print 'pid: %i; get_LSA_weights_BTSP_CA1: processing the following data_keys: %s' % \
              (os.getpid(), str(context.data_keys))
    self_consistent_cell_ids = [cell_id for (cell_id, induction) in context.data_keys if induction == 1 and
                                 (cell_id, 2) in context.data_keys]
    context.update(locals())
    context.cell_id = None
    context.induction = None


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
            raise KeyError('parallel_optimize_bidirectional_BTSP_CA1: no data found for cell_id: %s, induction: %s' %
                           (cell_key, induction_key))
        else:
            context.cell_id = cell_id
            context.induction = induction
        this_group = f['data'][cell_key][induction_key]
        context.induction_locs = this_group.attrs['induction_locs']
        context.induction_durs = this_group.attrs['induction_durs']
        context.exp_ramp_raw = {'after': this_group['raw']['exp_ramp']['after'][:]}
        if 'before' in this_group['raw']['exp_ramp']:
            context.exp_ramp_raw['before'] = this_group['raw']['exp_ramp']['before'][:]
        context.position = {}
        context.t = {}
        context.current = []
        for category in this_group['processed']['position']:
            context.position[category] = []
            context.t[category] = []
            for i in xrange(len(this_group['processed']['position'][category])):
                lap_key = str(i)
                context.position[category].append(this_group['processed']['position'][category][lap_key][:])
                context.t[category].append(this_group['processed']['t'][category][lap_key][:])
        for i in xrange(len(this_group['processed']['current'])):
            lap_key = str(i)
            context.current.append(this_group['processed']['current'][lap_key][:])
        context.mean_position = this_group['processed']['mean_position'][:]
        context.mean_t = this_group['processed']['mean_t'][:]
        context.exp_ramp = {'after': this_group['processed']['exp_ramp']['after'][:]}
        context.exp_ramp_vs_t = {'after': this_group['processed']['exp_ramp_vs_t']['after'][:]}
        if 'before' in this_group['processed']['exp_ramp']:
            context.exp_ramp['before'] = this_group['processed']['exp_ramp']['before'][:]
            context.exp_ramp_vs_t['before'] = this_group['processed']['exp_ramp_vs_t']['before'][:]
        context.complete_run_vel = this_group['complete']['run_vel'][:]
        context.complete_run_vel_gate = this_group['complete']['run_vel_gate'][:]
        context.complete_position = this_group['complete']['position'][:]
        context.complete_t = this_group['complete']['t'][:]
        context.induction_gate = this_group['complete']['induction_gate'][:]
    context.mean_induction_start_loc = np.mean(context.induction_locs)
    context.mean_induction_dur = np.mean(context.induction_durs)
    mean_induction_start_index = np.where(context.mean_position >= context.mean_induction_start_loc)[0][0]
    mean_induction_stop_index = np.where(context.mean_t >= context.mean_t[mean_induction_start_index] +
                                         context.mean_induction_dur)[0][0]
    context.mean_induction_stop_loc = context.mean_position[mean_induction_stop_index]
    induction_start_times = []
    induction_stop_times = []
    track_start_times = []
    track_stop_times = []
    running_position = 0.
    running_t = 0.
    for i, (this_induction_loc, this_induction_dur) in enumerate(zip(context.induction_locs, context.induction_durs)):
        this_induction_start_index = np.where(context.complete_position >= this_induction_loc + running_position)[0][0]
        this_induction_start_time = context.complete_t[this_induction_start_index]
        this_induction_stop_time = this_induction_start_time + this_induction_dur
        track_start_times.append(running_t)
        running_t += len(context.t['induction'][i]) * context.dt
        track_stop_times.append(running_t)
        induction_start_times.append(this_induction_start_time)
        induction_stop_times.append(this_induction_stop_time)
        running_position += context.track_length
    context.induction_start_times = np.array(induction_start_times)
    context.induction_stop_times = np.array(induction_stop_times)
    context.track_start_times = np.array(track_start_times)
    context.track_stop_times = np.array(track_stop_times)
    if context.verbose > 0:
        print 'get_LSA_weights_BTSP2_CA1: process: %i loaded data for cell: %i, induction: %i' % \
              (os.getpid(), cell_id, induction)


def export_data(export_file_path=None):
    """

    :param export_file_path: str (path)
    """
    if export_file_path is None:
        export_file_path = context.data_file_path
    with h5py.File(export_file_path, 'a') as f:
        if 'data' not in f:
            f.create_group('data')
        cell_key = str(context.cell_id)
        induction_key = str(context.induction)
        if cell_key not in f['data']:
            f['data'].create_group(cell_key)
        if induction_key not in f['data'][cell_key]:
            f['data'][cell_key].create_group(induction_key)
        this_group = f['data'][cell_key][induction_key]
        if 'processed' not in this_group:
            this_group.create_group('processed')
        this_group['processed'].create_group('LSA_ramp')
        this_group['processed']['LSA_ramp'].create_dataset('after', compression='gzip', compression_opts=9,
                                              data=context.LSA_ramp['after'])
        this_group['processed']['LSA_ramp']['after'].attrs['ramp_offset'] = context.ramp_offset['after']
        if 'before' in context.LSA_ramp:
            this_group['processed']['LSA_ramp'].create_dataset('before', compression='gzip', compression_opts=9,
                                                  data=context.LSA_ramp['before'])
            this_group['processed']['LSA_ramp']['before'].attrs['ramp_offset'] = context.ramp_offset['before']
        this_group['processed'].create_group('LSA_weights')
        this_group['processed']['LSA_weights'].create_dataset('before', compression='gzip', compression_opts=9,
                                                 data=context.delta_weights['before'])
        this_group['processed']['LSA_weights'].create_dataset('after', compression='gzip', compression_opts=9,
                                                 data=context.delta_weights['after'])
    if context.verbose > 0:
        print 'Exported data for cell: %i, induction: %i to %s' % (context.cell_id, context.induction, export_file_path)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1,sys.argv)+1):])
