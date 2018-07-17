__author__ = 'Aaron D. Milstein'
from BTSP_utils import *
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
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='data')
@click.option("--plot", type=int, default=1)
@click.option("--verbose", type=int, default=1)
@click.option("--export", is_flag=True)
@click.option("--data-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='data/20180411_BTSP2_CA1_data.hdf5')
@click.pass_context
def main(cli, min_delta_weight, data_dir, plot, verbose, export, data_file_path):
    """
    :param cli: contains unrecognized args as list of str
    :param min_delta_weight: float
    :param data_dir: str
    :param plot: bool
    :param verbose: int
    :param export: bool
    :param data_file_path: str (path)
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
                LSA_ramp['before'], delta_weights['before'], ramp_offset['before'] = \
                    get_delta_weights_LSA(context.exp_ramp['before'], context.input_rate_maps,
                                          bounds=(context.min_delta_weight, 3.), beta=2., plot=plot > 1,
                                          verbose=verbose)
                if verbose > 0:
                    print 'Process: %i; cell: %i: before induction: %i; ramp_offset: %.3f' % \
                          (os.getpid(), context.cell_id, context.induction, ramp_offset['before'])
                prev_allow_offset = False
            else:
                delta_weights['before'] = np.zeros_like(context.peak_locs)
                ramp_offset['before'] = 0.
                prev_allow_offset = True
            LSA_ramp['after'], delta_weights['after'], ramp_offset['after'] = \
                get_delta_weights_LSA(context.exp_ramp['after'], context.input_rate_maps,
                                      bounds=(context.min_delta_weight, 3.), beta=2., allow_offset=prev_allow_offset,
                                      plot=plot > 1, verbose=verbose)
            if verbose > 0:
                print 'Process: %i; cell: %i: after induction: %i; ramp_offset: %.3f' % \
                      (os.getpid(), context.cell_id, context.induction, ramp_offset['after'])
        else:
            if prev_cell_id is None or prev_cell_id != cell_id:
                LSA_ramp['before'], delta_weights['before'], ramp_offset['before'] = \
                    get_delta_weights_LSA(context.exp_ramp['before'], context.input_rate_maps,
                                          bounds=(context.min_delta_weight, 3.), beta=2., plot=plot > 1,
                                          allow_offset=True, verbose=verbose)
            else:
                LSA_ramp['before'], delta_weights['before'], ramp_offset['before'] = \
                    get_delta_weights_LSA(context.exp_ramp['before'], context.input_rate_maps,
                                          bounds=(context.min_delta_weight, 3.), beta=2., plot=plot > 1,
                                          allow_offset=prev_allow_offset, verbose=verbose)
            if verbose > 0:
                print 'Process: %i; cell: %i: before induction: %i; ramp_offset: %.3f' % \
                      (os.getpid(), context.cell_id, context.induction, ramp_offset['before'])
            LSA_ramp['after'], delta_weights['after'], ramp_offset['after'] = \
                get_delta_weights_LSA(context.exp_ramp['after'], context.input_rate_maps,
                                      bounds=(context.min_delta_weight, 3.), beta=2., allow_offset=False,
                                      impose_offset=ramp_offset['before'], plot=plot > 1, verbose=verbose)
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
            fig.suptitle('Cell: %i, Induction: %i' % (context.cell_id, context.induction))
            clean_axes(axes)
            fig.tight_layout()
            plt.show()
            plt.close()
        context.update(locals())
        if export:
            export_data()


def init_context():
    """

    """
    if context.data_file_path is None or not os.path.isfile(context.data_file_path):
        raise IOError('init_context: invalid data_file_path: %s' % context.data_file_path)
    with h5py.File(context.data_file_path, 'r') as f:
        dt = f['defaults'].attrs['dt']  # ms
        input_field_width = f['defaults'].attrs['input_field_width']  # cm
        input_field_peak_rate = f['defaults'].attrs['input_field_peak_rate']  # Hz
        num_inputs = f['defaults'].attrs['num_inputs']
        track_length = f['defaults'].attrs['track_length']  # cm
        binned_dx = f['defaults'].attrs['binned_dx']  # cm
        generic_dx = f['defaults'].attrs['generic_dx']  # cm
        default_run_vel = f['defaults'].attrs['default_run_vel']  # cm/s
        generic_position_dt = f['defaults'].attrs['generic_position_dt']  # ms
        default_interp_dx = f['defaults'].attrs['default_interp_dx']  # cm
        ramp_scaling_factor = f['defaults'].attrs['ramp_scaling_factor']
        binned_x = f['defaults']['binned_x'][:]
        generic_x = f['defaults']['generic_x'][:]
        generic_t = f['defaults']['generic_t'][:]
        default_interp_t = f['defaults']['default_interp_t'][:]
        default_interp_x = f['defaults']['default_interp_x'][:]
        extended_x = f['defaults']['extended_x'][:]
        input_rate_maps = f['defaults']['input_rate_maps'][:]
        peak_locs = f['defaults']['peak_locs'][:]
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
    context.update(locals())
    context.cell_id = None
    context.induction = None
    context.target_range = {'ramp_offset': 0.01, 'delta_min_val': 0.01, 'delta_peak_val': 0.01, 'residuals': 0.1,
                            'weights_smoothness': 0.01}


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


def get_residual_score(delta_weights, target_ramp, input_matrix, ramp_x, input_x, bounds=None,
                                allow_offset=False, impose_offset=None, disp=False, full_output=False):
    """

    :param delta_weights: array
    :param target_ramp: array
    :param input_matrix: array
    :param ramp_x: array
    :param input_x: array
    :param bounds: array
    :param allow_offset: bool (allow special case where baseline Vm before 1st induction is unknown)
    :param impose_offset: float (impose Vm offset from 1st induction on 2nd induction)
    :param disp: bool
    :param full_output: bool
    :return: float
    """
    if bounds is not None:
        min_weight, max_weight = bounds
        if np.min(delta_weights) < min_weight or np.max(delta_weights) > max_weight:
            if full_output:
                raise Exception('get_residual_score: input out of bounds; cannot return full_output')
            return 1e9
    model_ramp = delta_weights.dot(input_matrix)
    if len(model_ramp) != len(ramp_x):
        model_ramp = np.interp(ramp_x, input_x, model_ramp)

    Err = 0.
    if impose_offset is not None:
        ramp_offset = impose_offset
        model_ramp -= impose_offset
    elif allow_offset:
        model_ramp, ramp_offset = subtract_baseline(model_ramp)
        Err += (ramp_offset / context.target_range['ramp_offset']) ** 2.
    else:
        ramp_offset = 0.

    ramp_amp, ramp_width, peak_shift, ratio, start_loc, peak_loc, end_loc, min_val, min_loc = {}, {}, {}, {}, {}, {}, \
                                                                                              {}, {}, {}
    ramp_amp['target'], ramp_width['target'], peak_shift['target'], ratio['target'], start_loc['target'], \
    peak_loc['target'], end_loc['target'], min_val['target'], min_loc['target'] = \
        calculate_ramp_features(context, target_ramp, context.mean_induction_start_loc)

    ramp_amp['model'], ramp_width['model'], peak_shift['model'], ratio['model'], start_loc['model'], \
    peak_loc['model'], end_loc['model'], min_val['model'], min_loc['model'] = \
        calculate_ramp_features(context, model_ramp, context.mean_induction_start_loc)

    if disp:
        print 'exp: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, peak_loc: %.1f, ' \
              'end_loc: %.1f, min_val: %.1f, min_loc: %.1f' % \
              (ramp_amp['target'], ramp_width['target'], peak_shift['target'], ratio['target'], start_loc['target'],
               peak_loc['target'], end_loc['target'], min_val['target'], min_loc['target'])
        print 'model: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, peak_loc: %.1f' \
              ', end_loc: %.1f, min_val: %.1f, min_loc: %.1f' % \
              (ramp_amp['model'], ramp_width['model'], peak_shift['model'], ratio['model'], start_loc['model'],
               peak_loc['model'], end_loc['model'], min_val['model'], min_loc['model'])
    sys.stdout.flush()

    start_index, peak_index, end_index, min_index = \
        get_indexes_from_ramp_bounds_with_wrap(ramp_x, start_loc['target'], peak_loc['target'], end_loc['target'],
                                               min_loc['target'])

    model_val_at_target_min_loc = model_ramp[min_index]
    Err += ((model_val_at_target_min_loc - min_val['target']) / context.target_range['delta_min_val']) ** 2.
    model_val_at_target_peak_loc = model_ramp[peak_index]
    Err += ((model_val_at_target_peak_loc - ramp_amp['target']) / context.target_range['delta_peak_val']) ** 2.

    for i in xrange(len(target_ramp)):
        Err += ((target_ramp[i] - model_ramp[i]) / context.target_range['residuals']) ** 2.
    for delta in np.diff(np.insert(delta_weights, 0, delta_weights[-1])):
        Err += (delta / context.target_range['weights_smoothness']) ** 2.

    if full_output:
        return model_ramp, delta_weights, ramp_offset
    else:
        return Err


def get_delta_weights_LSA(target_ramp, input_rate_maps, initial_delta_weights=None, bounds=None, beta=2., ramp_x=None,
                          input_x=None, allow_offset=False, impose_offset=None, plot=False, verbose=1):
    """
    Uses least square approximation to estimate a set of weights to match any arbitrary place field ramp, agnostic
    about underlying kernel, induction velocity, etc.
    :param target_ramp: dict of array
    :param input_rate_maps: array; x=default_interp_x
    :param initial_delta_weights: array
    :param bounds: tuple of float
    :param beta: float; regularization parameter
    :param ramp_x: array (spatial resolution of ramp)
    :param input_x: array (spatial resolution of input_rate_maps)
    :param allow_offset: bool (allow special case where baseline Vm before 1st induction is unknown)
    :param impose_offset: float (impose Vm offset from 1st induction on 2nd induction)
    :param plot: bool
    :param verbose: int
    :return: tuple of array
    """
    if ramp_x is None:
        ramp_x = context.binned_x
    if input_x is None:
        input_x = context.binned_x
    if len(target_ramp) != len(input_x):
        exp_ramp = np.interp(input_x, ramp_x, target_ramp)
    else:
        exp_ramp = np.array(target_ramp)

    input_matrix = np.multiply(input_rate_maps, context.ramp_scaling_factor)
    if initial_delta_weights is None:
        [U, s, Vh] = np.linalg.svd(input_matrix)
        V = Vh.T
        D = np.zeros_like(input_matrix)
        D[np.where(np.eye(*D.shape))] = s / (s ** 2. + beta ** 2.)
        input_matrix_inv = V.dot(D.conj().T).dot(U.conj().T)
        initial_delta_weights = exp_ramp.dot(input_matrix_inv)
    initial_ramp = initial_delta_weights.dot(input_matrix)
    if bounds is None:
        bounds = (0., 3.)
    result = minimize(get_residual_score, initial_delta_weights,
                      args=(target_ramp, input_matrix, ramp_x, input_x, bounds, allow_offset, impose_offset),
                      method='L-BFGS-B', bounds=[bounds] * len(initial_delta_weights),
                      options={'disp': verbose > 1, 'maxiter': 100})

    if verbose > 1:
        print 'get_delta_weights_LSA: process: %i; cell: %i; induction: %i:' % \
              (os.getpid(), context.cell_id, context.induction)
    model_ramp, delta_weights, ramp_offset = \
        get_residual_score(result.x, target_ramp, input_matrix, ramp_x, input_x, bounds, allow_offset,
                                    impose_offset, disp=True, full_output=True)

    if plot:
        x_start = context.mean_induction_start_loc
        x_end = context.mean_induction_stop_loc
        ylim = max(np.max(target_ramp), np.max(model_ramp))
        ymin = min(np.min(target_ramp), np.min(model_ramp))
        fig, axes = plt.subplots(1)
        axes.plot(ramp_x, target_ramp, label='Experiment', color='k')
        axes.plot(ramp_x, initial_ramp, label='Model (Initial)', color='r')
        axes.plot(ramp_x, model_ramp, label='Model (LSA)', color='c')
        axes.hlines(ylim + 0.2, xmin=x_start, xmax=x_end, linewidth=2, colors='k')
        axes.set_xlabel('Location (cm)')
        axes.set_ylabel('Ramp amplitude (mV)')
        axes.set_xlim([0., context.track_length])
        axes.set_ylim([math.floor(ymin), max(math.ceil(ylim), ylim + 0.4)])
        axes.legend(loc='best', frameon=False, framealpha=0.5)
        axes.set_title('Vm ramp')
        clean_axes(axes)
        fig.tight_layout()
        ylim = np.max(delta_weights) + 1.
        ymin = np.min(delta_weights) + 1.
        fig1, axes1 = plt.subplots(1)
        axes1.plot(context.peak_locs, initial_delta_weights + 1., c='r', label='Model (Initial)')
        axes1.plot(context.peak_locs, delta_weights + 1., c='c', label='Model (LSA)')
        axes1.hlines(ylim + 0.2, xmin=x_start, xmax=x_end, linewidth=2, colors='k')
        axes1.set_xlabel('Location (cm)')
        axes1.set_ylabel('Candidate synaptic weights (a.u.)')
        axes1.set_xlim([0., context.track_length])
        axes1.set_ylim([math.floor(ymin), max(math.ceil(ylim), ylim + 0.4)])
        axes1.legend(loc='best', frameon=False, framealpha=0.5)
        clean_axes(axes1)
        fig1.tight_layout()
        plt.show()
        plt.close()

    return model_ramp, delta_weights, ramp_offset


def get_weights_LSA_scaling_factor(input_matrix, bounds=None, beta=2., ramp_x=None, input_x=None, plot=False,
                                   verbose=1):
    """
    Reality check for the least square approximation method to estimate weights. Used to calibrate ramp_scaling_factor.
    Forces weights to a truncated cosine with field_width = 1.2 * input_field_width and ramp_peak = 6 mV.
    :param input_matrix: array
    :param bounds: tuple of float
    :param beta: float; regularization parameter
    :param ramp_x: array (spatial resolution of ramp)
    :param input_x: array (spatial resolution of input_matrix)
    :param plot: bool
    :param verbose: int
    :return: tuple of array
    """
    if ramp_x is None:
        ramp_x = context.binned_x
    if input_x is None:
        input_x = context.binned_x
    modulated_field_center = context.track_length * 0.5
    peak_delta_weight = 1.5
    tuning_amp = peak_delta_weight / 2.
    tuning_offset = tuning_amp
    force_delta_weights = tuning_amp * np.cos(2. * np.pi / (context.input_field_width * 1.2) *
                                              (context.peak_locs - modulated_field_center)) + tuning_offset
    left = np.where(context.peak_locs >= modulated_field_center - context.input_field_width * 1.2 / 2.)[0][0]
    right = np.where(context.peak_locs > modulated_field_center + context.input_field_width * 1.2 / 2.)[0][0]
    force_delta_weights[:left] = 0.
    force_delta_weights[right:] = 0.
    exp_ramp = force_delta_weights.dot(input_matrix)
    ramp_scaling_factor = 6. / np.max(exp_ramp)
    input_matrix = np.multiply(input_matrix, ramp_scaling_factor)
    exp_ramp = force_delta_weights.dot(input_matrix)
    if len(exp_ramp) != len(ramp_x):
        target_ramp = np.interp(ramp_x, input_x, exp_ramp)
    else:
        target_ramp = np.array(exp_ramp)
        ramp_amp, ramp_width, peak_shift, ratio, start_loc, peak_loc, end_loc = {}, {}, {}, {}, {}, {}, {}
    ramp_amp['target'], ramp_width['target'], peak_shift['target'], ratio['target'], start_loc['target'], \
    peak_loc['target'], end_loc['target'] = calculate_ramp_features(target_ramp, context.mean_induction_start_loc)

    [U, s, Vh] = np.linalg.svd(input_matrix)
    V = Vh.T
    D = np.zeros_like(input_matrix)
    D[np.where(np.eye(*D.shape))] = s / (s ** 2. + beta ** 2.)
    input_matrix_inv = V.dot(D.conj().T).dot(U.conj().T)
    delta_weights = exp_ramp.dot(input_matrix_inv)
    SVD_delta_weights = np.array(delta_weights)
    if bounds is None:
        bounds = [context.min_delta_weight, 3.]
    result = optimize.minimize(get_weights_approx_error, delta_weights,
                               args=(target_ramp, input_matrix, ramp_x, input_x, bounds), method='L-BFGS-B',
                               bounds=[bounds] * len(delta_weights), options={'disp': verbose > 1, 'maxiter': 100})
    delta_weights = result.x
    model_ramp = delta_weights.dot(input_matrix)
    if len(model_ramp) != len(ramp_x):
        model_ramp = np.interp(ramp_x, input_x, model_ramp)
    ramp_amp['model'], ramp_width['model'], peak_shift['model'], ratio['model'], start_loc['model'], \
    peak_loc['model'], end_loc['model'] = calculate_ramp_features(model_ramp, context.mean_induction_start_loc)

    if verbose > 0:
        print 'exp: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, peak_loc: %.1f, ' \
              'end_loc: %.1f' % (ramp_amp['target'], ramp_width['target'], peak_shift['target'], ratio['target'],
                                 start_loc['target'], peak_loc['target'], end_loc['target'])
        print 'model: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, peak_loc: %.1f' \
              ', end_loc: %.1f' % (ramp_amp['model'], ramp_width['model'], peak_shift['model'], ratio['model'],
                                   start_loc['model'], peak_loc['model'], end_loc['model'])
    sys.stdout.flush()

    if plot:
        x_start = context.mean_induction_start_loc
        x_end = context.mean_induction_stop_loc
        ylim = max(np.max(target_ramp), np.max(model_ramp))
        ymin = min(np.min(target_ramp), np.min(model_ramp))
        fig, axes = plt.subplots(1)
        axes.plot(ramp_x, target_ramp, label='Experiment', color='k')
        axes.plot(ramp_x, model_ramp, label='Model (LSA)', color='c')
        axes.hlines(ylim + 0.2, xmin=x_start, xmax=x_end, linewidth=2, colors='k')
        axes.set_xlabel('Location (cm)')
        axes.set_ylabel('Ramp amplitude (mV)')
        axes.set_xlim([0., context.track_length])
        axes.set_ylim([math.floor(ymin), max(math.ceil(ylim), ylim + 0.4)])
        axes.legend(loc='best', frameon=False, framealpha=0.5)
        axes.set_title('Vm ramp')
        clean_axes(axes)
        fig.tight_layout()
        ylim = np.max(delta_weights) + 1.
        ymin = np.min(delta_weights) + 1.
        fig1, axes1 = plt.subplots(1)
        axes1.plot(context.peak_locs, force_delta_weights + 1., label='Target', c='k')
        axes1.plot(context.peak_locs, SVD_delta_weights + 1., c='r', label='Model (SVD)')
        axes1.plot(context.peak_locs, delta_weights + 1., c='c', label='Model (LSA)')
        axes1.hlines(ylim + 0.2, xmin=x_start, xmax=x_end, linewidth=2, colors='k')
        axes1.set_xlabel('Location (cm)')
        axes1.set_ylabel('Candidate synaptic weights (a.u.)')
        axes1.set_xlim([0., context.track_length])
        axes1.set_ylim([math.floor(ymin), max(math.ceil(ylim), ylim + 0.4)])
        axes1.legend(loc='best', frameon=False, framealpha=0.5)
        clean_axes(axes1)
        fig1.tight_layout()
        plt.show()
        plt.close()

    return target_ramp, force_delta_weights, model_ramp, delta_weights, ramp_scaling_factor


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1,sys.argv)+1):])