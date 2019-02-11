"""
These methods aim to optimize a parametric model of bidirectional, state-dependent behavioral timescale synaptic
plasticity to account for the width and amplitude of place fields in an experimental data set from the Magee lab that
includes:
1) Silent cells converted to place cells by spontaneous plateaus
2) Silent cells converted to place cells by experimentally induced plateaus
3) Existing place cells that shift their place field locations after an experimentally induced plateau

Features/assumptions of the phenomenological model:
1) Synaptic weights in a silent cell are all = 1 prior to field induction 1. w(t0) = 1
2) Activity at each synapse generates a long duration 'local plasticity signal', or an 'eligibility trace' for
synaptic plasticity.
3) Dendritic plateaus generate a long duration 'global plasticity signal', or a 'gating trace' for synaptic plasticity.
4) Synaptic weights w1 at time t1 after a plateau are a function of the initial weights w0 at time t0, and the two
plasticity signals.

Features/assumptions of the mechanistic model:
1) Synaptic strength is equivalent to the number of AMPA-Rs at a synapse (quantal size). 
2) Dendritic plateaus generate a global gating signal that mobilizes AMPA-Rs, allowing them to be either inserted or
removed from synapses.
3) Activity at each synapse generates a local plasticity signal that, in conjunction with the global gating signal,
can activate a forward process to increase the number of AMPA-Rs stably incorporated into synapses.
4) Activity at each synapse generates a local plasticity signal that, in conjunction with the global gating signal,
can activate a reverse process to decrease the number of AMPA-Rs stably incorporated into synapses.
5) AMPAR-s can be in 2 states (Markov-style kinetic scheme):

     rMC0 * global_signal * f_p(local_signal)
M (mobile) <----------------------> C (captured by a synapse)
     rCM0 * global_signal * f_d(local_signal)

6) At rest 100% of non-synaptic receptors are in state M, mobile and available for synaptic capture.
7) global_signals are pooled across all cells and normalized to a peak value of 1.
8) local_signals are pooled across all cells and normalized to a peak value of 1.
9) f_p represents the "sensitivity" of the forward process to the presence of the local_signal.  The transformation f_p
has the flexibility to be any segment of a sigmoid (so can be linear, exponential rise, or saturating).
10) f_d represents the "sensitivity" of the reverse process to the presence of the local_signal. local_signals
are pooled across all cells and normalized to a peak value of 1. The transformation f_d is a piecewise function with two
components: a rising phase and a decaying phase. Each phase has the flexibility to be any segment of a sigmoid (so
can be linear, exponential rise, or saturating).

BTSP_E vs. orig BTSP: Weights updated once per lap instead of once per time step. Relaxed constraints on sigmoid slope
compared to original version.
"""
__author__ = 'milsteina'
from BTSP_utils import *
from nested.optimize_utils import *
import click


context = Context()


def config_worker():
    """

    """
    context.data_file_path = context.output_dir + '/' + context.data_file_name
    init_context()


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
        allow_offset_cell_ids = [int(cell_id) for cell_id in f['data'] if '2' in f['data'][cell_id] and
                                 ('1' not in f['data'][cell_id] or
                                  'before' not in f['data'][cell_id]['1']['raw']['exp_ramp'])]
    if context.verbose > 1:
        print 'pid: %i; optimize_BTSP_E_CA1: processing the following data_keys: %s' % \
              (os.getpid(), str(context.data_keys))
    self_consistent_cell_ids = [cell_id for (cell_id, induction) in context.data_keys if induction == 1 and
                                 (cell_id, 2) in context.data_keys]
    down_dt = 10.  # ms, to speed up optimization
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
            raise KeyError('optimize_BTSP_E_CA1: no data found for cell_id: %s, induction: %s' %
                           (cell_key, induction_key))
        else:
            context.cell_id = cell_id
            context.induction = induction
        this_group = f['data'][cell_key][induction_key]
        context.induction_locs = this_group.attrs['induction_locs']
        context.induction_durs = this_group.attrs['induction_durs']
        context.exp_ramp_raw = {}
        context.exp_ramp = {}
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

        # TODO: Need to re-interpolate exp_ramp_vs_t when replacing exp_ramp['after'] with ['before'] from induction 2
        context.exp_ramp_vs_t = {'after': this_group['processed']['exp_ramp_vs_t']['after'][:]}
        if 'before' in this_group['processed']['exp_ramp']:
            context.exp_ramp['before'] = this_group['processed']['exp_ramp']['before'][:]
            context.exp_ramp_vs_t['before'] = this_group['processed']['exp_ramp_vs_t']['before'][:]
        context.LSA_ramp = {}
        context.LSA_ramp_offset = {}
        if 'before' in this_group['processed']['LSA_ramp']:
            context.LSA_ramp['before'] = this_group['processed']['LSA_ramp']['before'][:]
            context.LSA_ramp_offset['before'] = this_group['processed']['LSA_ramp']['before'].attrs['ramp_offset']
        context.LSA_weights = {}
        context.LSA_weights['before'] = this_group['processed']['LSA_weights']['before'][:]
        context.complete_run_vel = this_group['complete']['run_vel'][:]
        context.complete_run_vel_gate = this_group['complete']['run_vel_gate'][:]
        context.complete_position = this_group['complete']['position'][:]
        context.complete_t = this_group['complete']['t'][:]
        context.induction_gate = this_group['complete']['induction_gate'][:]
        # if both induction 1 and 2 are in dataset, for self-consistency, use same target_ramp
        if induction == 1 and '2' in f['data'][cell_key]:
            this_group = f['data'][cell_key]['2']
            context.exp_ramp_raw['after'] = this_group['raw']['exp_ramp']['before'][:]
            context.exp_ramp['after'] = this_group['processed']['exp_ramp']['before'][:]
            context.LSA_ramp['after'] = this_group['processed']['LSA_ramp']['before'][:]
            context.LSA_ramp_offset['after'] = this_group['processed']['LSA_ramp']['before'].attrs['ramp_offset']
            context.LSA_weights['after'] = this_group['processed']['LSA_weights']['before'][:]
        else:
            context.exp_ramp_raw['after'] = this_group['raw']['exp_ramp']['after'][:]
            context.exp_ramp['after'] = this_group['processed']['exp_ramp']['after'][:]
            context.LSA_ramp['after'] = this_group['processed']['LSA_ramp']['after'][:]
            context.LSA_ramp_offset['after'] = this_group['processed']['LSA_ramp']['after'].attrs['ramp_offset']
            context.LSA_weights['after'] = this_group['processed']['LSA_weights']['after'][:]
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
    context.complete_rate_maps = get_complete_rate_maps(context.input_rate_maps, context.binned_x)
    context.down_t = np.arange(context.complete_t[0], context.complete_t[-1] + context.down_dt / 2., context.down_dt)
    context.down_rate_maps = []
    for rate_map in context.complete_rate_maps:
        this_down_rate_map = np.interp(context.down_t, context.complete_t, rate_map)
        context.down_rate_maps.append(this_down_rate_map)
    context.down_induction_gate = np.interp(context.down_t, context.complete_t, context.induction_gate)
    if context.verbose > 1:
        print 'optimize_BTSP_E_CA1: process: %i loaded data for cell: %i, induction: %i' % \
              (os.getpid(), cell_id, induction)


def update_model_params(x, local_context):
    """

    :param x: array
    :param local_context: :class:'Context'
    """
    if local_context is None:
        local_context = context
    local_context.update(param_array_to_dict(x, local_context.param_names))


def plot_data():
    """

    """
    import matplotlib as mpl

    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['font.size'] = 14.
    # mpl.rcParams['font.size'] = 14.
    # mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['font.sans-serif'] = 'Calibri'
    # mpl.rcParams['font.sans-serif'] = 'Myriad Pro'
    mpl.rcParams['text.usetex'] = False
    # mpl.rcParams['figure.figsize'] = 6, 4.3

    fig, axes = plt.subplots(1)
    for group in context.position:
        for i, this_position in enumerate(context.position[group]):
            this_t = context.t[group][i]
            axes.plot(this_t / 1000., this_position, label=group + str(i))
    axes.set_xlabel('Time (s)')
    axes.set_ylabel('Position (cm)')
    axes.set_title('Interpolated position')
    axes.legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
    fig.tight_layout(h_pad=0.2)
    clean_axes(axes)

    fig, axes = plt.subplots(1)
    axes2 = axes.twinx()
    axes.plot(context.complete_t / 1000., context.complete_run_vel)
    axes2.plot(context.complete_t / 1000., context.complete_run_vel_gate, c='k')
    axes.set_xlabel('Time (s)')
    axes.set_ylabel('Running speed (cm/s)')
    clean_axes(axes)
    axes2.tick_params(direction='out')
    axes2.spines['top'].set_visible(False)
    axes2.spines['left'].set_visible(False)
    axes2.get_xaxis().tick_bottom()
    axes2.get_yaxis().tick_right()
    fig.tight_layout(h_pad=0.2)

    fig, axes = plt.subplots(2, 2)
    axes[1][0].plot(context.binned_x, context.exp_ramp['after'])
    if 'before' in context.exp_ramp:
        axes[1][0].plot(context.binned_x, context.exp_ramp['before'])
        axes[1][0].plot(context.binned_x, context.exp_ramp_raw['before'])
    axes[1][0].plot(context.binned_x, context.exp_ramp_raw['after'])
    axes[1][0].set_xlabel('Position (cm)')
    axes[0][0].set_xlabel('Position (cm)')
    axes[1][0].set_ylabel('Ramp amplitude (mV)')
    axes[1][1].set_ylabel('Ramp amplitude (mV)')
    axes[1][1].set_xlabel('Time (s)')
    axes[0][1].set_xlabel('Time (s)')
    axes[0][0].set_ylabel('Induction current (nA)')
    axes[0][1].set_ylabel('Induction gate (a.u.)')
    for i, this_position in enumerate(context.position['induction']):
        this_t = context.t['induction'][i]
        this_current = context.current[i]
        this_induction_gate = np.zeros_like(this_current)
        indexes = np.where(this_current >= 0.5 * np.max(this_current))[0]
        this_induction_gate[indexes] = 1.
        start_index = indexes[0]
        this_induction_loc = context.induction_locs[i]
        this_induction_dur = context.induction_durs[i]
        axes[0][0].plot(this_position, this_current, label='Lap %i: Loc: %i cm, Dur: %i ms' %
                                                           (i, this_induction_loc, this_induction_dur))
        axes[0][1].plot(np.subtract(this_t, this_t[start_index]) / 1000., this_induction_gate)
    mean_induction_index = np.where(context.mean_position >= context.mean_induction_start_loc)[0][0]
    mean_induction_onset = context.mean_t[mean_induction_index]
    peak_val, ramp_width, peak_shift, ratio, start_loc, peak_loc, end_loc, min_val, min_loc = \
        calculate_ramp_features(context, context.exp_ramp['after'], context.mean_induction_start_loc)
    start_index, peak_index, end_index, min_index = get_indexes_from_ramp_bounds_with_wrap(context.binned_x, start_loc, peak_loc,
                                                                                end_loc, min_loc)
    axes[1][0].scatter(context.binned_x[[start_index, peak_index, end_index]],
                       context.exp_ramp['after'][[start_index, peak_index, end_index]])
    start_index, peak_index, end_index, min_index = get_indexes_from_ramp_bounds_with_wrap(context.mean_position, start_loc,
                                                                                peak_loc, end_loc, min_loc)
    this_shifted_t = np.subtract(context.mean_t, mean_induction_onset) / 1000.
    axes[1][1].plot(this_shifted_t, context.exp_ramp_vs_t['after'])
    axes[1][1].scatter(this_shifted_t[[start_index, peak_index, end_index]],
                       context.exp_ramp_vs_t['after'][[start_index, peak_index, end_index]])
    if 'before' in context.exp_ramp_vs_t:
        axes[1][1].plot(this_shifted_t, context.exp_ramp_vs_t['before'])
    axes[0][0].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
    clean_axes(axes)
    fig.tight_layout(h_pad=0.2)

    fig, axes = plt.subplots(1, 2)
    x_start = context.mean_induction_start_loc
    x_end = context.mean_induction_stop_loc
    max_ramp = max(np.max(context.LSA_ramp['after']), np.max(context.exp_ramp['after']),
                   np.max(context.exp_ramp_raw['after']))
    max_weights = np.max(context.LSA_weights['after'])
    axes[0].plot(context.binned_x, context.LSA_ramp['after'])
    axes[1].plot(context.peak_locs, context.LSA_weights['after'] + 1., label='After induction')
    if 'before' in context.exp_ramp:
        axes[0].plot(context.binned_x, context.LSA_ramp['before'])
        axes[1].plot(context.peak_locs, context.LSA_weights['before'] + 1., label='Before induction')
        max_weights = max(max_weights, np.max(context.LSA_weights['before']))
    max_weights += 1
    axes[0].plot(context.binned_x, context.exp_ramp['after'])
    axes[0].plot(context.binned_x, context.exp_ramp_raw['after'])
    if 'before' in context.exp_ramp:
        axes[0].plot(context.binned_x, context.exp_ramp['before'])
        axes[0].plot(context.binned_x, context.exp_ramp_raw['before'])
        max_ramp = max(max_ramp, np.max(context.LSA_ramp['before']), np.max(context.exp_ramp['before']),
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
    fig.tight_layout(h_pad=0.2)
    plt.subplots_adjust(top=0.9)
    plt.show()
    plt.close()


def plot_data_summary():
    """

    """
    import matplotlib as mpl

    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['font.size'] = 20.
    # mpl.rcParams['font.size'] = 14.
    # mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['font.sans-serif'] = 'Calibri'
    # mpl.rcParams['font.sans-serif'] = 'Myriad Pro'
    mpl.rcParams['text.usetex'] = False
    # mpl.rcParams['figure.figsize'] = 6, 4.3

    fig, axes = plt.subplots(1, 3, figsize=[16., 3.75])
    pretty_position = np.array(context.complete_position % context.track_length)
    for i in xrange(1, len(pretty_position)):
        if pretty_position[i] - pretty_position[i-1] < -context.track_length / 2.:
            pretty_position[i] = np.nan

    axes[0].plot(context.complete_t / 1000., pretty_position, c='k')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Location (cm)')
    axes[0].hlines([context.track_length * 1.05] * len(context.induction_start_times),
                   xmin=context.induction_start_times / 1000.,
                   xmax=context.induction_stop_times / 1000., linewidth=2)
    axes[0].set_ylim([0., context.track_length * 1.1])
    ymax0 = max(10., np.max(context.exp_ramp['after']) + 1.)
    ymin0 = min(-1., np.min(context.exp_ramp['after']))
    if 'before' in context.exp_ramp:
        axes[1].plot(context.binned_x, context.exp_ramp['before'], c='darkgrey', label='1st induction')
        # axes[1].plot(context.binned_x, context.exp_ramp_raw['before'])
        ymax0 = max(ymax0, np.max(context.exp_ramp['before']) + 1.)
        ymin0 = min(ymin0, np.min(context.exp_ramp['before']))
        axes[1].plot(context.binned_x, context.exp_ramp['after'], c='r', label='2nd induction')
    else:
        axes[1].plot(context.binned_x, context.exp_ramp['after'], c='r', label='1st induction')
    axes[1].hlines(ymax0 * 1.05, xmin=context.mean_induction_start_loc, xmax=context.mean_induction_stop_loc)
    axes[1].set_xlabel('Location (cm)')
    axes[1].set_ylabel('Depolarization\namplitude (mV)')
    axes[2].set_ylabel('Change in\namplitude (mV)')
    axes[2].set_xlabel('Time (s)')
    axes[1].set_ylim([ymin0, ymax0 * 1.1])
    axes[1].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
    mean_induction_start_index = np.where(context.mean_position >= context.mean_induction_start_loc)[0][0]
    mean_induction_stop_index = np.where(context.mean_position >= context.mean_induction_stop_loc)[0][0]
    mean_induction_onset = context.mean_t[mean_induction_start_index]
    mean_induction_offset = context.mean_t[mean_induction_stop_index]
    mean_induction_dur = (mean_induction_offset - mean_induction_onset) / 1000.
    
    this_shifted_t = np.subtract(context.mean_t, mean_induction_onset) / 1000.
    this_delta_ramp = np.array(context.exp_ramp_vs_t['after'])
    if 'before' in context.exp_ramp_vs_t:
        this_delta_ramp = np.subtract(this_delta_ramp, context.exp_ramp_vs_t['before'])
    ymin1 = np.min(this_delta_ramp) - 1.
    ymax1 = np.max(this_delta_ramp) + 1.
    axes[2].plot(this_shifted_t, this_delta_ramp, c='k')
    axes[2].hlines(ymax1 * 1.05, xmin=0., xmax=mean_induction_dur)
    axes[2].set_ylim([ymin1, ymax1 * 1.1])
    xmin, xmax = axes[2].get_xlim()
    axes[2].plot([xmin, xmax], [0., 0.], c='darkgrey', alpha=0.5, ls='--')
    clean_axes(axes)
    fig.tight_layout(h_pad=0.2)
    fig.suptitle('Cell: %i' % context.cell_id, fontsize=mpl.rcParams['font.size'])
    plt.subplots_adjust(top=0.9)
    plt.show()
    plt.close()


def get_complete_rate_maps(input_rate_maps, input_x):
    """
    :param input_rate_maps: array
    :param input_x: array (x resolution of input)
    :return: list of array
    """
    complete_rate_maps = []
    for j in xrange(len(input_rate_maps)):
        this_complete_rate_map = np.array([])
        for group in ['pre', 'induction', 'post']:
            for i, this_position in enumerate(context.position[group]):
                this_rate_map = np.interp(this_position, input_x, input_rate_maps[j])
                this_complete_rate_map = np.append(this_complete_rate_map, this_rate_map)
        this_complete_rate_map = np.multiply(this_complete_rate_map, context.complete_run_vel_gate)
        if len(this_complete_rate_map) != len(context.complete_run_vel_gate):
            print 'get_complete_rate_maps: mismatched array length'
        complete_rate_maps.append(this_complete_rate_map)
    return complete_rate_maps


def get_filter(rise, decay, max_time_scale, dt=None):
    """
    :param rise: float
    :param decay: float
    :param max_time_scale: float
    :param dt: float
    :return: array, array
    """
    if dt is None:
        dt = context.dt
    filter_t = np.arange(0., 6. * max_time_scale, dt)
    filter = np.exp(-filter_t / decay) - np.exp(-filter_t / rise)
    peak_index = np.where(filter == np.max(filter))[0][0]
    decay_indexes = np.where(filter[peak_index:] < 0.001 * np.max(filter))[0]
    if np.any(decay_indexes):
        filter = filter[:peak_index + decay_indexes[0]]
    filter /= np.sum(filter)
    filter_t = filter_t[:len(filter)]
    return filter_t, filter


def get_signal_filters(local_signal_rise, local_signal_decay, global_signal_rise, global_signal_decay, dt=None,
                       plot=False):
    """
    :param local_signal_rise: float
    :param local_signal_decay: float
    :param global_signal_rise: float
    :param global_signal_decay: float
    :param dt: float
    :param plot: bool
    :return: array, array
    """
    max_time_scale = max(local_signal_rise + local_signal_decay, global_signal_rise + global_signal_decay)
    local_signal_filter_t, local_signal_filter = get_filter(local_signal_rise, local_signal_decay, max_time_scale, dt)
    global_filter_t, global_filter = get_filter(global_signal_rise, global_signal_decay, max_time_scale, dt)
    if plot:
        fig, axes = plt.subplots(1)
        axes.plot(local_signal_filter_t / 1000., local_signal_filter / np.max(local_signal_filter), color='r',
                  label='Local plasticity signal filter')
        axes.plot(global_filter_t / 1000., global_filter / np.max(global_filter), color='k',
                  label='Global plasticity signal filter')
        axes.set_xlabel('Time (s)')
        axes.set_ylabel('Normalized filter amplitude')
        axes.set_title('Plasticity signal filters')
        axes.legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
        axes.set_xlim(-0.5, max(5000., local_signal_filter_t[-1], global_filter_t[-1]) / 1000.)
        clean_axes(axes)
        fig.tight_layout()
        fig.show()
    return local_signal_filter_t, local_signal_filter, global_filter_t, global_filter


def get_local_signal(rate_map, local_filter, dt):
    """

    :param rate_map: array
    :param local_filter: array
    :param dt: float
    :return: array
    """
    return np.convolve(0.001 * dt * rate_map, local_filter)[:len(rate_map)]


def get_global_signal(induction_gate, global_filter):
    """

    :param induction_gate: array
    :param global_filter: array
    :return: array
    """
    return np.convolve(induction_gate, global_filter)[:len(induction_gate)]


def get_local_signal_population(local_filter):
    """

    :param local_filter:
    :return:
    """
    local_signals = []
    for rate_map in context.down_rate_maps:
        local_signals.append(get_local_signal(rate_map, local_filter, context.down_dt))
    return local_signals


def get_args_static_signal_amplitudes():
    """
    A nested map operation is required to compute model signal amplitudes. The arguments to be mapped are the same
    (static) for each set of parameters.
    :return: list of list
    """
    with h5py.File(context.data_file_path, 'r') as f:
        data_keys = []
        for cell_key in f['data']:
            for induction_key in f['data'][cell_key]:
                data_keys.append((int(cell_key), int(induction_key)))
    return zip(*data_keys)


def compute_features_signal_amplitudes(x, cell_id=None, induction=None, export=False, plot=False):
    """

    :param x: array
    :param cell_id: int
    :param induction: int
    :param export: bool
    :param plot: bool
    :return: dict
    """
    import_data(cell_id, induction)
    update_source_contexts(x, context)
    if context.verbose > 1:
        print 'Process: %i: computing signal_amplitude features for cell_id: %i, induction: %i with x: %s' % \
              (os.getpid(), context.cell_id, context.induction, ', '.join('%.3E' % i for i in x))
    start_time = time.time()
    local_signal_filter_t, local_signal_filter, global_filter_t, global_filter = \
        get_signal_filters(context.local_signal_rise, context.local_signal_decay, context.global_signal_rise,
                           context.global_signal_decay, context.down_dt, plot)
    global_signal = get_global_signal(context.down_induction_gate, global_filter)
    local_signals = get_local_signal_population(local_signal_filter)
    local_signal_peaks = [np.max(local_signal) for local_signal in local_signals]
    if plot:
        fig, axes = plt.subplots(1)
        hist, edges = np.histogram(local_signal_peaks, density=True)
        bin_width = edges[1] - edges[0]
        axes.plot(edges[:-1]+bin_width/2., hist * bin_width, c='r', label='Local plasticity signals')
        axes.set_xlabel('Peak local plasticity signal amplitudes (a.u.)')
        axes.set_ylabel('Probability')
        axes.set_title('Local signal amplitude distribution')
        axes.legend(loc='best', frameon=False, framealpha=0.5)
        clean_axes(axes)
        fig.tight_layout()
        fig.show()
    result = {'local_signal_peaks': local_signal_peaks,
              'global_signal_peak': np.max(global_signal)}
    if context.verbose > 1:
        print 'Process: %i: computing signal_amplitude features for cell_id: %i, induction: %i took %.1f s' % \
              (os.getpid(), context.cell_id, context.induction, time.time() - start_time)
    return {cell_id: {induction: result}}


def filter_features_signal_amplitudes(primitives, current_features, export=False, plot=False):
    """

    :param primitives: list of dict (each dict contains results from a single simulation)
    :param current_features: dict
    :param export: bool
    :param plot: bool
    :return: dict
    """
    all_inputs_local_signal_peaks = np.array([], dtype='float32')
    each_cell_local_signal_peak = []
    each_cell_global_signal_peak = []
    for this_dict in primitives:
        for cell_id in this_dict:
            for induction_id in this_dict[cell_id]:
                each_cell_global_signal_peak.append(this_dict[cell_id][induction_id]['global_signal_peak'])
                each_cell_local_signal_peak.append(
                    np.max(this_dict[cell_id][induction_id]['local_signal_peaks']))
                all_inputs_local_signal_peaks = np.append(all_inputs_local_signal_peaks,
                                                          this_dict[cell_id][induction_id]['local_signal_peaks'])
    if plot:
        fig, axes = plt.subplots(1)
        hist, edges = np.histogram(all_inputs_local_signal_peaks, bins=10, density=True)
        bin_width = edges[1] - edges[0]
        axes.plot(edges[:-1] + bin_width / 2., hist * bin_width, c='r', label='Local plasticity signals')
        axes.set_xlabel('Peak local plasticity signal amplitudes (a.u.)')
        axes.set_ylabel('Probability')
        axes.set_title('Local signal amplitude distribution (all inputs, all cells)')
        axes.legend(loc='best', frameon=False, framealpha=0.5)
        clean_axes(axes)
        fig.tight_layout()
        fig.show()

        fig, axes = plt.subplots(1)
        hist, edges = np.histogram(each_cell_local_signal_peak, bins=min(10, len(primitives)), density=True)
        bin_width = edges[1] - edges[0]
        axes.plot(edges[:-1] + bin_width / 2., hist * bin_width, c='r', label='Local potentiation signals')
        axes.set_xlabel('Peak local plasticity signal amplitudes (a.u.)')
        axes.set_ylabel('Probability')
        axes.set_title('Local signal amplitude distribution (each cell)')
        axes.legend(loc='best', frameon=False, framealpha=0.5)
        clean_axes(axes)
        fig.tight_layout()
        fig.show()

        hist, edges = np.histogram(each_cell_global_signal_peak, bins=min(10, len(primitives)), density=True)
        bin_width = edges[1] - edges[0]
        fig, axes = plt.subplots(1)
        axes.plot(edges[:-1] + bin_width / 2., hist * bin_width, c='k')
        axes.set_xlabel('Peak global plasticity signal amplitudes (a.u.)')
        axes.set_ylabel('Probability')
        axes.set_title('Global signal amplitude distribution (each cell)')
        clean_axes(axes)
        fig.tight_layout()
        fig.show()
    signal_amplitude_features = {'local_signal_max': np.max(each_cell_local_signal_peak),
                                 'global_signal_max': np.max(each_cell_global_signal_peak)
                                 }
    if context.verbose > 1:
        print 'Process: %i: signal_amplitude features; local_signal_max: %.2E, global_signal_max: %.2E' % \
              (os.getpid(), signal_amplitude_features['local_signal_max'],
               signal_amplitude_features['global_signal_max'])
    return signal_amplitude_features


def get_model_ramp(delta_weights, input_x=None, ramp_x=None, allow_offset=False, impose_offset=None, plot=False):
    """

    :param delta_weights: array
    :param input_x: array (x resolution of inputs)
    :param ramp_x: array (x resolution of ramp)
    :param allow_offset: bool (allow special case where baseline Vm before 1st induction is unknown)
    :param impose_offset: float (impose Vm offset from 1st induction on 2nd induction)
    :param plot: bool
    :return: array
    """
    if input_x is None:
        input_x = context.binned_x
    if ramp_x is None:
        ramp_x = context.binned_x
    model_ramp = np.multiply(delta_weights.dot(context.input_rate_maps), context.ramp_scaling_factor)
    if len(model_ramp) != len(ramp_x):
        model_ramp = np.interp(ramp_x, input_x, model_ramp)

    if impose_offset is not None:
        ramp_offset = impose_offset
        model_ramp -= impose_offset
    elif allow_offset:
        model_ramp, ramp_offset = subtract_baseline(model_ramp)
    else:
        ramp_offset = 0.

    if plot:
        fig, axes = plt.subplots(1)
        max_ramp = max(np.max(model_ramp), np.max(context.exp_ramp['after']))
        axes.hlines(max_ramp * 1.2,
                       xmin=context.mean_induction_start_loc,
                       xmax=context.mean_induction_stop_loc, linewidth=2)
        axes.plot(context.binned_x, context.exp_ramp['after'], label='Experiment')
        axes.plot(context.binned_x, model_ramp, label='Model')
        axes.set_ylim(-0.5, max_ramp * 1.4)
        axes.set_xlabel('Location (cm)')
        axes.set_ylabel('Ramp amplitude (mV)')
        axes.set_title('Cell_id: %i, Induction: %i' % (context.cell_id, context.induction))
        clean_axes(axes)
        fig.tight_layout()
        plt.show()
        plt.close()

    return model_ramp, ramp_offset


def get_residual_score(delta_weights, target_ramp, input_x=None, ramp_x=None, bounds=None, allow_offset=False,
                       impose_offset=None, disp=False, full_output=False):
    """

    :param delta_weights: array
    :param target_ramp: array
    :param input_x: array
    :param ramp_x: array
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
    if ramp_x is None:
        ramp_x = context.binned_x
    if input_x is None:
        input_x = context.binned_x
    if len(target_ramp) != len(input_x):
        exp_ramp = np.interp(input_x, ramp_x, target_ramp)
    else:
        exp_ramp = np.array(target_ramp)

    model_ramp, ramp_offset = get_model_ramp(delta_weights, input_x=input_x, ramp_x=ramp_x, allow_offset=allow_offset,
                                             impose_offset=impose_offset)
    Err = 0.
    if allow_offset:
        Err += (ramp_offset / context.target_range['ramp_offset']) ** 2.

    ramp_amp, ramp_width, peak_shift, ratio, start_loc, peak_loc, end_loc, min_val, min_loc = {}, {}, {}, {}, {}, {}, \
                                                                                              {}, {}, {}
    ramp_amp['target'], ramp_width['target'], peak_shift['target'], ratio['target'], start_loc['target'], \
    peak_loc['target'], end_loc['target'], min_val['target'], min_loc['target'] = \
        calculate_ramp_features(context, exp_ramp, context.mean_induction_start_loc)

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
    Err += ((min_val['model'] - min_val['target']) / context.target_range['delta_min_val']) ** 2.
    model_val_at_target_peak_loc = model_ramp[peak_index]
    Err += ((model_val_at_target_peak_loc - ramp_amp['target']) / context.target_range['delta_peak_val']) ** 2.

    for i in xrange(len(exp_ramp)):
        Err += ((exp_ramp[i] - model_ramp[i]) / context.target_range['residuals']) ** 2.
    # regularization
    for delta in np.diff(np.insert(delta_weights, 0, delta_weights[-1])):
        Err += (delta / context.target_range['weights_smoothness']) ** 2.

    if full_output:
        return model_ramp, delta_weights, ramp_offset, Err
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
                      args=(target_ramp, input_x, ramp_x, bounds, allow_offset, impose_offset),
                      method='L-BFGS-B', bounds=[bounds] * len(initial_delta_weights),
                      options={'disp': verbose > 1, 'maxiter': 100})

    if verbose > 1:
        print 'get_delta_weights_LSA: process: %i; cell: %i; induction: %i:' % \
              (os.getpid(), context.cell_id, context.induction)
    model_ramp, delta_weights, ramp_offset, residual_score = \
        get_residual_score(result.x, target_ramp, input_x, ramp_x, bounds, allow_offset, impose_offset,
                           disp=verbose > 1, full_output=True)

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

    return model_ramp, delta_weights, ramp_offset, residual_score


def calculate_model_ramp(local_signal_peak=None, global_signal_peak=None, export=False, plot=False):
    """

    :param local_signal_peak: float
    :param global_signal_peak: float
    :param export: bool
    :param plot: bool
    :return: dict
    """
    local_signal_filter_t, local_signal_filter, global_filter_t, global_filter = \
        get_signal_filters(context.local_signal_rise, context.local_signal_decay, context.global_signal_rise,
                           context.global_signal_decay, context.down_dt, plot)
    global_signal = np.divide(get_global_signal(context.down_induction_gate, global_filter), global_signal_peak)
    local_signals = np.divide(get_local_signal_population(local_signal_filter), local_signal_peak)

    signal_xrange = np.linspace(0., 1., 10000)
    pot_rate = np.vectorize(scaled_single_sigmoid(context.rMC_th, context.rMC_peak + context.rMC_th, signal_xrange))
    try:
        depot_rate = np.vectorize(scaled_double_sigmoid(context.rCM_th1, context.rCM_th1 + context.rCM_peak1,
                                                        context.rCM_th2, context.rCM_th2 - context.rCM_peak2,
                                                        signal_xrange, y_end=context.rCM_min2))
    except:
        if context.verbose > 0:
            print 'optimize_BTSP_E_CA1: calculate_model_ramp: pid: %i ; aborting - invalid parameters for ' \
                  'depot_rate' % os.getpid()
        return dict()
    if plot:
        fig, axes = plt.subplots(1)
        axes.plot(signal_xrange, pot_rate(signal_xrange), label='Potentiation rate')
        axes.plot(signal_xrange, depot_rate(signal_xrange), label='De-potentiation rate')
        axes.set_xlabel('Normalized plasticity signal amplitude (a.u.)')
        axes.set_ylabel('Normalized rate')
        axes.set_title('Plasticity signal transformations')
        axes.legend(loc='best', frameon=False, framealpha=0.5)
        clean_axes(axes)
        fig.tight_layout()
        fig.show()

    peak_weight = context.peak_delta_weight + 1.

    allow_offset = False
    initial_delta_weights = context.LSA_weights['before']
    # re-compute initial weights if they are out of the current weight bounds
    if context.induction == 1:
        if 'before' in context.exp_ramp:
            if not np.all((context.min_delta_weight <= initial_delta_weights) &
                          (initial_delta_weights <= context.peak_delta_weight)):
                initial_ramp, initial_delta_weights, initial_ramp_offset, discard_residual_score = \
                    get_delta_weights_LSA(context.exp_ramp['before'], context.input_rate_maps,
                                          initial_delta_weights=initial_delta_weights,
                                          bounds=(context.min_delta_weight, context.peak_delta_weight),
                                          verbose=context.verbose)
                if context.verbose > 1:
                    print 'Process: %i; re-computed initial weights: cell_id: %i, before induction: %i,' \
                          ' ramp_offset: %.3f' % (os.getpid(), context.cell_id, context.induction, initial_ramp_offset)
            else:
                initial_ramp = context.LSA_ramp['before']
            initial_ramp_offset = None
        else:
            initial_ramp, discard_ramp_offset = get_model_ramp(initial_delta_weights)
            initial_ramp_offset = context.LSA_ramp_offset['after']
    else:
        if context.cell_id in context.allow_offset_cell_ids:
            allow_offset = True
        if not np.all((context.min_delta_weight <= initial_delta_weights) &
                      (initial_delta_weights <= context.peak_delta_weight)):
            initial_ramp, initial_delta_weights, initial_ramp_offset, discard_residual_score = \
                get_delta_weights_LSA(context.exp_ramp['before'], context.input_rate_maps,
                                      initial_delta_weights=initial_delta_weights,
                                      bounds=(context.min_delta_weight, context.peak_delta_weight),
                                      allow_offset=allow_offset, verbose=context.verbose)
            if context.verbose > 1:
                print 'Process: %i; re-computed initial weights: cell_id: %i, before induction: %i, ' \
                      'ramp_offset: %.3f' % (os.getpid(), context.cell_id, context.induction, initial_ramp_offset)
        else:
            initial_ramp = context.LSA_ramp['before']
            initial_ramp_offset = context.LSA_ramp_offset['before']
        allow_offset = False

    delta_weights_snapshots = [initial_delta_weights]
    current_ramp = initial_ramp
    ramp_snapshots = [current_ramp]
    initial_weights = np.divide(np.add(initial_delta_weights, 1.), peak_weight)
    current_weights = np.array(initial_weights)

    target_ramp = context.exp_ramp['after']

    prev_residual_score = 0.
    for i in xrange(len(target_ramp)):
        prev_residual_score += ((current_ramp[i] - target_ramp[i]) / context.target_range['residuals']) ** 2.

    if plot:
        fig, axes = plt.subplots()
        fig.suptitle('Induction: %i' % context.induction)
        axes.plot(context.down_t / 1000., global_signal)
        axes.set_ylabel('Plasticity gating signal')
        axes.set_xlabel('Time (s)')

        fig2, axes2 = plt.subplots(1, 2, sharex=True)
        fig2.suptitle('Induction: %i' % context.induction)
        axes2[0].plot(context.binned_x, initial_ramp, c='k', label='Before')
        axes2[0].set_ylabel('Ramp amplitude (mV)')
        axes2[0].set_xlabel('Location (cm)')
        axes2[1].set_ylabel('Change in synaptic weight')
        axes2[1].set_xlabel('Location (cm)')

    for induction_lap in xrange(len(context.induction_start_times)):
        if induction_lap == 0:
            start_time = context.down_t[0]
        else:
            start_time = context.induction_stop_times[induction_lap-1]
        if induction_lap == len(context.induction_start_times) - 1:
            stop_time = context.down_t[-1]
        else:
            stop_time = context.induction_start_times[induction_lap + 1]
        indexes = np.where((context.down_t >= start_time) & (context.down_t <= stop_time))

        next_weights = []
        for i, this_local_signal in enumerate(local_signals):
            this_pot_rate = np.trapz(np.multiply(pot_rate(this_local_signal[indexes]), global_signal[indexes]),
                                     dx=context.down_dt/1000.)
            this_depot_rate = np.trapz(np.multiply(depot_rate(this_local_signal[indexes]), global_signal[indexes]),
                                       dx=context.down_dt/1000.)
            this_delta_weight = context.rMC0 * this_pot_rate * (1. - current_weights[i]) - \
                                context.rCM0 * this_depot_rate * current_weights[i]
            this_delta_weight = max(min(this_delta_weight, 1. - current_weights[i]), - current_weights[i])
            next_weights.append(current_weights[i] + this_delta_weight)
        if plot:
            axes2[1].plot(context.peak_locs,
                          np.multiply(np.subtract(next_weights, current_weights), peak_weight),
                          label='Induction lap: %i' % (induction_lap + 1))
        current_weights = np.array(next_weights)
        current_delta_weights = np.subtract(np.multiply(current_weights, peak_weight), 1.)
        delta_weights_snapshots.append(current_delta_weights)
        current_ramp, discard_ramp_offset = get_model_ramp(current_delta_weights, allow_offset=allow_offset,
                                                           impose_offset=initial_ramp_offset)
        if plot:
            axes2[0].plot(context.binned_x, current_ramp)
        ramp_snapshots.append(current_ramp)

        current_residual_score = 0.
        for i in xrange(len(target_ramp)):
            current_residual_score += ((current_ramp[i] - target_ramp[i]) / context.target_range['residuals']) ** 2.

        if current_residual_score > 1.1 * prev_residual_score:
            if context.verbose > 0:
                print 'optimize_BTSP_E_CA1: calculate_model_ramp: pid: %i; aborting - residual score not ' \
                      'decreasing; induction: %i, lap: %i' % (os.getpid(), context.induction, induction_lap + 1)
            if plot:
                axes2[1].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
                clean_axes(axes)
                clean_axes(axes2)
                fig.tight_layout()
                fig2.tight_layout()
                fig.show()
                fig2.show()
            return dict()
        prev_residual_score = current_residual_score

    if plot:
        axes2[1].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
        clean_axes(axes)
        clean_axes(axes2)
        fig.tight_layout()
        fig2.tight_layout()
        fig.show()
        fig2.show()

    delta_weights = np.subtract(current_delta_weights, initial_delta_weights)
    initial_weights = np.multiply(initial_weights, peak_weight)
    final_weights = np.add(current_delta_weights, 1.)

    if context.induction == 1:
        initial_ramp_offset = None
        if 'before' in context.exp_ramp:
            allow_offset = False
        else:
            allow_offset = True

    model_ramp, discard_delta_weights, model_ramp_offset, model_residual_score = \
        get_residual_score(current_delta_weights, target_ramp, allow_offset=allow_offset,
                           impose_offset=initial_ramp_offset, full_output=True)
    if allow_offset and context.induction == 1:
        initial_ramp, discard_ramp_offset = subtract_baseline(initial_ramp, model_ramp_offset)
    
    result = {}
    result['residual_score'] = model_residual_score

    if context.cell_id in context.allow_offset_cell_ids and context.induction == 1:
        LSA_delta_weights = context.LSA_weights['after']
        if not np.all((context.min_delta_weight <= LSA_delta_weights) &
                      (LSA_delta_weights <= context.peak_delta_weight)):
            LSA_ramp, LSA_delta_weights, LSA_ramp_offset, LSA_residual_score = \
                get_delta_weights_LSA(context.exp_ramp['after'], context.input_rate_maps,
                                      initial_delta_weights=LSA_delta_weights,
                                      bounds=(context.min_delta_weight, context.peak_delta_weight),
                                      allow_offset=allow_offset, impose_offset=initial_ramp_offset,
                                      verbose=context.verbose)
            if context.verbose > 1:
                print 'Process: %i; re-computed LSA weights: cell_id: %i, after induction: %i, ramp_offset: %.3f' % \
                      (os.getpid(), context.cell_id, context.induction, LSA_ramp_offset)
        else:
            LSA_ramp, LSA_delta_weights, LSA_ramp_offset, LSA_residual_score = \
                get_residual_score(LSA_delta_weights, target_ramp, allow_offset=allow_offset,
                                   impose_offset=initial_ramp_offset, full_output=True)
        result['self_consistent_delta_residual_score'] = max(0., model_residual_score - LSA_residual_score)
    else:
        LSA_ramp = None

    ramp_amp, ramp_width, peak_shift, ratio, start_loc, peak_loc, end_loc, min_val, min_loc = {}, {}, {}, {}, {}, {}, \
                                                                                              {}, {}, {}

    ramp_amp['target'], ramp_width['target'], peak_shift['target'], ratio['target'], start_loc['target'], \
    peak_loc['target'], end_loc['target'], min_val['target'], min_loc['target'] = \
        calculate_ramp_features(context, target_ramp, context.mean_induction_start_loc)

    ramp_amp['model'], ramp_width['model'], peak_shift['model'], ratio['model'], start_loc['model'], \
    peak_loc['model'], end_loc['model'], min_val['model'], min_loc['model'] = \
        calculate_ramp_features(context, model_ramp, context.mean_induction_start_loc)

    if LSA_ramp is not None:
        ramp_amp['LSA'], ramp_width['LSA'], peak_shift['LSA'], ratio['LSA'], start_loc['LSA'], \
        peak_loc['LSA'], end_loc['LSA'], min_val['LSA'], min_loc['LSA'] = \
            calculate_ramp_features(context, LSA_ramp, context.mean_induction_start_loc)

    if context.verbose > 0:
        print 'Process: %i; cell: %i; induction: %i:' % (os.getpid(), context.cell_id, context.induction)
        print 'exp: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, peak_loc: %.1f, ' \
              'end_loc: %.1f, min_val: %.1f, min_loc: %.1f' % \
              (ramp_amp['target'], ramp_width['target'], peak_shift['target'], ratio['target'], start_loc['target'],
               peak_loc['target'], end_loc['target'], min_val['target'], min_loc['target'])
        if LSA_ramp is not None:
            print 'LSA: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, ' \
                  'peak_loc: %.1f, end_loc: %.1f, min_val: %.1f, min_loc: %.1f, ramp_offset: %.3f' % \
                  (ramp_amp['LSA'], ramp_width['LSA'], peak_shift['LSA'], ratio['LSA'], start_loc['LSA'],
                   peak_loc['LSA'], end_loc['LSA'], min_val['LSA'], min_loc['LSA'], LSA_ramp_offset)
        print 'model: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, peak_loc: %.1f' \
              ', end_loc: %.1f, min_val: %.1f, min_loc: %.1f, ramp_offset: %.3f' % \
              (ramp_amp['model'], ramp_width['model'], peak_shift['model'], ratio['model'], start_loc['model'],
               peak_loc['model'], end_loc['model'], min_val['model'], min_loc['model'], model_ramp_offset)
        sys.stdout.flush()

    result['delta_amp'] = ramp_amp['model'] - ramp_amp['target']
    result['delta_width'] = ramp_width['model'] - ramp_width['target']
    result['delta_peak_shift'] = peak_shift['model'] - peak_shift['target']
    result['delta_asymmetry'] = ratio['model'] - ratio['target']
    result['delta_min_val'] = min_val['model'] - min_val['target']

    abs_delta_min_loc = abs(min_loc['model'] - min_loc['target'])
    if min_loc['model'] <= min_loc['target']:
        if abs_delta_min_loc > context.track_length / 2.:
            delta_min_loc = context.track_length - abs_delta_min_loc
        else:
            delta_min_loc = -abs_delta_min_loc
    else:
        if abs_delta_min_loc > context.track_length / 2.:
            delta_min_loc = -(context.track_length - abs_delta_min_loc)
        else:
            delta_min_loc = abs_delta_min_loc
    result['delta_min_loc'] = delta_min_loc
    if plot:
        bar_loc = max(10., np.max(model_ramp) + 1., np.max(target_ramp) + 1.) * 0.95
        fig, axes = plt.subplots(2)
        axes[1].plot(context.peak_locs, delta_weights)
        axes[1].hlines(peak_weight * 1.05, xmin=context.mean_induction_start_loc, xmax=context.mean_induction_stop_loc)
        axes[0].plot(context.binned_x, target_ramp, label='Experiment')
        axes[0].plot(context.binned_x, model_ramp, label='Model')
        axes[0].hlines(bar_loc, xmin=context.mean_induction_start_loc, xmax=context.mean_induction_stop_loc)
        axes[1].set_ylabel('Change in\nsynaptic weight')
        axes[1].set_xlabel('Location (cm)')
        axes[0].set_ylabel('Subthreshold\ndepolarization (mV)')
        axes[0].set_xlabel('Location (cm)')
        axes[0].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
        axes[0].set_ylim([min(-1., np.min(model_ramp) - 1., np.min(target_ramp) - 1.),
                          max(10., np.max(model_ramp) + 1., np.max(target_ramp) + 1.)])
        axes[1].set_ylim([-peak_weight, peak_weight * 1.1])
        clean_axes(axes)
        fig.suptitle('Cell_id: %i, Induction: %i' % (context.cell_id, context.induction))
        fig.tight_layout()
        fig.show()

    local_peak_loc, local_peak_shift = {}, {}
    local_peak_loc['target'], local_peak_shift['target'] = \
        get_local_peak_shift(context, target_ramp, context.mean_induction_start_loc)
    local_peak_loc['model'], local_peak_shift['model'] = \
        get_local_peak_shift(context, model_ramp, context.mean_induction_start_loc)

    if export:
        with h5py.File(context.temp_output_path, 'a') as f:
            shared_context_key = 'shared_context'
            if shared_context_key not in f:
                f.create_group(shared_context_key)
                group = f[shared_context_key]
                group.create_dataset('peak_locs', compression='gzip', data=context.peak_locs)
                group.create_dataset('binned_x', compression='gzip', data=context.binned_x)
                group.create_dataset('signal_xrange', compression='gzip', data=signal_xrange)
                group.create_dataset('param_names', compression='gzip', data=context.param_names)
            exported_data_key = 'exported_data'
            if exported_data_key not in f:
                f.create_group(exported_data_key)
                f[exported_data_key].attrs['enumerated'] = False
            cell_key = str(context.cell_id)
            induction_key = str(context.induction)
            if cell_key not in f[exported_data_key]:
                f[exported_data_key].create_group(cell_key)
            if induction_key not in f[exported_data_key][cell_key]:
                f[exported_data_key][cell_key].create_group(induction_key)
            description = 'model_ramp_features'
            if description not in f[exported_data_key][cell_key][induction_key]:
                f[exported_data_key][cell_key][induction_key].create_group(description)
            group = f[exported_data_key][cell_key][induction_key][description]
            if 'before' in context.exp_ramp:
                group.create_dataset('initial_exp_ramp', compression='gzip', data=context.exp_ramp['before'])
            group.create_dataset('target_ramp', compression='gzip', data=target_ramp)
            group.create_dataset('initial_model_ramp', compression='gzip', data=initial_ramp)
            if LSA_ramp is not None:
                group.create_dataset('LSA_ramp', compression='gzip', data=LSA_ramp)
                group.create_dataset('LSA_weights', compression='gzip', data=np.add(LSA_delta_weights, 1.))
            group.create_dataset('model_ramp', compression='gzip', data=model_ramp)
            group.create_dataset('model_weights', compression='gzip', data=final_weights)
            group.create_dataset('initial_weights', compression='gzip', data=initial_weights)
            group.create_dataset('global_signal', compression='gzip', data=global_signal)
            group.create_dataset('down_t', compression='gzip', data=context.down_t)
            group.create_dataset('pot_rate', compression='gzip', data=pot_rate(signal_xrange))
            group.create_dataset('depot_rate', compression='gzip', data=depot_rate(signal_xrange))
            group.create_dataset('param_array', compression='gzip', data=context.x_array)
            group.create_dataset('local_signal_filter_t', compression='gzip', data=local_signal_filter_t)
            group.create_dataset('local_signal_filter', compression='gzip', data=local_signal_filter)
            group.create_dataset('global_filter_t', compression='gzip', data=global_filter_t)
            group.create_dataset('global_filter', compression='gzip', data=global_filter)
            group.attrs['local_signal_peak'] = local_signal_peak
            group.attrs['global_signal_peak'] = global_signal_peak
            group.attrs['mean_induction_start_loc'] = context.mean_induction_start_loc
            group.attrs['mean_induction_stop_loc'] = context.mean_induction_stop_loc
            group.attrs['induction_start_times'] = context.induction_start_times
            group.attrs['induction_stop_times'] = context.induction_stop_times
            group.attrs['track_start_times'] = context.track_start_times
            group.attrs['track_stop_times'] = context.track_stop_times
            group.attrs['target_ramp_amp'] = ramp_amp['target']
            group.attrs['target_ramp_width'] = ramp_width['target']
            group.attrs['target_peak_shift'] = peak_shift['target']
            group.attrs['target_local_peak_loc'] = local_peak_loc['target']
            group.attrs['target_local_peak_shift'] = local_peak_shift['target']
            group.attrs['target_ratio'] = ratio['target']
            group.attrs['target_start_loc'] = start_loc['target']
            group.attrs['target_peak_loc'] = peak_loc['target']
            group.attrs['target_end_loc'] = end_loc['target']
            group.attrs['target_min_val'] = min_val['target']
            group.attrs['target_min_loc'] = min_loc['target']
            group.attrs['model_ramp_amp'] = ramp_amp['model']
            group.attrs['model_ramp_width'] = ramp_width['model']
            group.attrs['model_peak_shift'] = peak_shift['model']
            group.attrs['model_local_peak_loc'] = local_peak_loc['model']
            group.attrs['model_local_peak_shift'] = local_peak_shift['model']
            group.attrs['model_ratio'] = ratio['model']
            group.attrs['model_start_loc'] = start_loc['model']
            group.attrs['model_peak_loc'] = peak_loc['model']
            group.attrs['model_end_loc'] = end_loc['model']
            group.attrs['model_min_val'] = min_val['model']
            group.attrs['model_min_loc'] = min_loc['model']
            group.create_group('ramp_snapshots')
            for i, current_ramp in enumerate(ramp_snapshots):
                group['ramp_snapshots'].create_dataset(str(i), data=current_ramp)
            group.create_group('delta_weights_snapshots')
            for i, this_delta_weights in enumerate(delta_weights_snapshots):
                group['delta_weights_snapshots'].create_dataset(str(i), data=this_delta_weights)

    return {context.cell_id: {context.induction: result}}


def plot_model_summary_figure(cell_id, model_file_path=None):
    """

    :param cell_id: int
    :param model_file_path: str (path)
    :return: dict
    """
    if (cell_id, 2) not in context.data_keys:
        raise KeyError('plot_model_summary_figure: cell_id: %i, induction: 2 not found' % cell_id)
    if model_file_path is None:
        raise IOError('plot_model_summary_figure: no model file path provided')
    elif not os.path.isfile(model_file_path):
        raise IOError('plot_model_summary_figure: invalid model file path: %s' % model_file_path)
    with h5py.File(model_file_path, 'r') as f:
        if 'exported_data' not in f or str(cell_id) not in f['exported_data'] or \
                '2' not in f['exported_data'][str(cell_id)] or \
                'model_ramp_features' not in f['exported_data'][str(cell_id)]['2']:
            raise KeyError('plot_model_summary_figure: problem loading model results for cell_id: %i, '
                           'induction 2; from file: %s' % (cell_id, model_file_path))
    with h5py.File(model_file_path, 'r') as f:
        group = f['exported_data'][str(cell_id)]['2']['model_ramp_features']
        x = group['param_array'][:]
        if 'local_signal_peak' not in group.attrs or 'global_signal_peak' not in group.attrs:
            raise KeyError('plot_model_summary_figure: missing required attributes for cell_id: %i, '
                           'induction 2; from file: %s' % (cell_id, model_file_path))
        group = f['exported_data'][str(cell_id)]['2']['model_ramp_features']
        local_signal_peak = group.attrs['local_signal_peak']
        global_signal_peak = group.attrs['global_signal_peak']
        local_signal_filter_t = group['local_signal_filter_t'][:]
        local_signal_filter = group['local_signal_filter'][:]
        global_filter_t = group['global_filter_t'][:]
        global_filter = group['global_filter'][:]
        initial_weights = group['initial_weights'][:]
        initial_ramp = group['initial_model_ramp'][:]
        model_ramp = group['model_ramp'][:]
        ramp_snapshots = []
        for lap in xrange(len(group['ramp_snapshots'])):
            ramp_snapshots.append(group['ramp_snapshots'][str(lap)][:])
        delta_weights_snapshots = []
        for lap in xrange(len(group['delta_weights_snapshots'])):
            delta_weights_snapshots.append(group['delta_weights_snapshots'][str(lap)][:])
        final_weights = group['model_weights'][:]

    import_data(cell_id, 2)
    update_source_contexts(x)

    global_signal = np.divide(get_global_signal(context.down_induction_gate, global_filter), global_signal_peak)
    local_signals = np.divide(get_local_signal_population(local_signal_filter), local_signal_peak)

    signal_xrange = np.linspace(0., 1., 10000)
    pot_rate = np.vectorize(scaled_single_sigmoid(context.rMC_th, context.rMC_peak + context.rMC_th, signal_xrange))
    depot_rate = np.vectorize(scaled_double_sigmoid(context.rCM_th1, context.rCM_th1 + context.rCM_peak1,
                                                    context.rCM_th2, context.rCM_th2 - context.rCM_peak2,
                                                    signal_xrange, y_end=context.rCM_min2))

    resolution = 10
    input_sample_indexes = np.arange(0, len(context.peak_locs), resolution)

    example_input_dict = {}

    sample_time_delays = []
    mean_induction_start_time_index = np.where(context.mean_position > context.mean_induction_start_loc)[0][0]
    mean_induction_start_time = context.mean_t[mean_induction_start_time_index]
    for index in input_sample_indexes:
        this_peak_loc = context.peak_locs[index]
        this_time_index = np.where(context.mean_position > this_peak_loc)[0][0]
        this_delay = context.mean_t[this_time_index] - mean_induction_start_time
        sample_time_delays.append(this_delay)
    sample_time_delays = np.abs(sample_time_delays)

    target_time_delay = 5000.

    relative_indexes = np.where((sample_time_delays > target_time_delay) &
                                (final_weights[input_sample_indexes] > initial_weights[input_sample_indexes]))[0]
    distant_potentiating_indexes = input_sample_indexes[relative_indexes]
    if np.any(distant_potentiating_indexes):
        relative_index = np.argmax(np.subtract(final_weights, initial_weights)[distant_potentiating_indexes])
        this_example_index = distant_potentiating_indexes[relative_index]
    else:
        relative_index = np.argmax(np.subtract(final_weights, initial_weights)[input_sample_indexes])
        this_example_index = input_sample_indexes[relative_index]
    example_input_dict['Potentiating input example'] = this_example_index

    relative_indexes = np.where((sample_time_delays > target_time_delay) &
                                (final_weights[input_sample_indexes] < initial_weights[input_sample_indexes]))[0]
    distant_depotentiating_indexes = input_sample_indexes[relative_indexes]
    if np.any(distant_depotentiating_indexes):
        relative_index = np.argmin(np.subtract(final_weights, initial_weights)[distant_depotentiating_indexes])
        this_example_index = distant_depotentiating_indexes[relative_index]
    else:
        relative_index = np.argmin(np.subtract(final_weights, initial_weights)[input_sample_indexes])
        this_example_index = input_sample_indexes[relative_index]
    example_input_dict['De-potentiating input example'] = this_example_index

    import matplotlib as mpl
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['font.size'] = 11.
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['axes.titlepad'] = 2.
    mpl.rcParams['mathtext.default'] = 'regular'
    from matplotlib.pyplot import cm
    import matplotlib.gridspec as gridspec

    fig, axes = plt.figure(figsize=(12, 8.5)), []
    gs0 = gridspec.GridSpec(3, 4, wspace=0.55, hspace=0.9, left=0.075, right=0.965, top=0.925, bottom=0.075)
    gs1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[1:, 1:], wspace=0.325, hspace=0.9)

    this_axis = fig.add_subplot(gs0[0, 0])
    axes.append(this_axis)
    ymax = 0.
    for color, label, ramp in zip(['k', 'c'], ['Before induction 2', 'After induction 2'],
                                  [initial_ramp, model_ramp]):
        this_axis.plot(context.binned_x, ramp, c=color, label=label)
        ymax = max(ymax, np.max(ramp))
    this_axis.set_ylabel('Ramp\namplitude (mV)')
    this_axis.set_xlabel('Position (cm)')
    ymax = math.ceil(ymax) + 1.
    this_axis.set_ylim(-1., ymax)
    this_axis.set_xlim(0., context.track_length)
    bar_loc = ymax - 0.5
    this_axis.hlines(bar_loc, xmin=context.mean_induction_start_loc, xmax=context.mean_induction_stop_loc)
    # this_axis.set_title('Bidirectional BTSP model', fontsize=mpl.rcParams['font.size'])
    this_axis.legend(loc=(0.05, 0.95), frameon=False, framealpha=0.5, handlelength=1, fontsize=mpl.rcParams['font.size'])
    this_axis.set_xticks(np.arange(0., context.track_length, 45.))

    this_axis = fig.add_subplot(gs0[0, 2])
    axes.append(this_axis)
    xmax = max(5000., local_signal_filter_t[-1], global_filter_t[-1]) / 1000.
    xmax = math.ceil(xmax)
    this_axis.plot(local_signal_filter_t / 1000., local_signal_filter / np.max(local_signal_filter), color='lightgray',
                   label='Synaptic\neligibility signal')
    this_axis.plot(global_filter_t / 1000., global_filter / np.max(global_filter), color='k',
                   label='Dendritic\ngating signal')
    this_axis.set_xlabel('Time (s)')
    this_axis.set_ylabel('Normalized amplitude')
    this_axis.set_ylim(0., this_axis.get_ylim()[1])
    this_axis.set_xlim(-0.5, xmax)
    this_axis.set_title('Plasticity signal kinetics', fontsize=mpl.rcParams['font.size'])
    this_axis.legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, fontsize=mpl.rcParams['font.size'])
    this_axis.set_xlim(-0.5, max(5000., local_signal_filter_t[-1], global_filter_t[-1]) / 1000.)

    this_axis = fig.add_subplot(gs0[0, 3])
    axes.append(this_axis)
    if context.rMC0 > context.rCM0:
        pot_scale = 1.
        depot_scale = context.rCM0 / context.rMC0
    else:
        pot_scale = context.rMC0 / context.rCM0
        depot_scale = 1.
    this_axis.plot(signal_xrange, pot_rate(signal_xrange) * pot_scale, label='Potentiation', c='r')
    this_axis.plot(signal_xrange, depot_rate(signal_xrange) * depot_scale, label='De-potentiation', c='c')
    this_axis.set_xlabel('Normalized eligibility signal')
    this_axis.set_ylabel('Normalized rate')
    this_axis.set_ylim(0., this_axis.get_ylim()[1])
    this_axis.set_xlim(0., 1.)
    this_axis.set_title('State transition rates', fontsize=mpl.rcParams['font.size'])
    this_axis.legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, fontsize=mpl.rcParams['font.size'])

    clean_axes(axes)

    colors = ['r', 'c']
    """
    axes1 = [[fig.add_subplot(gs1[row, col]) for col in xrange(2)] for row in xrange(2)]
    axes1[0][0].get_shared_x_axes().join(axes1[0][0], axes1[0][1], axes1[1][0], axes1[1][1])
    axes1[0][0].get_shared_y_axes().join(axes1[0][0], axes1[0][1])
    axes1[1][0].get_shared_y_axes().join(axes1[1][0], axes1[1][1])
    ymax1 = np.max(global_signal)
    ymax2 = 0.
    axes1_0_right = [axes1[0][0].twinx(), axes1[0][1].twinx()]
    for i, (name, index) in enumerate(example_input_dict.iteritems()):
        this_rate_map = context.complete_rate_maps[index]
        this_local_signal = local_signal_history[index]
        this_weight_dynamics = weight_dynamics_history[index]
        ymax1 = max(ymax1, np.max(this_local_signal))
        axes1[0][i].plot(context.complete_t / 1000., this_rate_map, c='grey', linewidth=1.,
                         label='Presynaptic firing rate', linestyle='--')
        axes1_0_right[i].plot(context.down_t / 1000., this_local_signal, c=colors[i], label='Synaptic eligibility signal')
        axes1_0_right[i].plot(context.down_t / 1000., global_signal, c='k', label='Dendritic gating signal', linewidth=0.75)
        axes1_0_right[i].set_title('%s:' % name, fontsize=mpl.rcParams['font.size'])
        axes1_0_right[i].fill_between(context.down_t / 1000., 0., np.minimum(this_local_signal, global_signal), alpha=0.5,
                                facecolor=colors[i], label='Signal overlap')
        axes1[1][i].plot(context.down_t / 1000., this_weight_dynamics, c=colors[i])
        ymax2 = max(ymax2, np.max(this_weight_dynamics))
        axes1[0][i].set_xlabel('Time (s)')
        for label in axes1[0][i].get_xticklabels():
            label.set_visible(True)
        axes1[1][i].set_xlabel('Time (s)')
        axes1_0_right[i].legend(loc=(0.4, 1.), frameon=False, framealpha=0.5, handlelength=1,
                           fontsize=mpl.rcParams['font.size'])
        axes1[0][i].legend(loc=(-0.2, 1.2), frameon=False, framealpha=0.5, handlelength=1,
                                fontsize=mpl.rcParams['font.size'])
        end = min(2, len(context.induction_start_times) - 1)
        axes1[0][i].set_xlim(-2., context.induction_start_times[end] / 1000. + 5.)
        axes1[1][i].set_xlim(-2., context.induction_start_times[end] / 1000. + 5.)
    ymax1_right = ymax1 / 0.9
    ymax1_left = context.input_field_peak_rate / 0.9
    ymax1 = max(ymax1_left, ymax1_right)
    ymax2 = math.ceil(ymax2 / 0.95)
    axes1[0][0].set_ylim([0., ymax1_left])
    axes1[0][1].set_ylim([0., ymax1_left])
    axes1[1][0].set_ylim([0., ymax2])
    axes1[1][1].set_ylim([0., ymax2])
    axes1_0_right[0].set_ylim([0., ymax1_right])
    axes1_0_right[1].set_ylim([0., ymax1_right])
    bar_loc0 = ymax1 * 0.95
    bar_loc1 = ymax2 * 0.95
    axes1[0][0].hlines([bar_loc0] * len(context.induction_start_times),
                   xmin=context.induction_start_times / 1000.,
                   xmax=context.induction_stop_times / 1000., linewidth=2)
    axes1[0][1].hlines([bar_loc0] * len(context.induction_start_times),
                      xmin=context.induction_start_times / 1000.,
                      xmax=context.induction_stop_times / 1000., linewidth=2)
    axes1[1][0].hlines([bar_loc1] * len(context.induction_start_times),
                      xmin=context.induction_start_times / 1000.,
                      xmax=context.induction_stop_times / 1000., linewidth=2)
    axes1[1][1].hlines([bar_loc1] * len(context.induction_start_times),
                      xmin=context.induction_start_times / 1000.,
                      xmax=context.induction_stop_times / 1000., linewidth=2)
    axes1[0][0].set_ylabel('Firing rate (Hz)')
    axes1[1][0].set_ylabel('Synaptic weight')
    axes1[0][0].set_yticks(np.arange(0., context.input_field_peak_rate + 1., 10.))
    axes1[0][1].set_yticks(np.arange(0., context.input_field_peak_rate + 1., 10.))
    axes1_0_right[0].set_ylabel('Plasticity signal amplitude', rotation=-90, labelpad=15)
    axes1_0_right[0].set_yticklabels([i * 0.2 for i in range(5)])
    axes1_0_right[1].set_yticklabels([i * 0.2 for i in range(5)])
    clean_twin_right_axes(axes1_0_right)
    clean_axes(np.array(axes1))
    """
    axes2 = []
    ymax = 0.
    for row, weights in zip(range(1, 3), [initial_weights, final_weights]):
        this_axis = fig.add_subplot(gs0[row, 0])
        axes2.append(this_axis)
        this_max_rate_map = np.zeros_like(context.input_rate_maps[0])
        for i in (index for index in input_sample_indexes if index not in example_input_dict.itervalues()):
            rate_map = np.array(context.input_rate_maps[i])
            rate_map *= weights[i] * context.ramp_scaling_factor
            ymax = max(ymax, np.max(rate_map))
            this_axis.plot(context.binned_x, rate_map, c='lightgray', zorder=0, linewidth=0.75)  # , alpha=0.5)
        for i, (name, index) in enumerate(example_input_dict.iteritems()):
            rate_map = np.array(context.input_rate_maps[index])
            rate_map *= weights[index] * context.ramp_scaling_factor
            ymax = max(ymax, np.max(rate_map))
            this_axis.plot(context.binned_x, rate_map, c=colors[i], zorder=1, label=name)
        this_axis.set_xlim(0., context.track_length)
        this_axis.set_ylabel('Input\namplitude (mV)')
        this_axis.set_xticks(np.arange(0., context.track_length, 45.))
        this_axis.set_xlabel('Position (cm)')
    axes2[0].legend(loc=(-0.1, 1.), frameon=False, framealpha=0.5, handlelength=1, fontsize=mpl.rcParams['font.size'])
    ymax = math.ceil(10. * ymax / 0.95) / 10.
    bar_loc = ymax * 0.95
    for this_axis in axes2:
        this_axis.set_ylim(0., ymax)
        this_axis.hlines(bar_loc, xmin=context.mean_induction_start_loc, xmax=context.mean_induction_stop_loc)
    clean_axes(axes2)
    fig.show()

    fig, axes = plt.subplots()
    cmap = cm.jet
    for w in np.linspace(0., 1., 10):
        net_delta_weight = pot_rate(signal_xrange) * pot_scale * (1. - w) - depot_rate(signal_xrange) * depot_scale * w
        axes.plot(signal_xrange, net_delta_weight, c=cmap(w))
    axes.set_xlabel('Normalized eligibility signal')
    axes.set_ylabel('Net change in synaptic weight')
    axes.set_title('Sigmoidal q_+, non-monotonic q_-')
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm)
    cbar.set_label('Initial synaptic weight', rotation=270., labelpad=15.)
    clean_axes(axes)
    fig.tight_layout()
    fig.show()

    fig2, axes2 = plt.subplots(1, 2, sharex=True)
    fig2.suptitle('Induction: %i' % context.induction)
    axes2[0].plot(context.binned_x, ramp_snapshots[0], c='k', label='Before')
    axes2[0].set_ylabel('Ramp amplitude (mV)')
    axes2[0].set_xlabel('Location (cm)')
    axes2[1].set_ylabel('Change in synaptic weight')
    axes2[1].set_xlabel('Location (cm)')
    for i in xrange(1, len(ramp_snapshots)):
        current_ramp = ramp_snapshots[i]
        current_delta_weights = np.subtract(delta_weights_snapshots[i], delta_weights_snapshots[i - 1])
        axes2[0].plot(context.binned_x, current_ramp)
        axes2[1].plot(context.peak_locs, current_delta_weights, label='Induction lap: %i' % i)
    axes2[1].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
    clean_axes(axes2)
    fig2.tight_layout()
    fig2.show()

    context.update(locals())


def get_args_dynamic_model_ramp(x, features):
    """
    A nested map operation is required to compute model_ramp features. The arguments to be mapped depend on each set of
    parameters and prior features (dynamic).
    :param x: array
    :param features: dict
    :return: list of list
    """
    group_size = len(context.data_keys)
    return [list(item) for item in zip(*context.data_keys)] + [[features['local_signal_max']] * group_size] + \
           [[features['global_signal_max']] * group_size]


def compute_features_model_ramp(x, cell_id=None, induction=None, local_signal_peak=None, global_signal_peak=None,
                                export=False, plot=False):
    """

    :param x: array
    :param cell_id: int
    :param induction: int
    :param local_signal_peak: float
    :param global_signal_peak: float
    :param export: bool
    :param plot: bool
    :return: dict
    """
    import_data(cell_id, induction)
    update_source_contexts(x, context)
    start_time = time.time()
    if context.disp:
        print 'Process: %i: computing model_ramp_features for cell_id: %i, induction: %i with x: %s' % \
              (os.getpid(), context.cell_id, context.induction, ', '.join('%.3E' % i for i in x))
    result = calculate_model_ramp(local_signal_peak=local_signal_peak, global_signal_peak=global_signal_peak,
                                  export=export, plot=plot)
    if context.disp:
        print 'Process: %i: computing model_ramp_features for cell_id: %i, induction: %i took %.1f s' % \
              (os.getpid(), context.cell_id, context.induction, time.time() - start_time)
    return result


def filter_features_model_ramp(primitives, current_features, export=False):
    """

    :param primitives: list of dict (each dict contains results from a single simulation)
    :param current_features: dict
    :param export: bool
    :return: dict
    """
    features = {}
    groups = ['spont', 'exp1', 'exp2']
    grouped_feature_names = ['delta_amp', 'delta_width', 'delta_peak_shift', 'delta_asymmetry', 'delta_min_loc',
                             'delta_min_val', 'residual_score']
    feature_names = ['self_consistent_delta_residual_score']
    for this_result_dict in primitives:
        if not this_result_dict:
            if context.verbose > 0:
                print 'optimize_BTSP_E_CA1: filter_features_model_ramp: pid: %i; model failed' % os.getpid()
            return dict()
        for cell_id in this_result_dict:
            cell_id = int(cell_id)
            for induction in this_result_dict[cell_id]:
                induction = int(induction)
                if cell_id in context.spont_cell_id_list:
                    group = 'spont'
                else:
                    group = 'exp' + str(induction)
                for feature_name in grouped_feature_names:
                    key = group + '_' + feature_name
                    if key not in features:
                        features[key] = []
                    features[key].append(this_result_dict[cell_id][induction][feature_name])
                for feature_name in feature_names:
                    if feature_name in this_result_dict[cell_id][induction]:
                        if feature_name not in features:
                            features[feature_name] = []
                        features[feature_name].append(this_result_dict[cell_id][induction][feature_name])

    for feature_name in grouped_feature_names:
        for group in groups:
            key = group + '_' + feature_name
            if key in features and len(features[key]) > 0:
                features[key] = np.mean(features[key])
            else:
                features[key] = 0.

    for feature_name in feature_names:
        if feature_name in features and len(features[feature_name]) > 0:
            features[feature_name] = np.mean(features[feature_name])
        else:
            features[feature_name] = 0.

    return features


def get_objectives(features, export=False):
    """

    :param features: dict
    :param export: bool
    :return: tuple of dict
    """
    objectives = {}

    grouped_feature_names = ['residual_score']
    groups = ['spont', 'exp1', 'exp2']
    for feature_name in grouped_feature_names:
        for group in groups:
            objective_name = group + '_' + feature_name
            if objective_name in features:
                objectives[objective_name] = features[objective_name]

    feature_names = ['self_consistent_delta_residual_score']
    for feature_name in feature_names:
        if feature_name in context.objective_names and feature_name in features:
            objectives[feature_name] = features[feature_name]

    for objective_name in context.objective_names:
        if objective_name not in objectives:
            objectives[objective_name] = 0.
        else:
            objectives[objective_name] = np.mean(objectives[objective_name])

    return features, objectives


def get_features_interactive(x, plot=False):
    """

    :param x:
    :param plot:
    :return: dict
    """
    features = {}
    args = get_args_static_signal_amplitudes()
    group_size = len(args[0])
    sequences = [[x] * group_size] + args + [[context.export] * group_size]  # + [[plot] * group_size]
    primitives = map(compute_features_signal_amplitudes, *sequences)
    new_features = filter_features_signal_amplitudes(primitives, features, context.export, plot)
    features.update(new_features)

    args = get_args_dynamic_model_ramp(x, features)
    group_size = len(args[0])
    sequences = [[x] * group_size] + args + [[context.export] * group_size] + [[plot] * group_size]
    primitives = map(compute_features_model_ramp, *sequences)
    new_features = filter_features_model_ramp(primitives, features, context.export)
    features.update(new_features)

    return features


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True,))
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_BTSP_E_CA1_cli_config.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--verbose", type=int, default=2)
@click.option("--plot", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--model-summary-figure", is_flag=True)
@click.option("--model-file-path", type=str, default=None)
@click.pass_context
def main(cli, config_file_path, output_dir, export, export_file_path, label, verbose, plot, debug,
         model_summary_figure, model_file_path):
    """
    Parallel optimization is meant to be executed by the module nested.optimize using the syntax:
    for N processes:
    mpirun -n N python -m nested.optimize --config-file-path=$PATH_TO_CONFIG_FILE --disp --export

    This script can be executed as main with the command line interface to debug the model code, and to generate plots
    after optimization has completed, e.g.:

    ipython
    run optimize_BTSP_E_CA1 --model-summary-figure --cell_id=1 --model-file-path=$PATH_TO_MODEL_FILE

    :param cli: contains unrecognized args as list of str
    :param config_file_path: str (path)
    :param output_dir: str (path)
    :param export: bool
    :param export_file_path: str
    :param label: str
    :param verbose: int
    :param plot: bool
    :param debug: bool
    :param model_summary_figure: bool
    :param model_file_path: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    kwargs = get_unknown_click_arg_dict(cli.args)
    context.disp = verbose > 0
    config_optimize_interactive(__file__, config_file_path=config_file_path, output_dir=output_dir, export=export,
                                export_file_path=export_file_path, label=label, disp=context.disp, verbose=verbose,
                                **kwargs)

    x1_array = context.x0_array
    if 'params_path' in context.kwargs and os.path.isfile(context.kwargs['params_path']):
        param_source_dict = read_from_yaml(context.kwargs['params_path'])
        if 'cell_id' in context.kwargs:
            if int(context.kwargs['cell_id']) in param_source_dict:
                x1_dict = param_source_dict[int(context.kwargs['cell_id'])]
                x1_array = param_dict_to_array(x1_dict, context.param_names)
            elif 'all' in param_source_dict:
                x1_dict = param_source_dict['all']
                x1_array = param_dict_to_array(x1_dict, context.param_names)
            else:
                print 'optimize_BTSP_E: problem loading params for cell_id: %s from params_path: %s' % \
                      (kwargs['params_path'], context.kwargs['cell_id'])
        elif 'all' in param_source_dict:
            x1_dict = param_source_dict['all']
            x1_array = param_dict_to_array(x1_dict, context.param_names)
        else:
            raise RuntimeError('optimize_BTSP_E: problem loading params from params_path: %s' %
                               context.kwargs['params_path'])

    if debug:
        features = get_features_interactive(x1_array, plot=plot)
        features, objectives = get_objectives(features, context.export)
        if export and os.path.isfile(context.temp_output_path):
            merge_exported_data([context.temp_output_path], context.export_file_path, True)
            os.remove(context.temp_output_path)
        print 'features:'
        pprint.pprint({key: val for (key, val) in features.iteritems() if key in context.feature_names})
        print 'objectives'
        pprint.pprint({key: val for (key, val) in objectives.iteritems() if key in context.objective_names})

    if model_summary_figure:
        plot_model_summary_figure(int(context.kwargs['cell_id']), model_file_path)

    context.update(locals())


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
