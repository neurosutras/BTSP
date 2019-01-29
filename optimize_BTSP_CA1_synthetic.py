"""
These methods aim to optimize a single parameterization of a model of bidirectional, state-dependent behavioral time 
scale synaptic plasticity to account for the width and amplitude of all place fields in an experimental data set from 
the Magee lab that includes: 
1) Silent cells converted into place cells by spontaneous plateaus
2) Silent cells converted into place cells by experimentally induced plateaus
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
"""
__author__ = 'milsteina'
from BTSP_utils import *
from collections import defaultdict
from nested.optimize_utils import *
import click


context = Context()


def config_worker():
    """

    """
    init_context()


def init_context():
    """

    """
    dt = 1.  # ms
    input_field_width = 90.  # cm
    input_field_peak_rate = 40.  # Hz
    num_inputs = 200  # 200
    track_length = 187.  # cm

    binned_dx = track_length / 100.  # cm
    binned_x = np.arange(0., track_length + binned_dx / 2., binned_dx)[:100] + binned_dx / 2.
    generic_dx = binned_dx / 100.  # cm
    generic_x = np.arange(0., track_length, generic_dx)

    default_run_vel = 25.  # cm/s
    generic_position_dt = generic_dx / default_run_vel * 1000.  # ms
    generic_t = np.arange(0., len(generic_x) * generic_position_dt, generic_position_dt)[:len(generic_x)]

    default_interp_t = np.arange(0., generic_t[-1], dt)
    default_interp_x = np.interp(default_interp_t, generic_t, generic_x)
    default_interp_dx = dt * default_run_vel / 1000.  # cm

    extended_x = np.concatenate([generic_x - track_length, generic_x, generic_x + track_length])

    input_rate_maps, peak_locs = \
        generate_spatial_rate_maps(binned_x, num_inputs, input_field_peak_rate, input_field_width, track_length)

    down_dt = 10.  # ms, to speed up optimization
    sm = StateMachine(dt=down_dt)
    num_induction_laps = 5
    induction_dur = 300.  # ms

    context.update(locals())

    complete_induction_locs = {}
    complete_induction_durs = {}
    induction_stop_locs = {}
    for induction in context.induction_locs:
        complete_induction_locs[induction] = [context.induction_locs[induction] for i in xrange(num_induction_laps)]
        complete_induction_durs[induction] = [induction_dur for i in xrange(num_induction_laps)]
        induction_stop_index = np.where(default_interp_x >= context.induction_locs[induction])[0][0] + \
                               int(induction_dur / dt)
        induction_stop_locs[induction] = default_interp_x[induction_stop_index]

    position = {key: [] for key in ['pre', 'induction', 'post']}
    t = {key: [] for key in ['pre', 'induction', 'post']}
    current = {induction: [] for induction in context.induction_locs}

    position['pre'].append(default_interp_x)
    t['pre'].append(default_interp_t)
    for i in xrange(num_induction_laps):
        position['induction'].append(default_interp_x)
        t['induction'].append(default_interp_t)
    position['post'].append(default_interp_x)
    t['post'].append(default_interp_t)

    for induction in context.induction_locs:
        for i in xrange(num_induction_laps):
            start_index = np.where(default_interp_x >= complete_induction_locs[induction][i])[0][0]
            stop_index = np.where(default_interp_t >= default_interp_t[start_index] +
                                  complete_induction_durs[induction][i])[0][0]
            this_current = np.zeros_like(default_interp_t)
            this_current[start_index:stop_index] = 1.
            current[induction].append(this_current)

    complete_run_vel = np.array([])
    complete_position = np.array([])
    complete_t = np.array([])
    running_dur = 0.
    running_length = 0.
    
    for group in (group for group in ['pre', 'induction', 'post'] if group in position):
        for this_position, this_t in zip(position[group], t[group]):
            complete_position = np.append(complete_position, np.add(this_position, running_length))
            complete_t = np.append(complete_t, np.add(this_t, running_dur))
            running_length += track_length
            running_dur += len(this_t) * dt
    
    complete_run_vel = np.full_like(complete_t, default_run_vel)
    complete_run_vel_gate = np.ones_like(complete_run_vel)
    complete_run_vel_gate[np.where(complete_run_vel <= 5.)[0]] = 0.
    
    # pre lap
    induction_gate = {induction: np.zeros_like(default_interp_t) for induction in context.induction_locs}
    for induction in context.induction_locs:
        for this_current in current[induction]:
            induction_gate[induction] = np.append(induction_gate[induction], this_current)
        # post lap
        induction_gate[induction] = np.append(induction_gate[induction], np.zeros_like(default_interp_t))

    induction_start_indexes = {induction: [] for induction in context.induction_locs}
    induction_stop_indexes = {induction: [] for induction in context.induction_locs}
    induction_start_times = {induction: [] for induction in context.induction_locs}
    induction_stop_times = {induction: [] for induction in context.induction_locs}
    running_position = 0.
    running_t = 0.
    for i in xrange(num_induction_laps):
        for induction in context.induction_locs:
            this_induction_start_index = np.where(complete_position >= context.induction_locs[induction] + 
                                                  running_position)[0][0]
            this_induction_start_time = complete_t[this_induction_start_index]
            this_induction_stop_time = this_induction_start_time + induction_dur
            this_induction_stop_index = np.where(complete_t >= this_induction_stop_time)[0][0]
            running_t += len(t['induction'][i]) * dt
            induction_start_times[induction].append(this_induction_start_time)
            induction_stop_times[induction].append(this_induction_stop_time)
            induction_start_indexes[induction].append(this_induction_start_index)
            induction_stop_indexes[induction].append(this_induction_stop_index)
        running_position += track_length
    for induction in context.induction_locs:
        induction_start_times[induction] = np.array(induction_start_times[induction])
        induction_stop_times[induction] = np.array(induction_stop_times[induction])
        induction_start_indexes[induction] = np.array(induction_start_indexes[induction])
        induction_stop_indexes[induction] = np.array(induction_stop_indexes[induction])
    complete_rate_maps = get_complete_rate_maps(input_rate_maps, binned_x, position, complete_run_vel_gate)
    down_t = np.arange(complete_t[0], complete_t[-1] + down_dt / 2., down_dt)
    down_induction_gate = {induction: np.interp(down_t, complete_t, induction_gate[induction])
                           for induction in context.induction_locs}

    context.update(locals())

    context.initial_delta_weights, context.initial_ramp = {}, {}
    context.target_ramp = get_target_synthetic_ramp(context.induction_locs['1'],
                                                    target_peak_shift=context.target_val['peak_shift_1'],
                                                    target_peak_val=context.target_val['peak_val_1'])
    context.initial_delta_weights['1'] = np.zeros_like(peak_locs)
    context.initial_ramp['1'] = np.zeros_like(binned_x)
    if 'plot' not in context():
        context.plot = False
    context.initial_ramp['2_null'], context.initial_delta_weights['2_null'], ramp_offset, residual_score = \
        get_delta_weights_LSA(context.target_ramp, input_rate_maps, context.induction_locs['1'],
                              induction_stop_locs['1'], bounds=(context.min_delta_weight, 3.), verbose=context.verbose,
                              plot=context.plot)
    context.initial_ramp['2_shift'] = context.initial_ramp['2_null']
    context.initial_delta_weights['2_shift'] = context.initial_delta_weights['2_null']


def update_model_params(x, local_context):
    """

    :param x: array
    :param local_context: :class:'Context'
    """
    if local_context is None:
        local_context = context
    local_context.update(param_array_to_dict(x, local_context.param_names))


def get_complete_rate_maps(input_rate_maps, input_x, position, run_vel_gate):
    """
    :param input_rate_maps: array
    :param input_x: array (x resolution of input)
    :param position: dict
    :param run_vel_gate: array
    :return: list of array
    """
    complete_rate_maps = []
    for j in xrange(len(input_rate_maps)):
        this_complete_rate_map = np.array([])
        for group in ['pre', 'induction', 'post']:
            for i, this_position in enumerate(position[group]):
                this_rate_map = np.interp(this_position, input_x, input_rate_maps[j])
                this_complete_rate_map = np.append(this_complete_rate_map, this_rate_map)
        this_complete_rate_map = np.multiply(this_complete_rate_map, run_vel_gate)
        if len(this_complete_rate_map) != len(run_vel_gate):
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
    for i in xrange(len(context.peak_locs)):
        rate_map = np.interp(context.down_t, context.complete_t, context.complete_rate_maps[i])
        local_signals.append(get_local_signal(rate_map, local_filter, context.down_dt))
    return local_signals


def get_ramp_residual_score(delta_weights, target_ramp, input_matrix, induction_loc, ramp_x=None, input_x=None,
                         bounds=None, allow_offset=False, impose_offset=None, disp=False, full_output=False):
    """

    :param delta_weights: array
    :param target_ramp: array
    :param input_matrix: array
    :param induction_loc
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
                raise Exception('get_ramp_residual_score: input out of bounds; cannot return full_output')
            return 1e9
    if ramp_x is None:
        ramp_x = context.binned_x
    if input_x is None:
        input_x = context.binned_x
    if len(target_ramp) != len(input_x):
        interp_target_ramp = np.interp(input_x, ramp_x, target_ramp)
    else:
        interp_target_ramp = np.array(target_ramp)

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
        calculate_ramp_features(context, interp_target_ramp, induction_loc)

    ramp_amp['model'], ramp_width['model'], peak_shift['model'], ratio['model'], start_loc['model'], \
    peak_loc['model'], end_loc['model'], min_val['model'], min_loc['model'] = \
        calculate_ramp_features(context, model_ramp, induction_loc)

    if disp:
        print 'target: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, peak_loc: %.1f, ' \
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

    for i in xrange(len(interp_target_ramp)):
        Err += ((interp_target_ramp[i] - model_ramp[i]) / context.target_range['residuals']) ** 2.
    # regularization
    for delta in np.diff(np.insert(delta_weights, 0, delta_weights[-1])):
        Err += (delta / context.target_range['weights_smoothness']) ** 2.

    if full_output:
        return model_ramp, delta_weights, ramp_offset, Err
    else:
        return Err


def get_ramp_feature_score(delta_weights, initial_ramp, input_matrix, induction_loc, ramp_x=None, input_x=None,
                           verbose=1):
    """

    :param delta_weights: array
    :param initial_ramp: array
    :param input_matrix: array
    :param induction_loc
    :param ramp_x: array
    :param input_x: array
    :param verbose: int
    :return: float
    """
    if ramp_x is None:
        ramp_x = context.binned_x
    if input_x is None:
        input_x = context.binned_x
    if len(initial_ramp) != len(input_x):
        interp_initial_ramp = np.interp(input_x, ramp_x, initial_ramp)
    else:
        interp_initial_ramp = np.array(initial_ramp)
    
    model_ramp = delta_weights.dot(input_matrix)
    if len(model_ramp) != len(ramp_x):
        model_ramp = np.interp(ramp_x, input_x, model_ramp)
    
    Err = 0.
    
    ramp_amp, ramp_width, peak_shift, ratio, start_loc, peak_loc, end_loc, min_val, min_loc = {}, {}, {}, {}, {}, {}, \
                                                                                              {}, {}, {}
    ramp_amp['before'], ramp_width['before'], peak_shift['before'], ratio['before'], start_loc['before'], \
    peak_loc['before'], end_loc['before'], min_val['before'], min_loc['before'] = \
        calculate_ramp_features(context, interp_initial_ramp, induction_loc)

    ramp_amp['after'], ramp_width['after'], peak_shift['after'], ratio['after'], start_loc['after'], \
    peak_loc['after'], end_loc['after'], min_val['after'], min_loc['after'] = \
        calculate_ramp_features(context, model_ramp, induction_loc)

    start_index, peak_index, end_index, min_index = \
        get_indexes_from_ramp_bounds_with_wrap(ramp_x, start_loc['before'], peak_loc['before'], end_loc['before'],
                                               min_loc['before'])

    # Err += ((min_val['after'] - context.target_val['min_val_2']) / context.target_range['min_val']) ** 2.
    model_peak_val_initial = model_ramp[peak_index]
    delta_peak_val_initial = model_peak_val_initial - ramp_amp['before']
    Err += ((delta_peak_val_initial - context.target_val['delta_peak_val_1']) /
            context.target_range['delta_peak_val']) ** 2.
    Err += ((peak_shift['after'] - context.target_val['peak_shift_2']) / context.target_range['peak_loc']) ** 2.
    Err += ((ramp_amp['after'] - context.target_val['peak_shift_2']) / context.target_range['peak_loc']) ** 2.

    # regularization
    for delta in np.diff(np.insert(delta_weights, 0, delta_weights[-1])):
        Err += (delta / context.target_range['weights_smoothness']) ** 2.

    if verbose > 1:
        print 'before: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, ' \
              'peak_loc: %.1f, end_loc: %.1f, min_val: %.1f, min_loc: %.1f' % \
              (ramp_amp['before'], ramp_width['before'], peak_shift['before'], ratio['before'],
               start_loc['before'], peak_loc['before'], end_loc['before'], min_val['before'], min_loc['before'])
        print 'after: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, peak_loc: %.1f' \
              ', end_loc: %.1f, min_val: %.1f, min_loc: %.1f, peak_val_initial: %.1f' % \
              (ramp_amp['after'], ramp_width['after'], peak_shift['after'], ratio['after'], start_loc['after'],
               peak_loc['after'], end_loc['after'], min_val['after'], min_loc['after'], model_peak_val_initial)
        sys.stdout.flush()

    return model_ramp, delta_weights, delta_peak_val_initial, Err


def get_delta_weights_LSA(target_ramp, input_rate_maps, induction_loc, induction_stop_loc, initial_delta_weights=None,
                          bounds=None, beta=2., ramp_x=None, input_x=None, allow_offset=False, impose_offset=None, 
                          plot=False, verbose=1):
    """
    Uses least square approximation to estimate a set of weights to match any arbitrary place field ramp, agnostic
    about underlying kernel, induction velocity, etc.
    :param target_ramp: dict of array
    :param input_rate_maps: array; x=default_interp_x
    :param induction_loc: float
    :param induction_stop_loc: float
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
    start_time = time.time()
    if ramp_x is None:
        ramp_x = context.binned_x
    if input_x is None:
        input_x = context.binned_x
    if len(target_ramp) != len(input_x):
        interp_target_ramp = np.interp(input_x, ramp_x, target_ramp)
    else:
        interp_target_ramp = np.array(target_ramp)

    input_matrix = np.multiply(input_rate_maps, context.ramp_scaling_factor)
    if initial_delta_weights is None:
        [U, s, Vh] = np.linalg.svd(input_matrix)
        V = Vh.T
        D = np.zeros_like(input_matrix)
        D[np.where(np.eye(*D.shape))] = s / (s ** 2. + beta ** 2.)
        input_matrix_inv = V.dot(D.conj().T).dot(U.conj().T)
        initial_delta_weights = interp_target_ramp.dot(input_matrix_inv)
    initial_ramp = initial_delta_weights.dot(input_matrix)
    if bounds is None:
        bounds = (0., 3.)
    result = minimize(get_ramp_residual_score, initial_delta_weights,
                      args=(target_ramp, input_matrix, induction_loc, ramp_x, input_x, bounds, allow_offset,
                            impose_offset),
                      method='L-BFGS-B', bounds=[bounds] * len(initial_delta_weights),
                      options={'disp': verbose > 1, 'maxiter': 100})

    if verbose > 1:
        print 'get_delta_weights_LSA: process: %i took %.2f s' % (os.getpid(), time.time() - start_time)
    model_ramp, delta_weights, ramp_offset, residual_score = \
        get_ramp_residual_score(result.x, target_ramp, input_matrix, induction_loc, ramp_x, input_x, bounds,
                                allow_offset, impose_offset, disp=verbose > 1, full_output=True)

    if plot:
        x_start = induction_loc
        x_end = induction_stop_loc
        ylim = max(np.max(target_ramp), np.max(model_ramp))
        ymin = min(np.min(target_ramp), np.min(model_ramp))
        fig, axes = plt.subplots(1)
        axes.plot(ramp_x, target_ramp, label='Target', color='k')
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


def calculate_model_ramp(induction, export=False, plot=False):
    """

    :param induction: str
    :param export: bool
    :param plot: bool
    :return: dict
    """
    local_signal_filter_t, local_signal_filter, global_filter_t, global_filter = \
        get_signal_filters(context.local_signal_rise, context.local_signal_decay, context.global_signal_rise,
                           context.global_signal_decay, context.down_dt, plot)

    global_signal = get_global_signal(context.down_induction_gate[induction], global_filter)
    local_signals = get_local_signal_population(local_signal_filter)
    local_signal_peaks = [np.max(local_signal) for local_signal in local_signals]

    if plot:
        fig, axes = plt.subplots(1)
        hist, edges = np.histogram(local_signal_peaks, density=True)
        bin_width = edges[1] - edges[0]
        axes.plot(edges[:-1] + bin_width / 2., hist * bin_width, c='r', label='Local plasticity signals')
        axes.set_xlabel('Peak local plasticity signal amplitudes (a.u.)')
        axes.set_ylabel('Probability')
        axes.set_title('Local signal amplitude distribution')
        axes.legend(loc='best', frameon=False, framealpha=0.5)
        clean_axes(axes)
        fig.tight_layout()
        fig.show()

    global_signal_peak = np.max(global_signal)
    local_signal_peak = np.max(local_signal_peaks)
    global_signal /= global_signal_peak
    local_signals /= local_signal_peak

    signal_xrange = np.linspace(0., 1., 10000)
    pot_rate = np.vectorize(scaled_single_sigmoid(context.rMC_th, context.rMC_peak, signal_xrange))
    depot_rate = np.vectorize(scaled_double_sigmoid_orig(context.rCM_th1, context.rCM_peak1, context.rCM_th2,
                                                    context.rCM_peak2, signal_xrange, y_end=context.rCM_min2))
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

    initial_delta_weights = context.initial_delta_weights[induction]
    initial_ramp = context.initial_ramp[induction]
    induction_loc = context.induction_locs[induction]
    induction_stop_loc = context.induction_stop_locs[induction]

    # re-compute initial weights if they are out of the current weight bounds
    if induction == 2:
        if not np.all((initial_delta_weights >= context.min_delta_weight) &
                      (initial_delta_weights <= context.peak_delta_weight)):
            initial_delta_weights = np.minimum(np.maximum(initial_delta_weights, context.min_delta_weight),
                                               context.peak_delta_weight)
            initial_ramp, initial_delta_weights, initial_ramp_offset, discard_residual_score = \
                get_delta_weights_LSA(context.target_ramp, context.input_rate_maps, induction_loc,
                                      induction_stop_loc, initial_delta_weights=initial_delta_weights,
                                      bounds=(context.min_delta_weight, context.peak_delta_weight),
                                      verbose=context.verbose)
            if context.verbose > 1:
                print 'Process: %i; re-computed initial weights before induction: %s' % (os.getpid(), induction)

    initial_weights = np.divide(np.add(initial_delta_weights, 1.), peak_weight)
    weights = []
    for i in xrange(len(context.peak_locs)):
        # normalize total number of receptors
        initial_weight = initial_weights[i]
        available = 1. - initial_weight
        context.sm.update_states({'M': available, 'C': initial_weight})
        local_signal = local_signals[i]
        context.sm.update_rates(
            {'M': {'C': context.rMC0 * np.multiply(pot_rate(local_signal), global_signal)},
             'C': {'M': context.rCM0 * np.multiply(depot_rate(local_signal), global_signal)}})
        context.sm.reset()
        context.sm.run()
        if i == 100:
            example_weight_dynamics = np.array(context.sm.states_history['C'][:-1]) * peak_weight
            example_local_signal = np.array(local_signal)
            if plot:
                fig, axes = plt.subplots(2, sharex=True)
                ymax0 = max(np.max(local_signal), np.max(global_signal))
                bar_loc0 = ymax0 * 1.05
                axes[0].plot(context.down_t / 1000., example_local_signal, c='r', label='Local plasticity signal')
                axes[0].plot(context.down_t / 1000., global_signal, c='k', label='Global signal')
                axes[0].set_ylim([-0.1 * ymax0, 1.1 * ymax0])
                axes[0].hlines([bar_loc0] * len(context.induction_start_times[induction]),
                               xmin=context.induction_start_times[induction] / 1000.,
                               xmax=context.induction_stop_times[induction] / 1000., linewidth=2)
                axes[0].set_xlabel('Time (s)')
                axes[0].set_ylabel('Plasticity\nsignal amplitudes')
                axes[0].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
                axes[1].plot(context.down_t / 1000., example_weight_dynamics)
                axes[1].set_ylim([0., peak_weight * 1.1])
                axes[1].hlines([peak_weight * 1.05] * len(context.induction_start_times[induction]),
                               xmin=context.induction_start_times[induction] / 1000.,
                               xmax=context.induction_stop_times[induction] / 1000., linewidth=2)
                axes[1].set_ylabel('Synaptic weight\n(example\nsingle input)')
                axes[1].set_xlabel('Time (s)')
                clean_axes(axes)
                fig.tight_layout(h_pad=2.)
                fig.show()
        weights.append(context.sm.states['C'] * peak_weight)
    initial_weights = np.multiply(initial_weights, peak_weight)
    weights = np.array(weights)
    delta_weights = np.subtract(weights, initial_weights)

    input_matrix = np.multiply(context.input_rate_maps, context.ramp_scaling_factor)
    result = {}
    ramp_amp, ramp_width, peak_shift, ratio, start_loc, peak_loc, end_loc, min_val, min_loc = {}, {}, {}, {}, {}, {}, \
                                                                                              {}, {}, {}
    target_ramp = context.target_ramp
    if induction in ['1', '2_null']:
        model_ramp, discard_delta_weights, model_ramp_offset, model_ramp_residual_score = \
            get_ramp_residual_score(np.subtract(weights, 1.), target_ramp, input_matrix, induction_loc,
                                    full_output=True)
        result['ramp_residual_score'] = model_ramp_residual_score

        ramp_amp['target'], ramp_width['target'], peak_shift['target'], ratio['target'], start_loc['target'], \
        peak_loc['target'], end_loc['target'], min_val['target'], min_loc['target'] = \
            calculate_ramp_features(context, target_ramp, induction_loc)
    else:
        model_ramp, discard_delta_weights, delta_peak_val_initial, model_ramp_feature_score = \
            get_ramp_feature_score(np.subtract(weights, 1.), initial_ramp, input_matrix, induction_loc,
                                   verbose=context.verbose)
        result['ramp_feature_score'] = model_ramp_feature_score
    if induction in ['2_null', '2_shift']:
        ramp_amp['before'], ramp_width['before'], peak_shift['before'], ratio['before'], start_loc['before'], \
        peak_loc['before'], end_loc['before'], min_val['before'], min_loc['before'] = \
            calculate_ramp_features(context, initial_ramp, induction_loc)
    
    ramp_amp['after'], ramp_width['after'], peak_shift['after'], ratio['after'], start_loc['after'], \
    peak_loc['after'], end_loc['after'], min_val['after'], min_loc['after'] = \
        calculate_ramp_features(context, model_ramp, induction_loc)
    
    if context.verbose > 0:
        print 'Process: %i; induction: %s:' % (os.getpid(), induction)
        if induction in ['1', '2_null']:
            print 'target: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, ' \
                  'peak_loc: %.1f, end_loc: %.1f, min_val: %.1f, min_loc: %.1f' % \
                  (ramp_amp['target'], ramp_width['target'], peak_shift['target'], ratio['target'], 
                   start_loc['target'], peak_loc['target'], end_loc['target'], min_val['target'], min_loc['target'])
        if induction in ['2_null', '2_shift']:
            print 'before: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, ' \
                  'peak_loc: %.1f, end_loc: %.1f, min_val: %.1f, min_loc: %.1f' % \
                  (ramp_amp['before'], ramp_width['before'], peak_shift['before'], ratio['before'],
                   start_loc['before'], peak_loc['before'], end_loc['before'], min_val['before'], min_loc['before'])
        print 'after: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, peak_loc: %.1f' \
              ', end_loc: %.1f, min_val: %.1f, min_loc: %.1f' % \
              (ramp_amp['after'], ramp_width['after'], peak_shift['after'], ratio['after'], start_loc['after'],
               peak_loc['after'], end_loc['after'], min_val['after'], min_loc['after'])
        sys.stdout.flush()

    result['peak_val'] = ramp_amp['after']
    result['peak_shift'] = peak_shift['after']
    result['min_val'] = min_val['after']
    result['ramp_width'] = ramp_width['after']
    if induction == '2_shift':
        result['delta_peak_val_initial'] = delta_peak_val_initial
    
    induction_stop_loc = context.induction_stop_locs[induction]

    if plot:
        bar_loc = max(10., np.max(model_ramp) + 1., np.max(target_ramp) + 1., np.max(initial_ramp) + 1. ) * 0.95
        fig, axes = plt.subplots(2)
        axes[1].plot(context.peak_locs, delta_weights)
        axes[1].hlines(peak_weight * 1.05, xmin=induction_loc, xmax=induction_stop_loc)
        if induction in ['1', '2_null']:
            axes[0].plot(context.binned_x, target_ramp, label='Target')
        axes[0].plot(context.binned_x, initial_ramp, label='Before')
        axes[0].plot(context.binned_x, model_ramp, label='After')
        axes[0].hlines(bar_loc, xmin=induction_loc, xmax=induction_stop_loc)
        axes[1].set_ylabel('Change in\nsynaptic weight')
        axes[1].set_xlabel('Location (cm)')
        axes[0].set_ylabel('Subthreshold\ndepolarization (mV)')
        axes[0].set_xlabel('Location (cm)')
        axes[0].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
        axes[0].set_ylim([min(-1., np.min(model_ramp) - 1., np.min(target_ramp) - 1., np.min(initial_ramp) - 1.),
                          max(10., np.max(model_ramp) + 1., np.max(target_ramp) + 1., np.max(initial_ramp) + 1.)])
        axes[1].set_ylim([-peak_weight, peak_weight * 1.1])
        clean_axes(axes)
        fig.suptitle('Induction: %s' % induction)
        fig.tight_layout()
        fig.show()
    
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
            if induction not in f[exported_data_key]:
                f[exported_data_key].create_group(induction)
            description = 'model_ramp_features'
            if description not in f[exported_data_key][induction]:
                f[exported_data_key][induction].create_group(description)
            group = f[exported_data_key][induction][description]
            if induction in ['1', '2_null']:
                group.create_dataset('target_ramp', compression='gzip', data=target_ramp)
            group.create_dataset('initial_model_ramp', compression='gzip', data=initial_ramp)
            group.create_dataset('model_ramp', compression='gzip', data=model_ramp)
            group.create_dataset('model_weights', compression='gzip', data=weights)
            group.create_dataset('initial_weights', compression='gzip', data=initial_weights)
            group.create_dataset('example_local_signal', compression='gzip', data=example_local_signal)
            group.create_dataset('global_signal', compression='gzip', data=global_signal)
            group.create_dataset('down_t', compression='gzip', data=context.down_t)
            group.create_dataset('example_weight_dynamics', compression='gzip', data=example_weight_dynamics)
            group.create_dataset('pot_rate', compression='gzip', data=pot_rate(signal_xrange))
            group.create_dataset('depot_rate', compression='gzip', data=depot_rate(signal_xrange))
            group.create_dataset('param_array', compression='gzip', data=context.x_array)
            group.create_dataset('local_signal_filter_t', compression='gzip', data=local_signal_filter_t)
            group.create_dataset('local_signal_filter', compression='gzip', data=local_signal_filter)
            group.create_dataset('global_filter_t', compression='gzip', data=global_filter_t)
            group.create_dataset('global_filter', compression='gzip', data=global_filter)
            group.attrs['local_signal_peak'] = local_signal_peak
            group.attrs['global_signal_peak'] = global_signal_peak
            group.attrs['induction_start_loc'] = induction_loc
            group.attrs['induction_stop_loc'] = induction_stop_loc
            group.attrs['induction_start_times'] = context.induction_start_times[induction]
            group.attrs['induction_stop_times'] = context.induction_stop_times[induction]
            if induction in ['1', '2_null']:
                group.attrs['target_ramp_amp'] = ramp_amp['target']
                group.attrs['target_ramp_width'] = ramp_width['target']
                group.attrs['target_peak_shift'] = peak_shift['target']
                group.attrs['target_ratio'] = ratio['target']
                group.attrs['target_start_loc'] = start_loc['target']
                group.attrs['target_peak_loc'] = peak_loc['target']
                group.attrs['target_end_loc'] = end_loc['target']
                group.attrs['target_min_val'] = min_val['target']
                group.attrs['target_min_loc'] = min_loc['target']
            if induction in ['2_null', '2_shift']:
                group.attrs['initial_ramp_amp'] = ramp_amp['before']
                group.attrs['initial_ramp_width'] = ramp_width['before']
                group.attrs['initial_peak_shift'] = peak_shift['before']
                group.attrs['initial_ratio'] = ratio['before']
                group.attrs['initial_start_loc'] = start_loc['before']
                group.attrs['initial_peak_loc'] = peak_loc['before']
                group.attrs['initial_end_loc'] = end_loc['before']
                group.attrs['initial_min_val'] = min_val['before']
                group.attrs['initial_min_loc'] = min_loc['before']
                if induction == '2_shift':
                    group.attrs['delta_peak_val_initial'] = delta_peak_val_initial
            group.attrs['model_ramp_amp'] = ramp_amp['after']
            group.attrs['model_ramp_width'] = ramp_width['after']
            group.attrs['model_peak_shift'] = peak_shift['after']
            group.attrs['model_ratio'] = ratio['after']
            group.attrs['model_start_loc'] = start_loc['after']
            group.attrs['model_peak_loc'] = peak_loc['after']
            group.attrs['model_end_loc'] = end_loc['after']
            group.attrs['model_min_val'] = min_val['after']
            group.attrs['model_min_loc'] = min_loc['after']
    
    return {induction: result}


def get_args_static_model_ramp():
    """
    A nested map operation is required to compute model_ramp features. The arguments to be mapped are the same
    (static) for each set of parameters.
    :param x: array
    :param features: dict
    :return: list of list
    """

    return [context.induction_locs.keys()]


def compute_features_model_ramp(x, induction=None, export=False, plot=False):
    """

    :param x: array
    :param induction: str
    :param export: bool
    :param plot: bool
    :return: dict
    """
    update_source_contexts(x, context)
    start_time = time.time()
    if context.disp:
        print 'Process: %i: computing model_ramp features for induction: %s with x: %s' % \
              (os.getpid(), induction, ', '.join('%.3E' % i for i in x))
    result = calculate_model_ramp(induction, export=export, plot=plot)
    if context.disp:
        print 'Process: %i: computing model_ramp features for induction: %s took %.1f s' % \
              (os.getpid(), induction, time.time() - start_time)
    return result


def filter_features_model_ramp(primitives, current_features, export=False):
    """

    :param primitives: list of dict (each dict contains results from a single simulation)
    :param current_features: dict
    :param export: bool
    :return: dict
    """
    features = {}

    for this_result_dict in primitives:
        for induction_key, feature_dict in this_result_dict.iteritems():
            for feature_name, feature_val in feature_dict.iteritems():
                feature_key = 'induction' + induction_key + '_' + feature_name
                features[feature_key] = feature_val
    return features


def get_objectives(features, export=False):
    """

    :param features: dict
    :param export: bool
    :return: tuple of dict
    """
    objectives = {}

    for objective_name in context.objective_names:
        if objective_name not in features:
            return {}, {}
        objectives[objective_name] = features[objective_name]

    return features, objectives


def get_model_ramp(delta_weights, input_x=None, ramp_x=None, plot=False):
    """

    :param delta_weights: array
    :param input_x: array (x resolution of inputs)
    :param ramp_x: array (x resolution of ramp)
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
    return model_ramp


def get_features_interactive(x, plot=False):
    """

    :param x:
    :param plot:
    :return: dict
    """
    features = {}

    args = get_args_static_model_ramp()
    group_size = len(args[0])
    sequences = [[x] * group_size] + args + [[context.export] * group_size] + [[plot] * group_size]
    primitives = map(compute_features_model_ramp, *sequences)
    new_features = filter_features_model_ramp(primitives, features, context.export)
    features.update(new_features)

    return features


def get_target_synthetic_ramp(induction_loc, target_peak_shift, ramp_x=None, track_length=None, target_peak_val=8.,
                              target_asymmetry=2., target_ramp_width=120.):
    """

    :param induction_loc: float
    :param target_peak_shift: float
    :param ramp_x: array (spatial resolution of ramp)
    :param track_length: float
    :param target_peak_val: float
    :param target_asymmetry: float; (left_width - peak_shift) / (right_width + peak_shift)
    :param target_ramp_width: float
    :return: dict
    """
    if ramp_x is None:
        ramp_x = context.binned_x
    if track_length is None:
        track_length = context.track_length

    synthetic_peak_loc = induction_loc + target_peak_shift

    tuning_amp = target_peak_val / 2.
    tuning_offset = tuning_amp

    left_width = (target_asymmetry * (target_ramp_width + target_peak_shift) + target_peak_shift) / \
                 (1. + target_asymmetry)
    right_width = (target_ramp_width - left_width)
    extended_x = np.concatenate([ramp_x - track_length, ramp_x, ramp_x + track_length])
    peak_index = np.where(extended_x >= synthetic_peak_loc)[0][0]
    target_ramp = np.zeros_like(extended_x)
    target_ramp[:peak_index] = tuning_amp * np.cos(2. * np.pi / left_width / 2. *
                                                   (extended_x[:peak_index] - synthetic_peak_loc)) + tuning_offset
    target_ramp[peak_index:] = tuning_amp * np.cos(2. * np.pi / right_width / 2. *
                                                   (extended_x[peak_index:] - synthetic_peak_loc)) + tuning_offset
    left = np.where(extended_x >= synthetic_peak_loc - left_width)[0][0]
    right = np.where(extended_x > synthetic_peak_loc + right_width)[0][0]
    target_ramp[:left] = 0.
    target_ramp[right:] = 0.
    target_ramp = wrap_around_and_compress(target_ramp, ramp_x)

    return target_ramp


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True,))
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_BTSP_CA1_synthetic_config.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--verbose", type=int, default=2)
@click.option("--plot", is_flag=True)
@click.option("--debug", is_flag=True)
@click.pass_context
def main(cli, config_file_path, output_dir, export, export_file_path, label, verbose, plot, debug):
    """
    Parallel optimization is meant to be executed by the module nested.optimize using the syntax:
    for N processes:
    mpirun -n N python -m nested.optimize --config-file-path=$PATH_TO_CONFIG_FILE --disp --export

    This script can be executed as main with the command line interface to debug the model code, and to generate plots
    after optimization has completed, e.g.:

    ipython
    run optimize_BTSP_CA1_synthetic --plot --debug --config-file-path=$PATH_TO_CONFIG_FILE

    :param cli: contains unrecognized args as list of str
    :param config_file_path: str (path)
    :param output_dir: str (path)
    :param export: bool
    :param export_file_path: str
    :param label: str
    :param verbose: int
    :param plot: bool
    :param debug: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    kwargs = get_unknown_click_arg_dict(cli.args)
    context.disp = verbose > 0
    config_optimize_interactive(__file__, config_file_path=config_file_path, output_dir=output_dir, export=export,
                                export_file_path=export_file_path, label=label, disp=context.disp, verbose=verbose, **kwargs)

    x1_array = context.x0_array

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

    context.update(locals())


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
