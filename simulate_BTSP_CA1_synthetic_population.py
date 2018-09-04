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
from nested.parallel import *
from nested.optimize_utils import *
import click


context = Context()


def config_parallel_interface(config_file_path=None, output_dir=None, temp_output_path=None, export=False,
                              export_file_path=None, label=None, disp=True, **kwargs):
    """
    nested.optimize is meant to be executed as a module, and refers to a config_file to import required submodules and
    create a workflow for optimization. During development of submodules, it is useful to be able to execute a submodule
    as a standalone script (as '__main__'). config_interactive allows a single process to properly parse the
    config_file and initialize a Context for testing purposes.
    :param config_file_path: str (.yaml file path)
    :param output_dir: str (dir path)
    :param temp_output_path: str (.hdf5 file path)
    :param export: bool
    :param export_file_path: str (.hdf5 file path)
    :param label: str
    :param disp: bool
    """
    if config_file_path is not None:
        context.config_file_path = config_file_path
    if 'config_file_path' not in context() or context.config_file_path is None or \
            not os.path.isfile(context.config_file_path):
        raise Exception('nested.optimize: config_file_path specifying required parameters is missing or invalid.')
    config_dict = read_from_yaml(context.config_file_path)
    context.update(config_dict)
    context.kwargs = config_dict  # Extra arguments to be passed to imported sources

    if label is not None:
        context.label = label
    if 'label' not in context() or context.label is None:
        label = ''
    else:
        label = '_' + context.label

    if output_dir is not None:
        context.output_dir = output_dir
    if 'output_dir' not in context():
        context.output_dir = None
    if context.output_dir is None:
        output_dir_str = ''
    else:
        output_dir_str = context.output_dir + '/'

    if temp_output_path is not None:
        context.temp_output_path = temp_output_path
    if 'temp_output_path' not in context() or context.temp_output_path is None:
        context.temp_output_path = '%s%s_pid%i_%s_temp_output.hdf5' % \
                                   (output_dir_str, datetime.datetime.today().strftime('%Y%m%d%H%M'), os.getpid(),
                                    label)
    context.export = export
    if export_file_path is not None:
        context.export_file_path = export_file_path
    if 'export_file_path' not in context() or context.export_file_path is None:
        context.export_file_path = '%s%s_%s_interactive_exported_output.hdf5' % \
                                   (output_dir_str, datetime.datetime.today().strftime('%Y%m%d%H%M'), label)
    context.disp = disp

    config_worker(context.temp_output_path, context.export_file_path, context.output_dir, context.disp,
                    **context.kwargs)

    try:
        context.interface.start(disp=True)
        context.interface.ensure_controller()
    except Exception:
        pass


def config_worker(temp_output_path, export_file_path, output_dir, disp, verbose=1, **kwargs):
    """
    :param temp_output_path: str
    :param export_file_path: str
    :param output_dir: str (dir path)
    :param disp: bool
    :param verbose: int
    """
    context.update(locals())
    context.update(kwargs)
    init_context()


def init_context():
    """

    """
    context.update(context.x0)
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
    num_assay_laps = 5
    num_induction_laps = 5
    induction_dur = 300.  # ms
    basal_prob_plateau = 0.1
    reward_prob_plateau_factor = 5.

    context.update(locals())

    complete_induction_locs = {}
    complete_induction_durs = {}
    if context.induction_locs is None:
        context.induction_locs = {}
    induction_stop_locs = {}
    for induction in context.induction_locs:
        complete_induction_locs[induction] = [context.induction_locs[induction] for i in xrange(num_induction_laps)]
        complete_induction_durs[induction] = [induction_dur for i in xrange(num_induction_laps)]
        induction_stop_index = np.where(default_interp_x >= context.induction_locs[induction])[0][0] + \
                               int(induction_dur / dt)
        induction_stop_locs[induction] = default_interp_x[induction_stop_index]

    num_laps = 2 + num_assay_laps + (num_assay_laps + num_induction_laps) * len(context.induction_locs)
    induction_lap_indexes = {}
    start_lap = num_assay_laps
    for induction in context.induction_locs:
        end_lap = start_lap + num_induction_laps
        induction_lap_indexes[induction] = range(start_lap, end_lap)
        start_lap += num_induction_laps + num_assay_laps

    induction_start_indexes, induction_stop_indexes, induction_start_times, induction_stop_times = \
        defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    position = np.array([])
    t = np.array([])
    reward = np.array([])
    running_t = -len(default_interp_t) * dt
    running_position = -track_length
    for lap in xrange(num_laps):
        prev_len = len(t)
        position = np.append(position, np.add(default_interp_x, running_position))
        t = np.append(t, np.add(default_interp_t, running_t))
        reward = np.append(reward, np.zeros_like(default_interp_t))
        running_position += track_length
        running_t += len(default_interp_t) * dt
        for induction in context.induction_locs:
            if lap in induction_lap_indexes[induction]:
                start_index = prev_len + np.where(default_interp_x >= context.induction_locs[induction])[0][0]
                induction_start_indexes[induction].append(start_index)
                stop_index = start_index + int(induction_dur / dt)
                induction_stop_indexes[induction].append(stop_index)
                induction_start_times[induction].append(t[start_index])
                induction_stop_times[induction].append(t[stop_index])
                reward[start_index:stop_index] = 1.

    for induction in context.induction_locs:
        induction_start_times[induction] = np.array(induction_start_times[induction])
        induction_stop_times[induction] = np.array(induction_stop_times[induction])
        induction_start_indexes[induction] = np.array(induction_start_indexes[induction])
        induction_stop_indexes[induction] = np.array(induction_stop_indexes[induction])

    down_t = np.arange(t[0], t[-1] + down_dt / 2., down_dt)
    down_reward = np.interp(down_t, t, reward)

    complete_rate_maps = []
    down_rate_maps = []
    for this_rate_map in input_rate_maps:
        interp_rate_map = np.interp(default_interp_x, binned_x, this_rate_map)
        this_complete_rate_map = np.array([])
        for lap in xrange(num_laps):
            this_complete_rate_map = np.append(this_complete_rate_map, interp_rate_map)
        complete_rate_maps.append(this_complete_rate_map)
        this_down_rate_map = np.interp(down_t, t, this_complete_rate_map)
        down_rate_maps.append(this_down_rate_map)

    context.update(locals())

    local_signal_filter_t, local_signal_filter, global_filter_t, global_filter = \
        get_signal_filters(context.local_signal_rise, context.local_signal_decay, context.global_signal_rise,
                           context.global_signal_decay, down_dt)

    local_signals = get_local_signal_population(local_signal_filter)
    local_signal_peak = np.max(local_signals)
    local_signals /= local_signal_peak

    signal_xrange = np.linspace(0., 1., 10000)
    pot_rate = np.vectorize(scaled_single_sigmoid(context.rMC_th, context.rMC_peak, signal_xrange))
    depot_rate = np.vectorize(scaled_double_sigmoid(context.rCM_th1, context.rCM_peak1, context.rCM_th2,
                                                    context.rCM_peak2, signal_xrange, y_end=context.rCM_min2))

    target_initial_ramp = get_target_synthetic_ramp(-context.target_peak_shift,
                                                    target_peak_shift=context.target_peak_shift,
                                                    target_peak_val=context.initial_ramp_peak_val,
                                                    ramp_x=peak_locs)
    target_initial_ramp = np.interp(binned_x, peak_locs, target_initial_ramp)
    target_initial_induction_loc = -context.target_peak_shift
    target_initial_induction_stop_loc = target_initial_induction_loc + induction_dur / 1000. * default_run_vel
    initial_ramp, initial_delta_weights, discard_ramp_offset, discard_residual_score = \
        get_delta_weights_LSA(target_initial_ramp, input_rate_maps, target_initial_induction_loc,
                              target_initial_induction_stop_loc,
                              bounds=(context.min_delta_weight, context.peak_delta_weight), verbose=context.verbose)
    interp_initial_ramp = np.interp(peak_locs, binned_x, initial_ramp)

    ramp_xrange = np.linspace(0., 10., 10000)
    prob_plateau = np.vectorize(scaled_single_sigmoid(6., 9., ramp_xrange))
    target_prob_plateau = np.array(prob_plateau(interp_initial_ramp))
    target_prob_plateau = np.maximum(target_prob_plateau, 0.)
    prob_plateau_norm_factor = np.trapz(target_prob_plateau, dx=track_length/float(len(peak_locs)))
    target_prob_plateau *= basal_prob_plateau / prob_plateau_norm_factor

    binned_ramp_population = []
    initial_weights_population = []
    down_prob_plateau_population = []
    for i in xrange(len(peak_locs)):
        this_initial_ramp = np.interp(binned_x, peak_locs, np.roll(interp_initial_ramp, i))
        this_weights = np.add(np.roll(initial_delta_weights, i), 1.)
        binned_ramp_population.append(this_initial_ramp)
        initial_weights_population.append(this_weights)
        this_prob_plateau = np.interp(default_interp_x, peak_locs, np.roll(target_prob_plateau, i))
        this_complete_prob_plateau = np.zeros_like(default_interp_x)
        for lap in xrange(1, num_laps - 1):
            this_complete_prob_plateau = np.append(this_complete_prob_plateau, this_prob_plateau)
        this_complete_prob_plateau = np.append(this_complete_prob_plateau, np.zeros_like(default_interp_x))
        this_down_prob_plateau = np.interp(down_t, t, this_complete_prob_plateau)
        this_down_prob_plateau = np.multiply(this_down_prob_plateau,
                                             np.add(down_reward * (reward_prob_plateau_factor - 1.), 1.))
        down_prob_plateau_population.append(this_down_prob_plateau)

    plateau_start_indexes_population = []
    down_induction_gate_population = []
    global_signal_population = []
    down_plateau_len = int(induction_dur / down_dt)
    local_random = random.Random()
    local_random.seed(context.seed)

    for i in xrange(len(down_prob_plateau_population)):
        this_prob_plateau = np.array(down_prob_plateau_population[i])
        this_induction_gate = np.zeros_like(this_prob_plateau)
        this_plateau_start_indexes = []
        for j in xrange(len(this_prob_plateau)):
            if this_prob_plateau[j] > 0. and local_random.random() < this_prob_plateau[j]:
                this_plateau_start_indexes.append(j)
                this_induction_gate[j:j+down_plateau_len] = 1.
                this_prob_plateau[j:j+2*down_plateau_len] = 0.
        this_global_signal = get_global_signal(this_induction_gate, global_filter)
        plateau_start_indexes_population.append(this_plateau_start_indexes)
        down_induction_gate_population.append(this_induction_gate)
        global_signal_population.append(this_global_signal)
    global_signal_population /= np.max(global_signal_population)

    context.update(locals())


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

    Err += ((min_val['after'] - context.target_val['min_val_2']) / context.target_range['min_val']) ** 2.
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


def calculate_weight_dynamics(cell_index):
    """

    :param cell_index: int
    :return: array
    """
    start_time = time.time()
    cell_index = int(cell_index)
    global_signal = context.global_signal_population[cell_index]
    initial_weights = context.initial_weights_population[cell_index]
    peak_weight = context.peak_delta_weight + 1.
    norm_initial_weights = np.divide(initial_weights, peak_weight)
    weights_history = []
    for i in xrange(len(context.peak_locs)):
        # normalize total number of receptors
        this_initial_weight = norm_initial_weights[i]
        available = 1. - this_initial_weight
        context.sm.update_states({'M': available, 'C': this_initial_weight})
        local_signal = context.local_signals[i]
        context.sm.update_rates(
            {'M': {'C': context.rMC0 * np.multiply(context.pot_rate(local_signal), global_signal)},
             'C': {'M': context.rCM0 * np.multiply(context.depot_rate(local_signal), global_signal)}})
        context.sm.reset()
        context.sm.run()
        weights_history.append(context.sm.states_history['C'] * peak_weight)
    weights_history = np.array(weights_history)

    if context.disp:
        print 'Process: %i: calculating ramp for cell: %i took %.1f s' % \
              (os.getpid(), cell_index, time.time() - start_time)

    return weights_history


def calculate_ramp_population(export=False, plot=False):
    """

    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    num_cells = context.interface.num_workers  # len(context.global_signal_population)
    weights_history_population = context.interface.map(calculate_weight_dynamics, range(num_cells))
    weights_history_population = np.array(weights_history_population)

    if context.disp:
        print 'Process: %i: calculating ramp population took %.1f s' % (os.getpid(), time.time() - start_time)

    if plot:
        fig, axes = plt.subplots()
        axes.plot(context.peak_locs,
                     np.sum([this_weights_history[:,0] for this_weights_history in weights_history_population], axis=0),
                  label='Before')
        axes.plot(context.peak_locs,
                  np.sum([this_weights_history[:,-1] for this_weights_history in weights_history_population], axis=0),
                  label='After')
        axes.legend(loc='best', frameon=False, framealpha=0.5)
        clean_axes(axes)
        fig.show()

    reward_locs = [context.induction_locs[induction] for induction in xrange(1, len(context.induction_locs) + 1)]
    down_plateau_t_indexes = []
    plateau_cell_indexes = []
    for i, plateau_indexes in enumerate(context.plateau_start_indexes_population):
        plateau_cell_indexes.extend([i] * len(plateau_indexes))
        down_plateau_t_indexes.extend(plateau_indexes)

    if export:
        with h5py.File(context.export_file_path, 'a') as f:
            shared_context_key = 'shared_context'
            if shared_context_key not in f:
                f.create_group(shared_context_key)
                group = f[shared_context_key]
                group.create_dataset('peak_locs', compression='gzip', data=context.peak_locs)
                group.create_dataset('binned_x', compression='gzip', data=context.binned_x)
            exported_data_key = 'exported_data'
            if exported_data_key not in f:
                f.create_group(exported_data_key)
                f[exported_data_key].attrs['enumerated'] = False
            description = 'weights_history_population'
            if description not in f[exported_data_key]:
                f[exported_data_key].create_group(description)
            group = f[exported_data_key][description]
            group.attrs['track_length'] = context.track_length
            group.attrs['num_assay_laps'] = context.num_assay_laps
            group.attrs['num_induction_laps'] = context.num_induction_laps
            group.attrs['induction_dur'] = context.induction_dur
            group.attrs['reward_locs'] = reward_locs
            group.attrs['ramp_scaling_factor'] = context.ramp_scaling_factor
            group.create_dataset('down_t', compression='gzip', data=context.down_t)
            group.create_dataset('t', compression='gzip', data=context.t)
            group.create_dataset('position', compression='gzip', data=context.position)
            group.create_dataset('down_reward', compression='gzip', data=context.down_reward)
            group.create_dataset('reward', compression='gzip', data=context.reward)
            group.create_dataset('plateau_cell_indexes', compression='gzip', data=plateau_cell_indexes)
            group.create_dataset('down_plateau_t_indexes', compression='gzip', data=down_plateau_t_indexes)
            group.create_dataset('weights_history_population', compression='gzip',
                                 data=weights_history_population)
            group.create_dataset('input_rate_maps', compression='gzip',
                                 data=np.array(context.input_rate_maps))
        if context.disp:
            print 'Process: %i: exported weights history population data to file: %s' % \
                  (os.getpid(), context.export_file_path)

    return weights_history_population


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


def analyze_simulation_output(file_path):
    """

    :param file_path: str (path)
    """
    if not os.path.isfile(file_path):
        raise IOError('analyze_simulation_output: invalid file path: %s' % file_path)
    with h5py.File(file_path, 'r') as f:
        if 'shared_context' not in f or 'exported_data' not in f or \
                'weights_history_population' not in f['exported_data']:
            raise KeyError('analyze_simulation_output: invalid file contents at path: %s' % file_path)
        weights_history_population = f['exported_data']['weights_history_population']['weights_history_population'][:]
        peak_locs = f['shared_context']['peak_locs'][:]
    fig, axes = plt.subplots()
    axes.plot(peak_locs,
              np.sum([this_weights_history[:, 0] for this_weights_history in weights_history_population], axis=0),
              label='Before')
    axes.plot(peak_locs,
              np.sum([this_weights_history[:, -1] for this_weights_history in weights_history_population], axis=0),
              label='After')
    axes.legend(loc='best', frameon=False, framealpha=0.5)
    clean_axes(axes)
    plt.show()


@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/simulate_BTSP_CA1_synthetic_population_config.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default='BTSP_synthetic_population')
@click.option("--verbose", type=int, default=2)
@click.option("--plot", is_flag=True)
@click.option("--simulate", is_flag=True)
@click.option("--analyze", is_flag=True)
@click.option("--data-file-path", type=str, default=None)
@click.option("--interactive", is_flag=True)
def main(config_file_path, output_dir, export, export_file_path, label, verbose, plot, simulate, analyze,
         data_file_path, interactive):
    """
    Utilizes nested.parallel for parallel map. Requires mpi4py and NEURON.

    Execute with N processes:
    mpirun -n N python simulate_BTSP_CA1_synthetic_population.py --config-file-path=$PATH_TO_CONFIG_FILE --simulate \
        --export

    or interactively:

    mpirun -n N python -i simulate_BTSP_CA1_synthetic_population.py --config-file-path=$PATH_TO_CONFIG_FILE --simulate \
        --interactive

    :param config_file_path: str (path)
    :param output_dir: str (path)
    :param export: bool
    :param export_file_path: str
    :param label: str
    :param verbose: int
    :param plot: bool
    :param simulate: bool
    :param analyze: bool
    :param data_file_path: str (path)
    :param interactive: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    context.disp = verbose > 0
    if simulate:
        context.interface = ParallelContextInterface()
        context.interface.apply(config_parallel_interface, config_file_path=config_file_path, output_dir=output_dir,
                                export=export, export_file_path=export_file_path, label=label, disp=context.disp,
                                verbose=verbose)
        weights_history_population = calculate_ramp_population(export, plot)

    if analyze:
        analyze_simulation_output(data_file_path)

    context.update(locals())

    if not interactive:
        try:
            context.interface.stop()
        except Exception:
            pass


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
