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
    nested.parallel is used for parallel map operations. This method imports optional parameters from a config_file and
    initializes a Context object on each worker.
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
    if 'config_file_path' in context() and context.config_file_path is not None:
        if not os.path.isfile(context.config_file_path):
            raise Exception('nested.parallel: config_file_path specifying optional is invalid.')
        else:
            config_dict = read_from_yaml(context.config_file_path)
    else:
        config_dict = {}
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
        context.temp_output_path = '%s%s_pid%i%s_temp_output.hdf5' % \
                                   (output_dir_str, datetime.datetime.today().strftime('%Y%m%d_%H%M'), os.getpid(),
                                    label)
    context.export = export
    if export_file_path is not None:
        context.export_file_path = export_file_path
    if 'export_file_path' not in context() or context.export_file_path is None:
        context.export_file_path = '%s%s%s_exported_output.hdf5' % \
                                   (output_dir_str, datetime.datetime.today().strftime('%Y%m%d_%H%M'), label)
    context.disp = disp

    config_worker()

    try:
        context.interface.start(disp=True)
        context.interface.ensure_controller()
    except Exception:
        pass


def config_worker():
    """

    """
    init_context()


def init_context():
    """

    """
    context.update(context.x0)
    dt = 10.  # ms
    input_field_width = 90.  # cm
    input_field_peak_rate = 40.  # Hz
    num_inputs = 200
    track_length = 187.  # cm
    default_run_vel = 25.  # cm/s

    binned_dx = track_length / 100.  # cm
    binned_x = np.linspace(0., track_length, 100, endpoint=False) + binned_dx / 2.

    num_steps = int(track_length / default_run_vel / dt * 1000.)
    default_x = np.linspace(0., track_length, num_steps, endpoint=False)
    default_t = np.linspace(0., num_steps * dt, num_steps, endpoint=False)

    input_rate_maps, peak_locs = \
        generate_spatial_rate_maps(binned_x, num_inputs, input_field_peak_rate, input_field_width, track_length)

    sm = StateMachine(dt=dt)

    num_cells = context.num_cells
    initial_fraction_active = context.initial_fraction_active
    initial_active_cells = int(num_cells * initial_fraction_active)
    basal_target_representation_density = context.basal_target_representation_density
    reward_target_representation_density = context.reward_target_representation_density

    num_baseline_laps = context.num_baseline_laps
    num_assay_laps = context.num_assay_laps
    num_reward_laps = context.num_reward_laps
    initial_induction_dur = 300.  # ms
    pause_dur = 500.  # ms
    reward_dur = 500.  # ms
    plateau_dur = 300.  # ms
    peak_basal_prob_new_recruitment = 0.5
    peak_reward_prob_new_recruitment = 0.05
    peak_basal_plateau_prob_per_lap = context.peak_basal_plateau_prob_per_lap
    peak_reward_plateau_prob_per_lap = context.peak_reward_plateau_prob_per_lap

    context.update(locals())

    if context.reward_locs is None:
        context.reward_locs = {}
    reward_stop_locs = {}
    for induction in context.reward_locs:
        reward_stop_index = np.where(default_x >= context.reward_locs[induction])[0][0] + \
                               int(reward_dur / dt)
        reward_stop_locs[induction] = default_x[reward_stop_index]

    num_laps = 2 + num_baseline_laps + (num_assay_laps + num_reward_laps) * len(context.reward_locs)
    reward_lap_indexes = {}
    start_lap = 1 + num_baseline_laps
    for induction in context.reward_locs:
        end_lap = start_lap + num_reward_laps
        reward_lap_indexes[induction] = range(start_lap, end_lap)
        start_lap += num_reward_laps + num_assay_laps

    reward_start_indexes, reward_stop_indexes = defaultdict(list), defaultdict(list)
    position = np.array([])
    t = np.array([])
    reward = np.array([])
    running_t = -len(default_t) * dt
    running_position = -track_length
    lap_edge_indexes = []
    for lap in xrange(num_laps):
        prev_len = len(t)
        lap_edge_indexes.append(prev_len)
        position = np.append(position, np.add(default_x, running_position))
        t = np.append(t, np.add(default_t, running_t))
        reward = np.append(reward, np.zeros_like(default_t))
        running_position += track_length
        running_t += len(default_t) * dt
        for induction in context.reward_locs:
            if lap in reward_lap_indexes[induction]:
                start_index = prev_len + np.where(default_x >= context.reward_locs[induction])[0][0]
                reward_start_indexes[induction].append(start_index)
                stop_index = start_index + int(reward_dur / dt)
                reward_stop_indexes[induction].append(stop_index)
                reward[start_index:stop_index] = 1.
    lap_edge_indexes.append(len(t))

    for induction in context.reward_locs:
        reward_start_indexes[induction] = np.array(reward_start_indexes[induction])
        reward_stop_indexes[induction] = np.array(reward_stop_indexes[induction])

    complete_rate_maps = []

    for this_rate_map in input_rate_maps:
        interp_rate_map = np.interp(default_x, binned_x, this_rate_map, period=track_length)
        this_complete_rate_map = np.array([])
        for lap in xrange(num_laps):
            this_complete_rate_map = np.append(this_complete_rate_map, interp_rate_map)
        complete_rate_maps.append(this_complete_rate_map)

    context.update(locals())

    local_signal_filter_t, local_signal_filter, global_filter_t, global_filter = \
        get_signal_filters(context.local_signal_rise, context.local_signal_decay, context.global_signal_rise,
                           context.global_signal_decay, dt)

    local_signals = get_local_signal_population(complete_rate_maps, local_signal_filter, dt)
    local_signal_peak = np.max(local_signals)
    local_signals /= local_signal_peak

    signal_xrange = np.linspace(0., 1., 10000)
    pot_rate = np.vectorize(scaled_single_sigmoid(context.rMC_th, context.rMC_peak, signal_xrange))
    depot_rate = np.vectorize(scaled_double_sigmoid(context.rCM_th1, context.rCM_peak1, context.rCM_th2,
                                                    context.rCM_peak2, signal_xrange, y_end=context.rCM_min2))

    target_initial_ramp = get_target_synthetic_ramp(-context.target_peak_shift,
                                                    target_peak_shift=context.target_peak_shift,
                                                    target_peak_val=context.initial_ramp_peak_val)
    target_initial_induction_loc = -context.target_peak_shift
    target_initial_induction_stop_loc = target_initial_induction_loc + initial_induction_dur / 1000. * default_run_vel
    initial_ramp, initial_delta_weights, discard_ramp_offset, discard_residual_score = \
        get_delta_weights_LSA(target_initial_ramp, input_rate_maps, target_initial_induction_loc,
                              target_initial_induction_stop_loc,
                              bounds=(context.min_delta_weight, context.peak_delta_weight), verbose=context.verbose)
    
    max_ramp_population_sum = np.mean(initial_ramp) * num_cells
    
    initial_weights_population = []
    d_peak_indexes = int(len(peak_locs) / initial_active_cells)
    initial_peak_indexes = np.linspace(0, len(peak_locs), initial_active_cells, dtype=int, endpoint=False) + \
                           int(d_peak_indexes / 2)
    for i in xrange(num_cells):
        if i < initial_active_cells:
            roll_indexes = initial_peak_indexes[i]
            this_weights = np.add(np.roll(initial_delta_weights, roll_indexes), 1.)
            initial_weights_population.append(this_weights)
        else:
            initial_weights_population.append(np.ones_like(peak_locs))

    ramp_xscale = np.linspace(0., 10., 10000)
    basal_plateau_prob_f = scaled_single_sigmoid(4., 8., ramp_xscale, ylim=[peak_basal_prob_new_recruitment, 1.])
    basal_plateau_prob_f = np.vectorize(basal_plateau_prob_f)
    reward_plateau_prob_f = scaled_single_sigmoid(0., 4., ramp_xscale, ylim=[peak_reward_prob_new_recruitment, 1.])
    reward_plateau_prob_f = np.vectorize(reward_plateau_prob_f)

    interp_target_ramp = np.interp(default_x, binned_x, target_initial_ramp, period=track_length)
    basal_target_plateau_prob = basal_plateau_prob_f(interp_target_ramp)
    basal_plateau_prob_norm_factor = 1. / np.sum(basal_target_plateau_prob)
    reward_target_plateau_prob = reward_plateau_prob_f(interp_target_ramp)
    reward_plateau_prob_norm_factor = 1. / np.sum(reward_target_plateau_prob)

    basal_plateau_modulation_f = lambda this_representation_density: \
        max(0., peak_basal_plateau_prob_per_lap *
            (1. - this_representation_density / basal_target_representation_density))
    basal_plateau_modulation_f = np.vectorize(basal_plateau_modulation_f)

    reward_plateau_modulation_f = lambda this_representation_density, this_reward: \
        max(0., this_reward * peak_reward_plateau_prob_per_lap *
            (1. - this_representation_density / reward_target_representation_density))
    reward_plateau_modulation_f = np.vectorize(reward_plateau_modulation_f)

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


def get_local_signal_population(input_rate_maps, local_filter, dt):
    """

    :param input_rate_maps: list of array
    :param local_filter: array
    :param dt: float
    :return:
    """
    local_signals = []
    for rate_map in input_rate_maps:
        local_signals.append(get_local_signal(rate_map, local_filter, dt))
    return local_signals


def calculate_ramp_features(ramp, induction_loc, offset=False, smooth=False):
    """

    :param ramp: array
    :param induction_loc: float
    :param offset: bool
    :param smooth: bool
    :return tuple of float
    """
    binned_x = context.binned_x
    track_length = context.track_length
    default_x = context.default_x
    extended_binned_x = np.concatenate([binned_x - track_length, binned_x, binned_x + track_length])
    if smooth:
        local_ramp = signal.savgol_filter(ramp, 21, 3, mode='wrap')
    else:
        local_ramp = np.array(ramp)
    extended_binned_ramp = np.concatenate([local_ramp] * 3)
    extended_interp_x = np.concatenate([default_x - track_length, default_x,
                                        default_x + track_length])
    extended_ramp = np.interp(extended_interp_x, extended_binned_x, extended_binned_ramp)
    interp_ramp = extended_ramp[len(default_x):2 * len(default_x)]
    baseline_indexes = np.where(interp_ramp <= np.percentile(interp_ramp, 10.))[0]
    baseline = np.mean(interp_ramp[baseline_indexes])
    if offset:
        interp_ramp -= baseline
        extended_ramp -= baseline
    peak_index = np.where(interp_ramp == np.max(interp_ramp))[0][0] + len(interp_ramp)
    peak_val = extended_ramp[peak_index]
    peak_x = extended_interp_x[peak_index]
    start_index = np.where(extended_ramp[:peak_index] <=
                           0.15 * (peak_val - baseline) + baseline)[0][-1]
    end_index = peak_index + np.where(extended_ramp[peak_index:] <= 0.15 *
                                      (peak_val - baseline) + baseline)[0][0]
    start_loc = float(start_index % len(default_x)) / float(len(default_x)) * track_length
    end_loc = float(end_index % len(default_x)) / float(len(default_x)) * track_length
    peak_loc = float(peak_index % len(default_x)) / float(len(default_x)) * track_length
    min_index = np.where(interp_ramp == np.min(interp_ramp))[0][0] + len(interp_ramp)
    min_val = extended_ramp[min_index]
    min_loc = float(min_index % len(default_x)) / float(len(default_x)) * track_length
    peak_shift = peak_x - induction_loc
    if peak_shift > track_length / 2.:
        peak_shift = -(track_length - peak_shift)
    elif peak_shift < -track_length / 2.:
        peak_shift += track_length
    # ramp_width = extended_interp_x[end_index] - extended_interp_x[start_index]
    scaled_ramp = np.divide(local_ramp, peak_val)
    ramp_width = np.trapz(scaled_ramp, binned_x)
    before_width = induction_loc - start_loc
    if induction_loc < start_loc:
        before_width += track_length
    after_width = end_loc - induction_loc
    if induction_loc > end_loc:
        after_width += track_length
    if after_width == 0:
        ratio = before_width
    else:
        ratio = before_width / after_width
    return peak_val, ramp_width, peak_shift, ratio, start_loc, peak_loc, end_loc, min_val, min_loc


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
        interp_target_ramp = np.interp(input_x, ramp_x, target_ramp, period=context.track_length)
    else:
        interp_target_ramp = np.array(target_ramp)

    model_ramp = delta_weights.dot(input_matrix)
    if len(model_ramp) != len(ramp_x):
        model_ramp = np.interp(ramp_x, input_x, model_ramp, period=context.track_length)

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
        calculate_ramp_features(interp_target_ramp, induction_loc)

    ramp_amp['model'], ramp_width['model'], peak_shift['model'], ratio['model'], start_loc['model'], \
    peak_loc['model'], end_loc['model'], min_val['model'], min_loc['model'] = \
        calculate_ramp_features(model_ramp, induction_loc)

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


def get_delta_weights_LSA(target_ramp, input_rate_maps, induction_loc, induction_stop_loc, initial_delta_weights=None,
                          bounds=None, beta=2., ramp_x=None, input_x=None, allow_offset=False, impose_offset=None, 
                          plot=False, verbose=1):
    """
    Uses least square approximation to estimate a set of weights to match any arbitrary place field ramp, agnostic
    about underlying kernel, induction velocity, etc.
    :param target_ramp: dict of array
    :param input_rate_maps: array; x=default_x
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
        interp_target_ramp = np.interp(input_x, ramp_x, target_ramp, period=context.track_length)
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


def calculate_weight_dynamics(cell_index, lap, initial_weights, initial_ramp, initial_global_signal,
                              initial_plateau_prob, last_plateau_stop_time):
    """

    :param cell_index: int
    :param lap: int
    :param initial_weights: array
    :param initial_ramp: array
    :param initial_global_signal: array
    :param initial_plateau_prob: array
    :param last_plateau_stop_time: float
    :return: array
    """
    start_time = time.time()
    cell_index = int(cell_index)
    lap = int(lap)
    seed = context.seed_offset + 1e6 * lap + cell_index
    local_random = random.Random()
    local_random.seed(seed)

    lap_start_index, lap_end_index = context.lap_edge_indexes[lap], context.lap_edge_indexes[lap + 1]
    # compute convolution of induction gate and global signal filter for this and next lap
    # if there is no next lap, the carryover signal array will be empty
    if len(context.lap_edge_indexes) > lap + 2:
        next_lap_end_index = context.lap_edge_indexes[lap + 2]
    else:
        next_lap_end_index = context.lap_edge_indexes[lap + 1]
    this_lap_t = context.t[lap_start_index:lap_end_index]
    this_epoch_t = context.t[lap_start_index:next_lap_end_index]

    plateau_start_times = []
    plateau_stop_times = []
    induction_gate = np.zeros_like(this_epoch_t)

    plateau_len = int(context.plateau_dur / context.dt)
    pause_len = int(context.pause_dur / context.dt)

    if last_plateau_stop_time > this_lap_t[0]:
        start_index = np.where(this_lap_t > last_plateau_stop_time)[0][0]
        induction_gate[:start_index] = 1.
        start_index += pause_len
    else:
        start_index = 0

    plateau_prob = np.array(initial_plateau_prob)
    for j in xrange(start_index, len(this_lap_t)):
        this_t = this_lap_t[j]
        if plateau_prob[j] > 0. and local_random.random() < plateau_prob[j]:
            plateau_start_times.append(this_t)
            plateau_stop_times.append(this_t + context.plateau_dur)
            induction_gate[j:j + plateau_len] = 1.
            plateau_prob[j:j + plateau_len + pause_len] = 0.

    global_signal = get_global_signal(induction_gate, context.global_filter)
    global_signal /= context.global_signal_peak
    global_signal[:len(initial_global_signal)] += initial_global_signal

    peak_weight = context.peak_delta_weight + 1.
    norm_initial_weights = np.divide(initial_weights, peak_weight)
    weights_history = []
    if len(plateau_start_times) > 0:
        for i in xrange(len(context.peak_locs)):
            # normalize total number of receptors
            this_initial_weight = norm_initial_weights[i]
            available = 1. - this_initial_weight
            context.sm.update_states({'M': available, 'C': this_initial_weight})
            local_signal = context.local_signals[i][lap_start_index:next_lap_end_index]
            context.sm.update_rates(
                {'M': {'C': context.rMC0 * np.multiply(context.pot_rate(local_signal), global_signal)},
                 'C': {'M': context.rCM0 * np.multiply(context.depot_rate(local_signal), global_signal)}})
            context.sm.reset()
            context.sm.run()
            weights_history.append(context.sm.states_history['C'] * peak_weight)
    else:
        weights_history = \
            [np.append(np.ones_like(this_epoch_t), 1.) * initial_weight for initial_weight in initial_weights]

    weights_history = np.array(weights_history)

    next_global_signal = global_signal[len(context.default_t):]

    if context.disp:
        print 'Process: %i: calculating ramp for cell: %i, lap: %i took %.1f s' % \
              (os.getpid(), cell_index, lap, time.time() - start_time)

    return plateau_start_times, plateau_stop_times, next_global_signal, weights_history


def get_population_representation_density(ramp_population):
    """

    :param ramp_population: list of array (like binned_x)
    :return: array (like default_t)
    """
    binned_ramp_population_sum = np.sum(ramp_population, axis=0)
    ramp_population_sum = np.interp(context.default_x, context.binned_x, binned_ramp_population_sum,
                                    period=context.track_length)
    population_representation_density = ramp_population_sum / context.max_ramp_population_sum
    return population_representation_density


def get_plateau_probability_population(ramp_population, population_representation_density, current_reward):
    """

    :param ramp_population: list of array (like binned_x)
    :param population_representation_density: array (like default_x)
    :param current_reward: array (like default_x)
    :return: list of array (like default_x)
    """
    basal_plateau_modulation_factor = context.basal_plateau_modulation_f(population_representation_density)
    reward_plateau_modulation_factor = \
        context.reward_plateau_modulation_f(population_representation_density, current_reward)
    plateau_probability_population = []
    for ramp in ramp_population:
        interp_ramp = np.interp(context.default_x, context.binned_x, ramp, period=context.track_length)
        plateau_prob = context.basal_plateau_prob_f(interp_ramp) * context.basal_plateau_prob_norm_factor * \
                       basal_plateau_modulation_factor
        reward_plateau_prob = context.reward_plateau_prob_f(interp_ramp) * context.reward_plateau_prob_norm_factor * \
                              reward_plateau_modulation_factor
        plateau_prob += reward_plateau_prob
        plateau_probability_population.append(plateau_prob)
    return plateau_probability_population


def calculate_population_dynamics(export=False, plot=False):
    """

    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    cell_indexes = range(len(context.initial_weights_population))
    if 'debug' in context() and context.debug:
        cell_indexes = cell_indexes[::int(len(context.initial_weights_population)/context.interface.num_workers)]
        cell_indexes = cell_indexes[:context.interface.num_workers]
        print 'cell_indexes: %s' % cell_indexes
    group_size = len(cell_indexes)

    plateau_start_times_population = defaultdict(list)
    plateau_stop_times_population = defaultdict(list)
    weights_population_full_history = []
    plateau_probability_history = []
    population_representation_density_history = []
    weights_snapshots = []
    ramp_snapshots = []
    current_weights_population = np.array(context.initial_weights_population)[cell_indexes]
    weights_snapshots.append(current_weights_population)
    current_ramp_population = map(get_model_ramp, current_weights_population)
    ramp_snapshots.append(current_ramp_population)
    initial_global_signal_population = [np.zeros_like(context.default_t)] * group_size

    for lap in xrange(1, context.num_laps -1):
        lap_start_index, lap_end_index = context.lap_edge_indexes[lap], context.lap_edge_indexes[lap+1]
        current_reward = context.reward[lap_start_index:lap_end_index]
        population_representation_density = get_population_representation_density(current_ramp_population)
        population_representation_density_history.append(population_representation_density)
        current_plateau_probability_population = \
            get_plateau_probability_population(current_ramp_population, population_representation_density,
                                               current_reward)
        last_plateau_stop_times = []
        for cell_index in cell_indexes:
            if cell_index in plateau_stop_times_population and len(plateau_stop_times_population[cell_index]) > 0:
                last_plateau_stop_times.append(plateau_stop_times_population[cell_index][-1])
            else:
                last_plateau_stop_times.append(0.)

        result = context.interface.map(calculate_weight_dynamics, cell_indexes,
                                       [lap] * group_size, current_weights_population, current_ramp_population,
                                       initial_global_signal_population,
                                       current_plateau_probability_population,
                                       last_plateau_stop_times)
        this_plateau_start_times_population, this_plateau_stop_times_population, initial_global_signal_population, \
            this_weights_population_history = zip(*result)
        weights_population_full_history.append(np.array(this_weights_population_history))
        plateau_probability_history.append(current_plateau_probability_population)
        for cell_index, this_plateau_start_times, this_plateau_stop_times in \
                zip(cell_indexes, this_plateau_start_times_population, this_plateau_stop_times_population):
            plateau_start_times_population[cell_index].extend(this_plateau_start_times)
            plateau_stop_times_population[cell_index].extend(this_plateau_stop_times)
        current_weights_population = []
        for this_weights_history in this_weights_population_history:
            current_weights_population.append(this_weights_history[:,-1])
        weights_snapshots.append(current_weights_population)
        current_ramp_population = map(get_model_ramp, current_weights_population)
        ramp_snapshots.append(current_ramp_population)

    population_representation_density = get_population_representation_density(current_ramp_population)
    population_representation_density_history.append(population_representation_density)
    population_representation_density_history = np.array(population_representation_density_history)
    weights_population_full_history = np.array(weights_population_full_history)
    weights_snapshots = np.array(ramp_snapshots)
    ramp_snapshots = np.array(ramp_snapshots)

    if context.disp:
        print 'Process: %i: calculating ramp population took %.1f s' % (os.getpid(), time.time() - start_time)

    reward_locs_array = [context.reward_locs[induction] for induction in context.reward_locs]

    if plot:
        plot_ramp_snapshots(ramp_snapshots, reward_locs_array, context.binned_x, context.track_length,
                            context.num_baseline_laps, context.num_assay_laps, context.num_reward_laps)

    plateau_start_times_array = []
    plateau_stop_times_array = []
    plateau_times_cell_indexes = []
    for cell_index, this_plateau_start_times in plateau_start_times_population.iteritems():
        plateau_start_times_array.extend(this_plateau_start_times)
        plateau_stop_times_array.extend(plateau_start_times_population[cell_index])
        plateau_times_cell_indexes.extend([cell_index] * len(this_plateau_start_times))

    if export:
        with h5py.File(context.export_file_path, 'a') as f:
            shared_context_key = 'shared_context'
            if shared_context_key not in f:
                f.create_group(shared_context_key)
                group = f[shared_context_key]
                group.create_dataset('peak_locs', compression='gzip', data=context.peak_locs)
                group.create_dataset('binned_x', compression='gzip', data=context.binned_x)
                group.create_dataset('default_x', compression='gzip', data=context.default_x)
                group.create_dataset('input_rate_maps', compression='gzip', data=np.array(context.input_rate_maps))
                group.attrs['track_length'] = context.track_length
                group.attrs['dt'] = context.dt
            exported_data_key = 'exported_data'
            if exported_data_key not in f:
                f.create_group(exported_data_key)
                f[exported_data_key].attrs['enumerated'] = False
            description = 'weights_population_history'
            if description not in f[exported_data_key]:
                f[exported_data_key].create_group(description)
            group = f[exported_data_key][description]
            group.attrs['num_baseline_laps'] = context.num_baseline_laps
            group.attrs['num_assay_laps'] = context.num_assay_laps
            group.attrs['num_reward_laps'] = context.num_reward_laps
            group.attrs['plateau_dur'] = context.plateau_dur
            group.attrs['reward_dur'] = context.reward_dur
            group.attrs['reward_locs_array'] = reward_locs_array
            group.attrs['ramp_scaling_factor'] = context.ramp_scaling_factor
            group.create_dataset('t', compression='gzip', data=context.t)
            group.create_dataset('position', compression='gzip', data=context.position)
            group.create_dataset('reward', compression='gzip', data=context.reward)
            group.create_dataset('plateau_times_cell_indexes', compression='gzip', data=plateau_times_cell_indexes)
            group.create_dataset('plateau_start_times_array', compression='gzip', data=plateau_start_times_array)
            group.create_dataset('plateau_stop_times_array', compression='gzip', data=plateau_stop_times_array)
            group.create_dataset('weights_population_full_history', compression='gzip',
                                 data=weights_population_full_history)
            group.create_dataset('plateau_probability_history', compression='gzip',
                                 data=plateau_probability_history)
            group.create_dataset('population_representation_density_history', compression='gzip',
                                 data=population_representation_density_history)
            group.create_dataset('weights_snapshots', compression='gzip', data=weights_snapshots)
            group.create_dataset('ramp_snapshots', compression='gzip', data=ramp_snapshots)
        if context.disp:
            print 'Process: %i: exported weights population history data to file: %s' % \
                  (os.getpid(), context.export_file_path)

    return plateau_start_times_population, plateau_stop_times_population, weights_population_full_history, \
           plateau_probability_history, weights_snapshots, ramp_snapshots


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


def get_model_ramp(weights):
    """

    :param weights: array
    :return: array
    """
    this_delta_weights = weights - 1.
    this_ramp = np.multiply(this_delta_weights.dot(context.input_rate_maps), context.ramp_scaling_factor)
    return this_ramp


def plot_ramp_snapshots(ramp_snapshots, reward_locs_array, binned_x, track_length, num_baseline_laps, num_assay_laps,
                        num_reward_laps):
    """

    :param ramp_snapshots: 3D array
    :param reward_locs_array: array
    :param binned_x: array
    :param track_length: float
    :param num_baseline_laps: int
    :param num_assay_laps: int
    :param num_reward_laps: int
    """
    snapshot_laps = [0, num_baseline_laps]
    lap_labels = ['Lap 1: Before', 'Laps 2-%i: No reward' % (num_baseline_laps + 1)]
    for reward_loc in reward_locs_array:
        start = max(snapshot_laps)
        stop = start + num_reward_laps
        snapshot_laps.append(stop)
        lap_labels.append('Laps %i-%i: Reward at %i cm' % (start + 2, stop + 1, reward_loc))
        start = max(snapshot_laps)
        stop = start + num_reward_laps
        snapshot_laps.append(stop)
        lap_labels.append('Laps %i-%i: No reward' % (start + 2, stop + 1))
    pprint.pprint(lap_labels)

    peak_loc_history = defaultdict(list)
    delta_peak_loc_history = defaultdict(list)
    active_cell_index_set = set()

    num_cells = len(ramp_snapshots[0])
    peak_locs_snapshots = []
    for lap, this_ramp_snapshot in enumerate(ramp_snapshots):
        this_peak_locs_snapshot = []
        for cell_index, this_ramp in enumerate(this_ramp_snapshot):
            if np.any(this_ramp > 0.):
                active_cell_index_set.add(cell_index)
                peak_index = np.argmax(this_ramp)
                this_peak_loc = binned_x[peak_index]
                this_peak_locs_snapshot.append(this_peak_loc)
            else:
                this_peak_loc = np.nan
            if lap > 0:
                if np.isnan(this_peak_loc) or np.isnan(peak_loc_history[cell_index][-1]):
                    this_delta_peak_loc = np.nan
                else:
                    this_delta_peak_loc = this_peak_loc - peak_loc_history[cell_index][-1]
                    if this_delta_peak_loc < -track_length / 2.:
                        this_delta_peak_loc += track_length
                    elif this_delta_peak_loc > track_length / 2.:
                        this_delta_peak_loc -= track_length
                delta_peak_loc_history[cell_index].append(this_delta_peak_loc)
            peak_loc_history[cell_index].append(this_peak_loc)
        peak_locs_snapshots.append(np.array(this_peak_locs_snapshot))

    peak_shift_count = 0
    for cell_index, this_delta_peak_loc_history in delta_peak_loc_history.iteritems():
        this_delta_peak_loc_history_array = np.array(this_delta_peak_loc_history)
        valid = this_delta_peak_loc_history_array[~np.isnan(this_delta_peak_loc_history_array)]
        if np.any(valid > 0.):
            peak_shift_count += 1
    print 'peak_shift_count: %i / %i active cells' % (peak_shift_count, len(active_cell_index_set))

    edges = np.linspace(0., track_length, 51)
    bin_width = edges[1] - edges[0]
    context.update(locals())

    fig1, axes1 = plt.subplots()
    fig2, axes2 = plt.subplots()
    for lap, lap_label in zip(snapshot_laps, lap_labels):
        this_ramp_snapshot = ramp_snapshots[lap]
        this_peak_locs_snapshot = peak_locs_snapshots[lap]
        axes1.plot(binned_x, np.sum(this_ramp_snapshot, axis=0), label=lap_label)
        hist, _edges = np.histogram(this_peak_locs_snapshot, bins=edges)
        hist = hist.astype(float)
        hist /= float(num_cells)
        axes2.plot(edges[:-1] + bin_width / 2., hist, label=lap_label)
    axes1.set_xlabel('Location (cm)')
    axes1.set_ylabel('Summed population activity')
    axes1.set_ylim([0., axes1.get_ylim()[1]])
    axes1.legend(loc='best', frameon=False, framealpha=0.5)
    axes2.set_xlabel('Place field peak locations (cm)')
    axes2.set_ylabel('Probability')
    axes2.legend(loc='best', frameon=False, framealpha=0.5)
    clean_axes([axes1, axes2])
    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()


def analyze_simulation_output(file_path):
    """

    :param file_path: str (path)
    """
    if not os.path.isfile(file_path):
        raise IOError('analyze_simulation_output: invalid file path: %s' % file_path)
    with h5py.File(file_path, 'r') as f:
        if 'shared_context' not in f or 'exported_data' not in f or \
                'weights_population_history' not in f['exported_data']:
            raise KeyError('analyze_simulation_output: invalid file contents at path: %s' % file_path)
        binned_x = f['shared_context']['binned_x'][:]
        track_length = f['shared_context'].attrs['track_length']
        group = f['exported_data']['weights_population_history']
        reward_locs_array = group.attrs['reward_locs_array']
        ramp_snapshots = group['ramp_snapshots'][:]
        num_baseline_laps = group.attrs['num_baseline_laps']
        num_assay_laps = group.attrs['num_assay_laps']
        num_reward_laps = group.attrs['num_reward_laps']

    plot_ramp_snapshots(ramp_snapshots, reward_locs_array, binned_x, track_length, num_baseline_laps, num_assay_laps,
                        num_reward_laps)


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
@click.option("--debug", is_flag=True)
def main(config_file_path, output_dir, export, export_file_path, label, verbose, plot, simulate, analyze,
         data_file_path, interactive, debug):
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
    :param debug: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    context.disp = verbose > 0
    if simulate:
        context.interface = ParallelContextInterface()
        context.interface.apply(config_parallel_interface, config_file_path=config_file_path, output_dir=output_dir,
                                export=export, export_file_path=export_file_path, label=label, disp=context.disp,
                                verbose=verbose)
        plateau_start_times_population, plateau_stop_times_population, weights_population_full_history, \
            plateau_probability_history, weights_snapshots, ramp_snapshots = calculate_population_dynamics(export, plot)
    elif debug:
        config_parallel_interface(config_file_path=config_file_path, output_dir=output_dir, export=export,
                                  export_file_path=export_file_path, label=label, disp=context.disp, verbose=verbose)

    if analyze:
        analyze_simulation_output(data_file_path)

    context.update(locals())

    if simulate and not interactive:
        try:
            context.interface.stop()
        except Exception:
            pass


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
