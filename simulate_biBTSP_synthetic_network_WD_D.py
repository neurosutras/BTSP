"""
These methods aim to optimize a parametric model of bidirectional, state-dependent behavioral timescale synaptic
plasticity to account for the width and amplitude of place fields in an experimental data set from the Magee lab that
includes:
1) Silent cells converted to place cells by spontaneous plateaus
2) Silent cells converted to place cells by experimentally induced plateaus
3) Place cells with pre-existing place fields that shift their place field locations after an experimentally induced
plateau

Features/assumptions of the phenomenological model:
1) Synaptic weights in a silent cell are all = 1 prior to field induction 1. w(t0) = 1
2) Activity at each synapse generates a long duration 'local plasticity signal', or 'eligibility trace' for synaptic
plasticity.
3) Dendritic plateaus generate a long duration 'global plasticity signal', or an 'instructive trace' for synaptic
plasticity.
4) Changes in weight at each synapse are integrated over periods of nonzero overlap between eligibility and instructive
signals, and updated once per lap.

Features/assumptions of weight-dependent model D:
1) Dendritic plateaus generate a global instructive signal that provides a necessary cofactor required to convert
plasticity eligibility signals at each synapse into either increases or decreases in synaptic strength.
2) Activity at each synapse generates a local plasticity eligibility signal that, in conjunction with the global
instructive signal, can activate both a forward process to increase synaptic strength and a reverse process to decrease
synaptic strength.
3) Synaptic resources can be in 2 states (Markov-style kinetic scheme):

        k_pot * f_pot(local_signal * global_signal)
I (inactive) <------------------------------> A (active)
        k_dep * f_dep(local_signal * global_signal)

4) global_signals are pooled across all cells and normalized to a peak value of 1.
5) local_signals are pooled across all cells and normalized to a peak value of 1.
6) f_pot represents the "sensitivity" of the forward process to the presence of the local_signal. The transformation
f_pot has the flexibility to be any segment of a sigmoid (so can be linear, exponential rise, or saturating).
7) f_dep represents the "sensitivity" of the reverse process to the presence of the local_signal. The transformation
f_dep has the flexibility to be any segment of a sigmoid (so can be linear, exponential rise, or saturating).

biBTSP_WD_D:
"""
__author__ = 'milsteina'
from biBTSP_utils import *
from nested.parallel import *
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
        raise IOError('simulate_biBTSP_synthetic_network: init_context: invalid data_file_path: %s' %
                      context.data_file_path)

    context.verbose = int(context.verbose)
    if 'plot' not in context():
        context.plot = False

    context.update(context.x0)

    with h5py.File(context.data_file_path, 'r') as f:
        dt = f['defaults'].attrs['dt']  # ms
        input_field_peak_rate = f['defaults'].attrs['input_field_peak_rate']  # Hz
        num_inputs = f['defaults'].attrs['num_inputs']
        track_length = f['defaults'].attrs['track_length']  # cm
        binned_dx = f['defaults'].attrs['binned_dx']  # cm
        generic_dx = f['defaults'].attrs['generic_dx']  # cm
        if 'default_run_vel' not in context() or context.default_run_vel is None:
            default_run_vel = f['defaults'].attrs['default_run_vel']  # cm/s
        else:
            context.default_run_vel = float(context.default_run_vel)
        generic_position_dt = f['defaults'].attrs['generic_position_dt']  # ms
        default_interp_dx = f['defaults'].attrs['default_interp_dx']  # cm
        binned_x = f['defaults']['binned_x'][:]
        generic_x = f['defaults']['generic_x'][:]
        generic_t = f['defaults']['generic_t'][:]
        default_interp_t = f['defaults']['default_interp_t'][:]
        default_interp_x = f['defaults']['default_interp_x'][:]
        extended_x = f['defaults']['extended_x'][:]

        if 'input_field_width' not in context() or context.input_field_width is None:
            raise RuntimeError('simulate_biBTSP_synthetic_network: init context: missing required parameter: '
                               'input_field_width')
        context.input_field_width = float(context.input_field_width)
        input_field_width_key = str(int(context.input_field_width))
        if 'calibrated_input' not in f or input_field_width_key not in f['calibrated_input']:
            raise RuntimeError('simulate_biBTSP_synthetic_network: init context: data for input_field_width: %.1f not '
                               'found in the provided data_file_path: %s' %
                               (float(context.input_field_width), context.data_file_path))
        input_field_width = f['calibrated_input'][input_field_width_key].attrs['input_field_width']  # cm
        input_rate_maps, peak_locs = \
            generate_spatial_rate_maps(binned_x, num_inputs, input_field_peak_rate, input_field_width, track_length)
        ramp_scaling_factor = f['calibrated_input'][input_field_width_key].attrs['ramp_scaling_factor']

    down_dt = 10.  # ms, to speed up optimization

    num_cells = int(context.num_cells)
    initial_fraction_active = context.initial_fraction_active
    initial_active_cells = int(num_cells * initial_fraction_active)
    basal_target_representation_density = context.basal_target_representation_density
    reward_target_representation_density = context.reward_target_representation_density

    initial_induction_dur = 300.  # ms
    pause_dur = 500.  # ms
    reward_dur = 500.  # ms
    plateau_dur = 300.  # ms
    peak_basal_plateau_prob_per_dt = \
        context.peak_basal_plateau_prob_per_sec / 1000. * dt
    peak_reward_plateau_prob_per_dt = \
        context.peak_reward_plateau_prob_per_sec / 1000. * dt

    context.update(locals())

    lap_start_times = [-len(default_interp_t) * dt, 0.]
    position = np.append(np.add(default_interp_x, -track_length), default_interp_x)
    t = np.append(np.add(default_interp_t, -len(default_interp_t) * dt), default_interp_t)
    running_position = track_length
    running_t = len(default_interp_t) * dt
    num_laps = 2

    reward_start_times = [None]
    for track_phase in context.track_phases:
        if track_phase['lap_type'] == 'reward':
            reward_loc = float(track_phase['reward_loc'])
        else:
            reward_loc = None
        for i in range(int(track_phase['num_laps'])):
            prev_len = len(t)
            lap_start_times.append(running_t)
            if reward_loc is not None:
                this_reward_start_index = np.where(default_interp_x >= reward_loc)[0]
                if len(this_reward_start_index) == 0:
                    raise RuntimeError('simulate_biBTSP_synthetic_network: invalid reward_loc: %.1f' % reward_loc)
                this_reward_start_index = len(t) - len(default_interp_t) + this_reward_start_index[0]
                reward_start_times.append(t[this_reward_start_index])
            else:
                reward_start_times.append(None)
            position = np.append(position, np.add(default_interp_x, running_position))
            t = np.append(t, np.add(default_interp_t, running_t))
            running_position += track_length
            running_t += len(default_interp_t) * dt
            num_laps += 1
    reward_start_times.append(None)

    complete_rate_maps = []

    for this_rate_map in input_rate_maps:
        interp_rate_map = np.interp(default_interp_x, binned_x, this_rate_map, period=track_length)
        this_complete_rate_map = np.array([])
        for lap in range(num_laps):
            this_complete_rate_map = np.append(this_complete_rate_map, interp_rate_map)
        complete_rate_maps.append(this_complete_rate_map)

    down_t = np.arange(t[0], t[-1] + down_dt / 2., down_dt)
    down_rate_maps = []
    for rate_map in complete_rate_maps:
        this_down_rate_map = np.interp(down_t, t, rate_map)
        down_rate_maps.append(this_down_rate_map)

    context.update(locals())

    local_signal_filter_t, local_signal_filter, global_filter_t, global_filter = \
        get_dual_exp_decay_signal_filters(context.local_signal_decay, context.global_signal_decay, context.down_dt)
    local_signals = get_local_signal_population(local_signal_filter,
                                                context.down_rate_maps / context.input_field_peak_rate)
    local_signal_peak = np.max(local_signals)
    local_signals /= local_signal_peak
    down_plateau_len = int(plateau_dur / down_dt)
    example_gate_len = max(down_plateau_len, 2 * len(global_filter_t))
    example_induction_gate = np.zeros(example_gate_len)
    example_induction_gate[:down_plateau_len] = 1.
    example_global_signal = get_global_signal(example_induction_gate, global_filter)
    global_signal_peak = np.max(example_global_signal)

    signal_xrange = np.linspace(0., 1., 10000)
    pot_rate = np.vectorize(scaled_single_sigmoid(
        context.f_pot_th, context.f_pot_th + context.f_pot_half_width, signal_xrange))
    dep_rate = np.vectorize(scaled_single_sigmoid(
        context.f_dep_th, context.f_dep_th + context.f_dep_half_width, signal_xrange))

    target_initial_induction_loc = -context.target_peak_shift
    target_initial_induction_stop_loc = target_initial_induction_loc + \
                                        initial_induction_dur / 1000. * context.default_run_vel
    target_initial_ramp = get_target_synthetic_ramp(target_initial_induction_loc, ramp_x=context.binned_x,
                                            track_length=context.track_length,
                                            target_peak_val=context.initial_ramp_peak_val * 1.4,
                                            target_min_val=0., target_asymmetry=context.target_asymmetry,
                                            target_peak_shift=context.target_peak_shift,
                                            target_ramp_width=context.target_ramp_width)
    spike_threshold = 2.  # mV
    f_I_slope = context.input_field_peak_rate / (context.initial_ramp_peak_val * 1.4 - spike_threshold)
    f_I = np.vectorize(lambda x: 0. if x < spike_threshold else (x - spike_threshold) * f_I_slope)
    # f_I = lambda x: x
    max_population_rate_sum = np.mean(f_I(target_initial_ramp)) * num_cells

    initial_weights_population = [np.ones_like(peak_locs) for _ in range(num_cells)]

    if initial_active_cells > 0:
        d_peak_indexes = int(len(peak_locs) / initial_active_cells)
        initial_peak_indexes = np.linspace(0, len(peak_locs), initial_active_cells, dtype=int, endpoint=False) + \
                               int(d_peak_indexes / 2)
        _, initial_delta_weights, _, _ = \
            get_delta_weights_LSA(target_initial_ramp, ramp_x=context.binned_x, input_x=context.binned_x,
                                  interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                  peak_locs=context.peak_locs, ramp_scaling_factor=context.ramp_scaling_factor,
                                  induction_start_loc=target_initial_induction_loc,
                                  induction_stop_loc=target_initial_induction_stop_loc,
                                  track_length=context.track_length, target_range=context.target_range,
                                  bounds=(context.min_delta_weight, context.peak_delta_weight),
                                  verbose=context.verbose, plot=context.plot)

        for i in range(initial_active_cells):
            roll_indexes = initial_peak_indexes[i]
            this_weights = np.add(np.roll(initial_delta_weights, roll_indexes), 1.)
            initial_weights_population[i] = this_weights

    ramp_xscale = np.linspace(0., 10., 10000)
    # plateau_prob_ramp_sensitivity_f = scaled_single_sigmoid(0., 4., ramp_xscale)
    plateau_prob_ramp_sensitivity_f = lambda x: x / 10.
    plateau_prob_ramp_sensitivity_f = np.vectorize(plateau_prob_ramp_sensitivity_f)

    basal_representation_xscale = np.linspace(0., basal_target_representation_density, 10000)
    basal_plateau_prob_f = \
        scaled_single_sigmoid(basal_target_representation_density,
                              basal_target_representation_density + basal_target_representation_density / 3.,
                              basal_representation_xscale, ylim=[peak_basal_plateau_prob_per_dt, 0.])
    basal_plateau_prob_f = np.vectorize(basal_plateau_prob_f)

    reward_representation_xscale = np.linspace(0., reward_target_representation_density, 10000)
    reward_delta_representation_density = reward_target_representation_density - basal_target_representation_density
    reward_plateau_prob_f = \
        scaled_single_sigmoid(reward_target_representation_density,
                              reward_target_representation_density + reward_delta_representation_density / 2.,
                              reward_representation_xscale,
                              ylim=[peak_reward_plateau_prob_per_dt, 0.])
    reward_plateau_prob_f = np.vectorize(reward_plateau_prob_f)
    context.update(locals())


def get_ramp(weights):
    """

    :param weights: array
    :return: array
    """
    delta_weights = np.subtract(weights, 1.)
    ramp, _ = get_model_ramp(delta_weights, ramp_x=context.binned_x, input_x=context.binned_x,
                             input_rate_maps=context.input_rate_maps,
                             ramp_scaling_factor=context.ramp_scaling_factor)
    return ramp


def get_population_representation_density(ramp_population):
    """

    :param ramp_population: list of array (like binned_x)
    :return: array (like default_interp_t)
    """
    binned_population_rate_sum = np.sum([context.f_I(this_ramp) for this_ramp in ramp_population], axis=0)
    population_rate_sum = np.interp(context.default_interp_x, context.binned_x, binned_population_rate_sum,
                                    period=context.track_length)
    population_representation_density = population_rate_sum / context.max_population_rate_sum
    return population_representation_density


def get_plateau_probability(ramp, population_representation_density, prev_plateau_start_times, lap):
    """

    :param ramp: array (like binned_x)
    :param population_representation_density: array (like default_interp_x)
    :param prev_plateau_start_times: list of float
    :param lap: int
    :return: array (like default_interp_x)
    """
    t_start_time = context.lap_start_times[lap]
    this_t = np.add(np.append(context.default_interp_t,
                              context.default_interp_t + len(context.default_interp_t) * context.dt),
                    t_start_time)
    interp_ramp = np.interp(context.default_interp_x, context.binned_x, ramp, period=context.track_length)
    interp_ramp = np.append(interp_ramp, interp_ramp)
    this_rep_density = np.append(population_representation_density, population_representation_density)
    plateau_prob_ramp_modulation = np.maximum(np.minimum(context.plateau_prob_ramp_sensitivity_f(interp_ramp), 1.), 0.)
    plateau_prob = np.maximum(np.minimum(context.basal_plateau_prob_f(this_rep_density), 1.), 0.)
    plateau_prob *= (1. + context.basal_plateau_prob_ramp_sensitivity * plateau_prob_ramp_modulation)
    if context.reward_start_times[lap] is not None:
        this_reward_start_time = context.reward_start_times[lap]
        this_reward_stop_time = this_reward_start_time + context.reward_dur
        reward_indexes = np.where((this_t >= this_reward_start_time) & (this_t < this_reward_stop_time))
        reward_plateau_prob = \
            np.maximum(np.minimum(context.reward_plateau_prob_f(this_rep_density[reward_indexes]), 1.), 0.)
        reward_plateau_prob *= \
            (1. + context.reward_plateau_prob_ramp_sensitivity * plateau_prob_ramp_modulation[reward_indexes])
        plateau_prob[reward_indexes] = reward_plateau_prob

    for this_plateau_start_time in prev_plateau_start_times:
        this_plateau_stop_time = this_plateau_start_time + context.plateau_dur + context.pause_dur
        if this_plateau_stop_time > t_start_time:
            plateau_indexes = np.where((this_t >= this_plateau_start_time) & (this_t < this_plateau_stop_time))
            plateau_prob[plateau_indexes] = 0.

    return plateau_prob


def get_plateau_times(plateau_prob, lap, cell_id):
    """

    :param plateau_prob: array
    :param lap: int
    :return: list of float
    """
    seed = context.seed_offset + 1e8 * int(context.trial) + 1e6 * lap + cell_id
    local_random = random.Random()
    local_random.seed(seed)

    t_start_time = context.lap_start_times[lap]
    this_t = np.add(np.append(context.default_interp_t,
                              context.default_interp_t + len(context.default_interp_t) * context.dt),
                    t_start_time)

    plateau_len = int(context.plateau_dur / context.dt)
    pause_len = int(context.pause_dur / context.dt)
    plateau_start_times = []
    i = 0
    while i < len(this_t):
        if plateau_prob[i] > 0. and local_random.random() < plateau_prob[i]:
            plateau_start_times.append(this_t[i])
            i += plateau_len + pause_len
        else:
            i += 1
    return plateau_start_times


def update_weights(current_weights, plateau_start_times, lap):
    """

    :param current_weights: array
    :param plateau_start_times: list of float
    :param lap: int
    :return: array
    """
    t_start_time = context.lap_start_times[lap]
    this_t = np.add(np.append(context.default_interp_t,
                              context.default_interp_t + len(context.default_interp_t) * context.dt),
                    t_start_time)
    plateau_len = int(context.plateau_dur / context.dt)
    if len(plateau_start_times) == 0:
        return current_weights
    induction_gate = np.zeros_like(this_t)
    for plateau_start_time in plateau_start_times:
        start_index = np.where(this_t >= plateau_start_time)[0][0]
        induction_gate[start_index:start_index+plateau_len] = 1.
    this_down_t = np.arange(this_t[0], this_t[-1] + context.down_dt/2., context.down_dt)
    down_induction_gate = np.interp(this_down_t, this_t, induction_gate)
    global_signal = np.minimum(get_global_signal(down_induction_gate, context.global_filter),
                               context.global_signal_peak)
    global_signal /= context.global_signal_peak
    local_signals = context.local_signals
    pot_rate = context.pot_rate
    dep_rate = context.dep_rate

    peak_weight = context.peak_delta_weight + 1.
    current_normalized_weights = np.divide(current_weights, peak_weight)

    start_index = np.where(context.down_t >= t_start_time)[0][0]
    stop_index = start_index + len(this_down_t)

    next_normalized_weights = []
    for i, this_local_signal in enumerate(local_signals):
        this_pot_rate = np.trapz(pot_rate(np.multiply(this_local_signal[start_index:stop_index], global_signal)),
                                 dx=context.down_dt / 1000.)
        this_dep_rate = np.trapz(dep_rate(np.multiply(this_local_signal[start_index:stop_index], global_signal)),
                                 dx=context.down_dt / 1000.)
        this_normalized_delta_weight = context.k_pot * this_pot_rate * (1. - current_normalized_weights[i]) - \
                                       context.k_dep * this_dep_rate * current_normalized_weights[i]
        this_next_normalized_weight = max(0., min(1., current_normalized_weights[i] + this_normalized_delta_weight))
        next_normalized_weights.append(this_next_normalized_weight)

    next_weights = np.multiply(next_normalized_weights, peak_weight)

    return next_weights


def simulate_network(export=False, plot=False):
    """

    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    current_weights_population = context.initial_weights_population
    weights_pop_history = []
    ramp_pop_history = []
    pop_rep_density_history = []
    prev_plateau_start_times = [[] for _ in range(context.num_cells)]
    plateau_start_times_history = [prev_plateau_start_times]

    for lap in range(1, context.num_laps - 1):
        weights_pop_history.append(current_weights_population)
        current_ramp_population = context.interface.map(get_ramp, current_weights_population)
        ramp_pop_history.append(current_ramp_population)
        current_pop_representation_density = \
            get_population_representation_density(current_ramp_population)
        pop_rep_density_history.append(current_pop_representation_density)

        sequences = [current_ramp_population] + [[current_pop_representation_density] * context.num_cells] + \
                    [prev_plateau_start_times] + [[lap] * context.num_cells]
        pop_plateau_probability = context.interface.map(get_plateau_probability, *sequences)

        sequences = [pop_plateau_probability] + [[lap] * context.num_cells] + [list(range(context.num_cells))]
        plateau_start_times = context.interface.map(get_plateau_times, *sequences)
        plateau_start_times_history.append(plateau_start_times)

        sequences = [current_weights_population] + [plateau_start_times] + [[lap] * context.num_cells]
        current_weights_population = context.interface.map(update_weights, *sequences)
        prev_plateau_start_times = plateau_start_times

    weights_pop_history.append(current_weights_population)
    current_ramp_population = context.interface.map(get_ramp, current_weights_population)
    ramp_pop_history.append(current_ramp_population)
    current_pop_representation_density = \
        get_population_representation_density(current_ramp_population)
    pop_rep_density_history.append(current_pop_representation_density)
    plateau_start_times_history.append([[] for _ in range(context.num_cells)])

    context.update(locals())

    if context.disp:
        print('simulate_biBTSP_synthetic_network: Process: %i: simulating network took %.1f s' %
              (os.getpid(), time.time() - start_time))
        sys.stdout.flush()

    if plot:
        plot_network_history(ramp_pop_history, pop_rep_density_history)

    if export:
        with h5py.File(context.export_file_path, 'a') as f:
            exported_data_key = 'biBTSP_synthetic_network_history'
            if exported_data_key in f:
                raise RuntimeError('simulate_biBTSP_synthetic_network: data has already been exported to '
                                   'export_file_path: %s' % context.export_file_path)
            group = f.create_group(exported_data_key)
            group.attrs['enumerated'] = False
            set_h5py_attr(group.attrs, 'data_file_name', context.data_file_name)
            set_h5py_attr(group.attrs, 'config_file_path', context.config_file_path)
            group.attrs['input_field_width'] = context.input_field_width
            group.attrs['min_delta_weight'] = context.min_delta_weight
            group.attrs['initial_ramp_peak_val'] = context.initial_ramp_peak_val
            group.attrs['target_peak_shift'] = context.target_peak_shift
            group.attrs['num_cells'] = int(context.num_cells)
            group.attrs['initial_fraction_active'] = context.initial_fraction_active
            group.attrs['basal_target_representation_density'] = context.basal_target_representation_density
            group.attrs['reward_target_representation_density'] = context.reward_target_representation_density
            group.attrs['peak_basal_plateau_prob_per_sec'] = context.peak_basal_plateau_prob_per_sec
            group.attrs['peak_reward_plateau_prob_per_sec'] = context.peak_reward_plateau_prob_per_sec
            group.attrs['basal_plateau_prob_ramp_sensitivity'] = context.basal_plateau_prob_ramp_sensitivity
            group.attrs['reward_plateau_prob_ramp_sensitivity'] = context.reward_plateau_prob_ramp_sensitivity
            group.attrs['seed_offset'] = int(context.seed_offset)
            group.attrs['trial'] = int(context.trial)

            group.create_dataset('weights_pop_history', compression='gzip', data=np.array(weights_pop_history))
            group.create_dataset('ramp_pop_history', compression='gzip', data=np.array(ramp_pop_history))
            group.create_dataset('pop_rep_density_history', compression='gzip', data=np.array(pop_rep_density_history))
            group = group.create_group('plateau_start_times_history')
            for i, lap_plateau_start_times in enumerate(plateau_start_times_history):
                if np.any([len(plateau_start_times) > 0 for plateau_start_times in lap_plateau_start_times]):
                    lap_key = str(i)
                    group.create_group(lap_key)
                    for cell_id, plateau_start_times in enumerate(lap_plateau_start_times):
                        if len(plateau_start_times) > 0:
                            group[lap_key].create_dataset(str(cell_id), compression='gzip',
                                                          data=np.array(plateau_start_times))

        if context.disp:
            print('simulate_biBTSP_synthetic_network: Process: %i: exported data to file: %s' %
                  (os.getpid(), context.export_file_path))
            sys.stdout.flush()


def plot_network_history(ramp_pop_history, pop_rep_density_history):
    """

    :param ramp_pop_history: 3D array
    :param pop_rep_density_history: 2D array
    """
    import matplotlib as mpl
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['font.size'] = 11.
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['axes.titlepad'] = 2.
    mpl.rcParams['mathtext.default'] = 'regular'

    reward_loc_by_track_phase = dict()
    colors = dict()

    track_phase_summary = [(0, 0, 'before')]
    lap_num = 0
    for track_phase in context.track_phases:
        this_num_laps = int(track_phase['num_laps'])
        start_lap = lap_num + 1
        lap_num += this_num_laps
        end_lap = lap_num
        label = track_phase['label']
        colors[label] = track_phase['color']
        if track_phase['lap_type'] == 'reward':
            reward_loc_by_track_phase[label] = float(track_phase['reward_loc'])
        track_phase_summary.append((start_lap, end_lap, label))

    initial_loc_by_cell = dict()
    initial_locs_by_lap = []
    initial_locs_by_track_phase = defaultdict(list)
    initial_ramp_by_track_phase = defaultdict(list)
    start_locs_by_track_phase = defaultdict(dict)
    end_locs_by_track_phase = defaultdict(dict)
    delta_locs_by_track_phase = defaultdict(list)
    delta_reward_distance_by_track_phase = defaultdict(lambda: defaultdict(list))

    this_lap_locs = dict()
    for start_lap, end_lap, label in track_phase_summary:
        lap = start_lap
        start_locs_by_track_phase[label] = this_lap_locs
        while lap <= end_lap:
            this_lap_initial_locs = []
            this_lap_locs = dict()
            ramp_pop = ramp_pop_history[lap]
            for cell_index, ramp in enumerate(ramp_pop):
                this_peak_index = np.argmax(ramp)
                this_peak_loc = context.binned_x[this_peak_index]
                if np.max(ramp) - np.min(ramp) > 2.:
                    if cell_index not in initial_loc_by_cell:
                        initial_loc_by_cell[cell_index] = this_peak_loc
                        this_lap_initial_locs.append(this_peak_loc)
                    this_lap_locs[cell_index] = this_peak_loc
                if lap == start_lap:
                    initial_ramp_by_track_phase[label].append(ramp)
            initial_locs_by_lap.append(this_lap_initial_locs)
            initial_locs_by_track_phase[label].extend(this_lap_initial_locs)
            lap += 1
        end_locs_by_track_phase[label] = this_lap_locs
    for label in end_locs_by_track_phase:
        for cell_index in end_locs_by_track_phase[label]:
            if cell_index in start_locs_by_track_phase[label]:
                this_delta_loc = get_circular_distance(start_locs_by_track_phase[label][cell_index],
                                                       end_locs_by_track_phase[label][cell_index], context.track_length)
            elif cell_index in initial_loc_by_cell:
                this_delta_loc = get_circular_distance(initial_loc_by_cell[cell_index],
                                                       end_locs_by_track_phase[label][cell_index], context.track_length)
            if this_delta_loc < -1. or this_delta_loc > 1.:
                delta_locs_by_track_phase[label].append(this_delta_loc)
            for reward_loc in reward_loc_by_track_phase.values():
                end_reward_distance = abs(get_circular_distance(end_locs_by_track_phase[label][cell_index],
                                                                reward_loc, context.track_length))
                if cell_index in start_locs_by_track_phase[label]:
                    initial_reward_distance = abs(get_circular_distance(start_locs_by_track_phase[label][cell_index],
                                                                        reward_loc, context.track_length))
                elif cell_index in initial_loc_by_cell:
                    initial_reward_distance = abs(get_circular_distance(initial_loc_by_cell[cell_index],
                                                                        reward_loc, context.track_length))
                this_delta_reward_distance = end_reward_distance - initial_reward_distance
                if this_delta_reward_distance < -1. or this_delta_reward_distance > 1.:
                    delta_reward_distance_by_track_phase[reward_loc][label].append(this_delta_reward_distance)

    fig1, axes1 = plt.subplots(3, 3, figsize=(11, 9))
    this_cmap = 'viridis'

    norm_basal_plateau_prob = context.basal_plateau_prob_f(context.basal_representation_xscale) / context.dt * 1000.
    norm_reward_plateau_prob = context.reward_plateau_prob_f(context.reward_representation_xscale) / context.dt * 1000.
    axes1[0][1].plot(context.basal_representation_xscale, norm_basal_plateau_prob, label='Outside goal', c='k')
    axes1[0][1].plot(context.reward_representation_xscale, norm_reward_plateau_prob, label='Inside goal', c='r')
    axes1[0][1].set_xticks([i * 0.25 for i in range(5)])
    axes1[0][1].set_xlim(0., 1.)
    axes1[0][1].set_ylim(0., axes1[0][1].get_ylim()[1])
    axes1[0][1].set_xlabel('Normalized population activity')
    axes1[0][1].set_ylabel('Plateau probability / sec')
    axes1[0][1].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
    axes1[0][1].set_title('Modulation of plateau probability\nby feedback inhibition\nand instructive input',
                          fontsize=mpl.rcParams['font.size'], y=1.1)

    plateau_prob_ramp_modulation = context.plateau_prob_ramp_sensitivity_f(context.ramp_xscale)
    axes1[0][0].plot(context.ramp_xscale, plateau_prob_ramp_modulation *
                     context.basal_plateau_prob_ramp_sensitivity + 1., label='No reward', c='k')
    axes1[0][0].plot(context.ramp_xscale, plateau_prob_ramp_modulation *
                     context.reward_plateau_prob_ramp_sensitivity + 1., label='Reward', c='r')
    axes1[0][0].set_xlim(0., 10.)
    axes1[0][0].set_xlabel('Ramp amplitude (mV)')
    axes1[0][0].set_ylabel('Normalized\nplateau probability')
    axes1[0][0].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
    axes1[0][0].set_title('Modulation of plateau probability\nby ramp amplitude',
                          fontsize=mpl.rcParams['font.size'], y=1.1)

    bins = 20
    for label in delta_locs_by_track_phase:
        edges = np.linspace(0., context.track_length, bins)
        bin_width = edges[1] - edges[0]
        hist, _ = np.histogram(initial_locs_by_track_phase[label], bins=edges)
        axes1[2][0].bar(edges[:-1] + bin_width / 2., hist, label=label.capitalize(), alpha=0.5, width=bin_width,
                        color=colors[label])
        edges = np.linspace(-context.track_length / 2., context.track_length / 2., bins)
        bin_width = edges[1] - edges[0]
        hist, _ = np.histogram(delta_locs_by_track_phase[label], bins=edges)
        axes1[2][1].bar(edges[:-1] + bin_width / 2., hist, label=label.capitalize(), alpha=0.5, width=bin_width,
                        color=colors[label])
        for reward_loc in delta_reward_distance_by_track_phase:
            hist, _ = np.histogram(delta_reward_distance_by_track_phase[reward_loc][label], bins=edges)
            axes1[2][2].bar(edges[:-1] + bin_width / 2., hist, label=label.capitalize(), alpha=0.5, width=bin_width,
                            color=colors[label])
    axes1[2][0].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, fontsize=mpl.rcParams['font.size'])
    axes1[2][0].set_ylabel('Cell count')
    axes1[2][0].set_xlabel('Location (cm)')
    axes1[2][0].set_xlim(0., context.track_length)
    axes1[2][0].set_xticks(np.arange(0., context.track_length, 45.))
    axes1[2][0].set_title('Emergence of new place fields', fontsize=mpl.rcParams['font.size'], y=1.1)

    axes1[2][1].set_ylabel('Cell count')
    axes1[2][1].set_xlabel('Change in location (cm)')
    axes1[2][1].set_title('Place field translocation', fontsize=mpl.rcParams['font.size'], y=1.1)
    axes1[2][1].set_xticks(np.arange(-90., 91., 45.))

    axes1[2][2].set_ylabel('Cell count')
    axes1[2][2].set_xlabel('Change in distance to reward (cm)')
    axes1[2][2].set_xticks(np.arange(-90., 91., 45.))
    axes1[2][2].set_title('Translocation towards reward location', fontsize=mpl.rcParams['font.size'], y=1.1)

    X, Y = np.meshgrid(context.default_interp_x, range(len(pop_rep_density_history) + 1))
    hm = axes1[0][2].pcolormesh(X, Y, pop_rep_density_history, cmap=this_cmap, edgecolors='face')  #, vmin=0., vmax=1.)
    cb = plt.colorbar(hm, ax=axes1[0][2])
    cb.ax.set_ylabel('Normalized\npopulation activity', rotation=270)
    cb.ax.get_yaxis().labelpad = 25
    axes1[0][2].set_ylim(len(pop_rep_density_history), 0)
    yticks = sorted(list(range(0, len(pop_rep_density_history), 5)), reverse=True)
    axes1[0][2].set_yticks(np.add(yticks, 1))
    axes1[0][2].set_yticklabels(yticks)
    axes1[0][2].set_ylabel('Lap #')
    axes1[0][2].set_xlabel('Location (cm)')
    axes1[0][2].set_xlim(0., context.track_length)
    axes1[0][2].set_xticks(np.arange(0., context.track_length, 45.))
    axes1[0][2].set_title('Summed population activity', fontsize=mpl.rcParams['font.size'], y=1.1)
    clean_axes(axes1)
    fig1.subplots_adjust(left=0.075, hspace=0.5, wspace=0.6, right=0.925)
    fig1.show()

    max_count = len(initial_loc_by_cell)
    prev_sorted_cell_indexes = None
    for start_lap, end_lap, label in track_phase_summary:
        fig2, axes2 = plt.subplots(3, 3, figsize=(11, 9))
        ramp_pop = ramp_pop_history[end_lap]
        cell_indexes = list(initial_loc_by_cell.keys())
        max_index = []
        for cell_index in cell_indexes:
            ramp = ramp_pop[cell_index]
            max_index.append(np.argmax(ramp))
        sorted_cell_indexes = [cell_index for (_, cell_index) in sorted(zip(max_index, cell_indexes))]
        modulated_cell_indexes = []
        nonmodulated_cell_indexes = []
        modulated_ramp_pop = []
        nonmodulated_ramp_pop = []
        for i in sorted_cell_indexes:
            this_ramp = ramp_pop[i]
            if np.max(this_ramp) - np.min(this_ramp) > 2.:
                modulated_ramp_pop.append(context.f_I(this_ramp))
                modulated_cell_indexes.append(i)
            else:
                nonmodulated_ramp_pop.append(context.f_I(this_ramp))
                nonmodulated_cell_indexes.append(i)
        sorted_ramp_pop = modulated_ramp_pop + nonmodulated_ramp_pop
        X, Y = np.meshgrid(context.binned_x, range(len(sorted_ramp_pop)))
        hm = axes2[0][2].pcolormesh(X, Y, sorted_ramp_pop, cmap=this_cmap, vmin=0., vmax=context.input_field_peak_rate,
                                    edgecolors='face')
        cb = plt.colorbar(hm, ax=axes2[0][2])
        cb.ax.set_ylabel('Firing rate (Hz)', rotation=270, fontsize=mpl.rcParams['font.size'])
        cb.ax.get_yaxis().labelpad = 15
        axes2[0][2].set_ylim(len(sorted_ramp_pop) - 1, 0)
        axes2[0][2].set_ylabel('Sorted cell #', fontsize=mpl.rcParams['font.size'])
        axes2[0][2].set_xlabel('Location (cm)', fontsize=mpl.rcParams['font.size'])
        axes2[0][2].set_xlim(0., context.track_length)
        axes2[0][2].set_xticks(np.arange(0., context.track_length, 45.))
        axes2[0][2].set_title('Sorted after %s' % label, fontsize=mpl.rcParams['font.size'], y=1.1)
        if prev_sorted_cell_indexes is not None:
            prev_sorted_delta_rate = []
            for i in prev_sorted_cell_indexes:
                this_initial_rate = context.f_I(initial_ramp_by_track_phase[label][i])
                this_rate = context.f_I(ramp_pop[i])
                this_delta_rate = this_rate - this_initial_rate
                prev_sorted_delta_rate.append(this_delta_rate)
            X, Y = np.meshgrid(context.binned_x, range(len(prev_sorted_delta_rate)))
            hm = axes2[0][1].pcolormesh(X, Y, prev_sorted_delta_rate, vmin=-context.input_field_peak_rate,
                                        vmax=context.input_field_peak_rate, cmap='seismic', edgecolors='face') # cmap='bwr',
            cb = plt.colorbar(hm, ax=axes2[0][1])
            cb.ax.set_ylabel('Change in firing rate (Hz)', rotation=270, fontsize=mpl.rcParams['font.size'])
            cb.ax.get_yaxis().labelpad = 15
            axes2[0][1].set_ylim(len(prev_sorted_delta_rate) - 1, 0)
            axes2[0][1].set_ylabel('Sorted cell #', fontsize=mpl.rcParams['font.size'])
            axes2[0][1].set_xlabel('Location (cm)', fontsize=mpl.rcParams['font.size'])
            axes2[0][1].set_xlim(0., context.track_length)
            axes2[0][1].set_xticks(np.arange(0., context.track_length, 45.))
            axes2[0][1].set_title('Sorted before %s' % label, fontsize=mpl.rcParams['font.size'], y=1.1)
        prev_sorted_cell_indexes = modulated_cell_indexes + nonmodulated_cell_indexes
        clean_axes(axes2)
        if end_lap == 0:
            fig2.suptitle('Before (lap 0)', fontsize=mpl.rcParams['font.size'])
        else:
            fig2.suptitle('After %s (laps %i - %i)' % (label, start_lap, end_lap), fontsize=mpl.rcParams['font.size'])
        fig2.subplots_adjust(left=0.075, hspace=0.5, wspace=0.6, right=0.925)
        fig2.show()

    fig3, axes3 = plt.subplots()
    for lap, pop_rep_density in enumerate(pop_rep_density_history):
        axes3.plot(context.default_interp_x, pop_rep_density)
    axes3.set_ylabel('Normalized population activity')
    axes3.set_xlabel('Location (cm)')
    clean_axes(axes3)
    fig3.show()

    context.update(locals())


def plot_model_summary(model_file_path):
    """

    :param model_file_path: str (path)
    """
    if not os.path.isfile(model_file_path):
        raise IOError('plot_model_summary: invalid model_file_path: %s' % model_file_path)
    exported_data_key = 'biBTSP_synthetic_network_history'
    with h5py.File(model_file_path, 'r') as f:
        if exported_data_key not in f:
            raise KeyError('plot_model_summary: invalid file contents at path: %s' % model_file_path)
        group = f[exported_data_key]
        loaded_config_file_path = get_h5py_attr(group.attrs, 'config_file_path')
        loaded_seed_offset = group.attrs['seed_offset']
        loaded_trial = group.attrs['trial']
        loaded_num_cells = group.attrs['num_cells']
        if not all([loaded_config_file_path == context.config_file_path,
                    loaded_seed_offset == int(context.seed_offset),
                    loaded_trial == int(context.trial),
                    loaded_num_cells == int(context.num_cells)]):
            raise RuntimeError('simulate_biBTSP_synethic_network: plot_model_summary: configuration loaded from '
                               'config_file_path: %s is inconsistent with data loaded from model_file_path: %s' %
                               (context.config_file_path, model_file_path))
        weights_pop_history = group['weights_pop_history'][:]
        ramp_pop_history = group['ramp_pop_history'][:]
        pop_rep_density_history = group['pop_rep_density_history'][:]
        plateau_start_times_history = []
        group = group['plateau_start_times_history']
        num_laps = len(pop_rep_density_history)
        for i in range(num_laps):
            lap_key = str(i)
            if lap_key in group:
                lap_plateau_start_times = []
                for cell_id in range(loaded_num_cells):
                    cell_key = str(cell_id)
                    if cell_key in group[lap_key]:
                        lap_plateau_start_times.append(group[lap_key][cell_key][:])
                    else:
                        lap_plateau_start_times.append([])
            else:
                lap_plateau_start_times = [[] for _ in range(loaded_num_cells)]
            plateau_start_times_history.append(lap_plateau_start_times)

    plot_network_history(ramp_pop_history, pop_rep_density_history)
    context.update(locals())


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/simulate_biBTSP_synthetic_network_WD_D_config.yaml')
@click.option("--trial", type=int, default=0)
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default='biBTSP_synthetic_network_WD_D')
@click.option("--verbose", type=int, default=2)
@click.option("--plot", is_flag=True)
@click.option("--interactive", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--plot-summary-figure", is_flag=True)
@click.option("--model-file-path", type=str, default=None)
@click.pass_context
def main(cli, config_file_path, trial, output_dir, export, export_file_path, label, verbose, plot, interactive, debug,
         plot_summary_figure, model_file_path):
    """
    To execute on a single process:
    python -i simulate_biBTSP_synthetic_network_WD_D.py --plot --framework=serial --interactive

    To execute using MPI parallelism with 1 controller process and N - 1 worker processes:
    mpirun -n N python -i -m mpi4py.futures simulate_biBTSP_synthetic_network_WD_D.py --plot --framework=mpi --interactive

    To plot results previously exported to a file on a single process:
    python -i simulate_biBTSP_synthetic_network_WD_D.py --plot-summary-figure --model-file-path=$PATH_TO_MODEL_FILE \
        --framework=serial --interactive

    :param cli: contains unrecognized args as list of str
    :param config_file_path: str (path)
    :param trial: int
    :param output_dir: str (path)
    :param export: bool
    :param export_file_path: str
    :param label: str
    :param verbose: int
    :param plot: bool
    :param interactive: bool
    :param debug: bool
    :param plot_summary_figure: bool
    :param model_file_path: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    kwargs = get_unknown_click_arg_dict(cli.args)
    context.disp = verbose > 0

    context.interface = get_parallel_interface(source_file=__file__, source_package=__package__, **kwargs)
    context.interface.start(disp=context.disp)
    context.interface.ensure_controller()
    config_parallel_interface(__file__, config_file_path=config_file_path, output_dir=output_dir, export=export,
                              export_file_path=export_file_path, label=label, disp=context.disp,
                              interface=context.interface, verbose=verbose, plot=plot, **kwargs)
    if not context.interface.controller_is_worker:
        config_worker()
    if plot_summary_figure:
        plot_model_summary(model_file_path)
        plot = True
    elif not debug:
        simulate_network(export, plot)
    if plot:
        context.interface.apply(plt.show)
        plt.show()

    if context.interactive:
        context.update(locals())
    else:
        context.interface.stop()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
