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
3) Dendritic plateaus generate a long duration 'global plasticity signal', or a 'gating trace' for synaptic plasticity.
4) Changes in weight at each synapse are integrated over periods of nonzero overlap between eligibility and gating
signals, and updated once per lap.

Features/assumptions of synaptic resource-limited model B:
1) Dendritic plateaus generate a global gating signal that provides a necessary cofactor required to convert plasticity
eligibility signals at each synapse into either increases or decreases in synaptic strength.
2) Activity at each synapse generates a local plasticity eligibility signal that, in conjunction with the global
gating signal, can activate both a forward process to increase synaptic strength and a reverse process to decrease
synaptic strength.
3) Synaptic resources can be in 2 states (Markov-style kinetic scheme):

        k_pot * global_signal * f_pot(local_signal)
I (inactive) <------------------------------> A (active)
        k_dep * global_signal * f_dep(local_signal)

4) global_signals are pooled across all cells and normalized to a peak value of 1.
5) local_signals are pooled across all cells and normalized to a peak value of 1.
6) f_pot represents the "sensitivity" of the forward process to the presence of the local_signal. The transformation
f_pot has the flexibility to be any segment of a sigmoid (so can be linear, exponential rise, or saturating).
7) f_dep represents the "sensitivity" of the reverse process to the presence of the local_signal. The transformation
f_dep has the flexibility to be any segment of a sigmoid (so can be linear, exponential rise, or saturating).

biBTSP_SRL_B: Single eligibility signal filter. Sigmoidal f_pot and f_dep.
"""
__author__ = 'milsteina'
from BTSP_utils import *
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
            raise RuntimeError('simulate_biBTSP_synthetic_network: init context: data for input_field_width: %.1f not found in the '
                               'provided data_file_path: %s' %
                               (float(context.input_field_width), context.data_file_path))
        input_field_width = f['calibrated_input'][input_field_width_key].attrs['input_field_width']  # cm
        input_rate_maps, peak_locs = \
            generate_spatial_rate_maps(binned_x, num_inputs, input_field_peak_rate, input_field_width, track_length)
        ramp_scaling_factor = f['calibrated_input'][input_field_width_key].attrs['ramp_scaling_factor']

    down_dt = 10.  # ms, to speed up optimization

    num_cells = context.num_cells
    initial_fraction_active = context.initial_fraction_active
    initial_active_cells = int(num_cells * initial_fraction_active)
    basal_target_representation_density = context.basal_target_representation_density
    reward_target_representation_density = context.reward_target_representation_density

    initial_induction_dur = 300.  # ms
    pause_dur = 500.  # ms
    reward_dur = 500.  # ms
    plateau_dur = 300.  # ms
    peak_basal_plateau_prob_per_dt = \
        context.peak_basal_plateau_prob_per_lap / len(default_interp_t)
    peak_reward_plateau_prob_per_dt = \
        context.peak_reward_plateau_prob_per_lap / int(reward_dur / dt)

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
        get_dual_signal_filters(context.local_signal_rise, context.local_signal_decay, context.global_signal_rise,
                                context.global_signal_decay, context.down_dt)
    local_signals = get_local_signal_population(local_signal_filter, context.down_rate_maps, context.down_dt)
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
        context.f_pot_th, context.f_pot_th + context.f_pot_peak, signal_xrange))
    dep_rate = np.vectorize(scaled_single_sigmoid(
        context.f_dep_th, context.f_dep_th + context.f_dep_peak, signal_xrange))

    target_initial_induction_loc = -context.target_peak_shift
    target_initial_induction_stop_loc = target_initial_induction_loc + \
                                        initial_induction_dur / 1000. * context.default_run_vel
    target_initial_ramp = \
        get_target_synthetic_ramp(target_initial_induction_loc, ramp_x=context.binned_x,
                                  track_length=context.track_length, target_peak_val=context.initial_ramp_peak_val,
                                  target_min_val=0., target_asymmetry=1.8, target_peak_shift=context.target_peak_shift,
                                  target_ramp_width=187.)
    max_ramp_population_sum = np.mean(target_initial_ramp) * num_cells

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
    plateau_prob_ramp_sensitivity_f = scaled_single_sigmoid(0., 4., ramp_xscale)

    basal_representation_xscale = np.linspace(0., basal_target_representation_density, 10000)
    basal_plateau_prob_f = \
        scaled_single_sigmoid(basal_target_representation_density, basal_target_representation_density / 2.,
                              basal_representation_xscale, ylim=[peak_basal_plateau_prob_per_dt, 0.])
    basal_plateau_prob_f = np.vectorize(basal_plateau_prob_f)

    reward_representation_xscale = np.linspace(0., reward_target_representation_density, 10000)
    reward_delta_representation_density = reward_target_representation_density - basal_target_representation_density
    reward_plateau_prob_f = \
        scaled_single_sigmoid(reward_target_representation_density,
                              basal_target_representation_density / 2. + reward_delta_representation_density,
                              reward_representation_xscale,
                              ylim=[peak_reward_plateau_prob_per_dt, 0.])
    reward_plateau_prob_f = np.vectorize(reward_plateau_prob_f)
    context.update(locals())


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
    seed = context.seed_offset + 1e8 * context.trial + 1e6 * lap + cell_index
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
    for j in range(start_index, len(this_lap_t)):
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
        for i in range(len(context.peak_locs)):
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

    next_global_signal = global_signal[len(context.default_interp_t):]

    if context.disp:
        print('Process: %i: calculating ramp for cell: %i, lap: %i took %.1f s' %
              (os.getpid(), cell_index, lap, time.time() - start_time))

    return plateau_start_times, plateau_stop_times, next_global_signal, weights_history


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
    binned_ramp_population_sum = np.sum(ramp_population, axis=0)
    ramp_population_sum = np.interp(context.default_interp_x, context.binned_x, binned_ramp_population_sum,
                                    period=context.track_length)
    population_representation_density = ramp_population_sum / context.max_ramp_population_sum
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
    interp_ramp = np.interp(context.default_interp_x, context.binned_x, ramp)
    interp_ramp = np.append(interp_ramp, interp_ramp)
    this_rep_density = np.append(population_representation_density, population_representation_density)
    plateau_prob_ramp_modulation = context.plateau_prob_ramp_sensitivity_f(interp_ramp)
    plateau_prob = context.basal_plateau_prob_f(this_rep_density) * \
                   (1. + context.basal_plateau_prob_ramp_sensitivity * plateau_prob_ramp_modulation)
    if context.reward_start_times[lap] is not None:
        this_reward_start_time = context.reward_start_times[lap]
        this_reward_stop_time = this_reward_start_time + context.reward_dur
        reward_indexes = np.where((this_t >= this_reward_start_time) & (this_t < this_reward_stop_time))
        plateau_prob[reward_indexes] = \
            context.reward_plateau_prob_f(this_rep_density[reward_indexes]) * \
            (1. + context.reward_plateau_prob_ramp_sensitivity * plateau_prob_ramp_modulation[reward_indexes])
    for this_plateau_start_time in prev_plateau_start_times:
        this_plateau_stop_time = this_plateau_start_time + context.plateau_dur + context.pause_dur
        if this_plateau_stop_time > t_start_time:
            plateau_indexes = np.where((this_t >= this_plateau_start_time) & (this_t < this_plateau_stop_time))
            plateau_prob[plateau_indexes] = 0.
    plt.plot(this_t, plateau_prob)
    plt.show()
    return plateau_prob


def get_plateau_times(plateau_prob, lap, cell_id):
    """

    :param plateau_prob: array
    :param lap: int
    :return: list of float
    """
    seed = context.seed_offset + 1e8 * context.trial + 1e6 * lap + cell_id
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


def simulate_network(export=False, plot=False):
    """

    :param export: bool
    :param plot: bool
    :return: dict
    """
    current_weights_population = context.initial_weights_population
    ramp_pop_history = []
    pop_rep_density_history = []
    prev_plateau_start_times = [[] for _ in range(context.num_cells)]
    plateau_start_times_history = [prev_plateau_start_times]
    for lap in range(1, context.num_laps - 1):
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
    current_ramp_population = context.interface.map(get_ramp, current_weights_population)
    ramp_pop_history.append(current_ramp_population)
    current_pop_representation_density = \
        get_population_representation_density(current_ramp_population)
    pop_rep_density_history.append(current_pop_representation_density)
    plateau_start_times_history.append([[] for _ in range(context.num_cells)])


def simulate_network_orig(export=False, plot=False):
    """

    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    cell_indexes = list(range(len(context.initial_weights_population)))
    if 'debug' in context() and context.debug:
        cell_indexes = cell_indexes[::int(len(context.initial_weights_population) / context.interface.num_workers)]
        cell_indexes = cell_indexes[:context.interface.num_workers]
        print('cell_indexes: %s' % cell_indexes)
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
    current_ramp_population = list(map(get_model_ramp, current_weights_population))
    ramp_snapshots.append(current_ramp_population)
    initial_global_signal_population = [np.zeros_like(context.default_interp_t)] * group_size

    for lap in range(1, context.num_laps - 1):
        lap_start_index, lap_end_index = context.lap_edge_indexes[lap], context.lap_edge_indexes[lap + 1]
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
        this_weights_population_history = list(zip(*result))
        weights_population_full_history.append(np.array(this_weights_population_history))
        plateau_probability_history.append(current_plateau_probability_population)
        for cell_index, this_plateau_start_times, this_plateau_stop_times in \
                zip(cell_indexes, this_plateau_start_times_population, this_plateau_stop_times_population):
            plateau_start_times_population[cell_index].extend(this_plateau_start_times)
            plateau_stop_times_population[cell_index].extend(this_plateau_stop_times)
        current_weights_population = []
        for this_weights_history in this_weights_population_history:
            current_weights_population.append(this_weights_history[:, -1])
        weights_snapshots.append(current_weights_population)
        current_ramp_population = list(map(get_model_ramp, current_weights_population))
        ramp_snapshots.append(current_ramp_population)

    population_representation_density = get_population_representation_density(current_ramp_population)
    population_representation_density_history.append(population_representation_density)
    population_representation_density_history = np.array(population_representation_density_history)
    weights_population_full_history = np.array(weights_population_full_history)
    weights_snapshots = np.array(ramp_snapshots)
    ramp_snapshots = np.array(ramp_snapshots)

    if context.disp:
        print('Process: %i: calculating ramp population took %.1f s' % (os.getpid(), time.time() - start_time))

    reward_locs_array = [context.reward_locs[induction] for induction in context.reward_locs]

    if plot:
        plot_population_history_snapshots(ramp_snapshots, population_representation_density_history, reward_locs_array,
                                          context.binned_x, context.default_interp_x, context.track_length,
                                          context.num_baseline_laps, context.num_assay_laps, context.num_reward_laps)

    plateau_start_times_array = []
    plateau_stop_times_array = []
    plateau_times_cell_indexes = []
    for cell_index, this_plateau_start_times in viewitems(plateau_start_times_population):
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
                group.create_dataset('default_interp_x', compression='gzip', data=context.default_interp_x)
                group.create_dataset('input_rate_maps', compression='gzip', data=np.array(context.input_rate_maps))
                group.attrs['track_length'] = context.track_length
                group.attrs['dt'] = context.dt
            exported_data_key = 'BTSP_population_history'
            if exported_data_key not in f:
                f.create_group(exported_data_key)
                f[exported_data_key].attrs['enumerated'] = True
            group_id = str(len(f[exported_data_key]))
            f[exported_data_key].create_group(group_id)
            group = f[exported_data_key][group_id]
            group.attrs['num_baseline_laps'] = context.num_baseline_laps
            group.attrs['num_assay_laps'] = context.num_assay_laps
            group.attrs['num_reward_laps'] = context.num_reward_laps
            group.attrs['plateau_dur'] = context.plateau_dur
            group.attrs['reward_dur'] = context.reward_dur
            group.attrs['reward_locs_array'] = reward_locs_array
            group.attrs['ramp_scaling_factor'] = context.ramp_scaling_factor
            group.attrs['num_cells'] = context.num_cells
            group.attrs['initial_fraction_active'] = context.initial_fraction_active
            group.attrs['initial_active_cells'] = context.initial_active_cells
            group.attrs['basal_target_representation_density'] = context.basal_target_representation_density
            group.attrs['reward_target_representation_density'] = context.reward_target_representation_density
            group.attrs['peak_basal_prob_new_recruitment'] = context.peak_basal_prob_new_recruitment
            group.attrs['peak_reward_prob_new_recruitment'] = context.peak_reward_prob_new_recruitment
            group.attrs['peak_basal_plateau_prob_per_lap'] = context.peak_basal_plateau_prob_per_lap
            group.attrs['peak_reward_plateau_prob_per_lap'] = context.peak_reward_plateau_prob_per_lap
            group.attrs['random_seed_offset'] = context.seed_offset
            group.attrs['trial'] = context.trial
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
            print('Process: %i: exported weights population history data to file: %s' %
                  (os.getpid(), context.export_file_path))

    return plateau_start_times_population, plateau_stop_times_population, weights_population_full_history, \
           plateau_probability_history, weights_snapshots, ramp_snapshots


def plot_population_history_snapshots(ramp_snapshots, population_representation_density_history, reward_locs_array,
                                      binned_x, default_interp_x, track_length, num_baseline_laps, num_assay_laps,
                                      num_reward_laps, trial):
    """

    :param ramp_snapshots: 3D array
    :param population_representation_density_history: 2D array
    :param reward_locs_array: array
    :param binned_x: array
    :param default_interp_x: array
    :param track_length: float
    :param num_baseline_laps: int
    :param num_assay_laps: int
    :param num_reward_laps: int
    :param trial: int
    """
    import matplotlib as mpl
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['font.size'] = 11.
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['axes.titlepad'] = 2.
    mpl.rcParams['mathtext.default'] = 'regular'

    snapshot_laps = [0, num_baseline_laps]
    summary_laps = [0]
    if len(reward_locs_array) > 0:
        lap_labels = ['Familiar environment:\nBefore', 'After laps 1-%i:\nNo reward' % num_baseline_laps]
        summary_lap_labels = ['Familiar environment:\nBefore']
        for reward_loc in reward_locs_array:
            start = max(snapshot_laps)
            stop = start + num_reward_laps
            snapshot_laps.append(stop)
            summary_laps.append(stop)
            lap_labels.append('After laps %i-%i: Reward at %i cm' % (start + 1, stop, reward_loc))
            summary_lap_labels.append('After laps %i-%i:\nReward at %i cm' % (start + 1, stop, reward_loc))
            start = max(snapshot_laps)
            stop = start + num_assay_laps
            snapshot_laps.append(stop)
            lap_labels.append('After laps %i-%i: No reward' % (start + 1, stop))
    else:
        summary_laps.append(num_baseline_laps)
        lap_labels = ['Novel environment:\nBefore', 'After laps 1-%i:\nNo reward' % num_baseline_laps]
        summary_lap_labels = list(lap_labels)

    print('Trial: %i' % trial)
    pprint.pprint(lap_labels)

    peak_loc_history = defaultdict(list)
    delta_peak_loc_history = defaultdict(list)
    active_cell_index_set = set()

    num_cells = len(ramp_snapshots[0])
    num_laps = len(ramp_snapshots)
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

    this_peak_shift_count = 0
    for lap in range(1, snapshot_laps[-1] + 1, 1):
        for cell_index, this_delta_peak_loc_history in viewitems(delta_peak_loc_history):
            this_delta_peak_loc = this_delta_peak_loc_history[lap - 1]
            if not np.isnan(this_delta_peak_loc) and this_delta_peak_loc > 0.:
                this_peak_shift_count += 1
        if lap in snapshot_laps:
            print('lap: %i; peak_shift_count: %i cells' % (lap, this_peak_shift_count))
            this_peak_shift_count = 0
    print('total active cells: %i / %i' % (len(active_cell_index_set), num_cells))

    num_bins = 50
    edges = np.linspace(0., track_length, num_bins + 1)
    bin_width = edges[1] - edges[0]

    peak_locs_histogram_history = []
    for lap in range(num_laps):
        this_peak_locs_snapshot = peak_locs_snapshots[lap]
        hist, _edges = np.histogram(this_peak_locs_snapshot, bins=edges)
        hist = hist.astype(float) / num_cells
        peak_locs_histogram_history.append(hist)
    peak_locs_histogram_history = np.array(peak_locs_histogram_history)

    fig1, axes1 = plt.subplots()
    fig2, axes2 = plt.subplots()
    for lap, lap_label in zip(snapshot_laps, lap_labels):
        this_population_representation_density = population_representation_density_history[lap]
        this_peak_locs_hist = peak_locs_histogram_history[lap]
        axes1.plot(default_interp_x, this_population_representation_density, label=lap_label)
        axes2.plot(edges[:-1] + bin_width / 2., this_peak_locs_hist, label=lap_label)
    axes1.set_xlabel('Position (cm)')
    axes1.set_xticks(np.arange(0., track_length, 45.))
    axes1.set_title('Summed population activity', fontsize=mpl.rcParams['font.size'])
    axes1.set_ylabel('Normalized population activity')
    axes1.set_ylim([0., axes1.get_ylim()[1]])
    axes1.legend(loc='best', frameon=False, framealpha=0.5)
    axes2.set_title('Place field peak locations', fontsize=mpl.rcParams['font.size'])
    axes2.set_xlabel('Position (cm)')
    axes2.set_xticks(np.arange(0., track_length, 45.))
    axes2.set_ylabel('Fraction of cells')
    axes2.legend(loc='best', frameon=False, framealpha=0.5)
    clean_axes([axes1, axes2])
    fig1.tight_layout()
    fig2.tight_layout()

    num_snapshots = 3
    num_cols = num_snapshots + 2
    start_col = num_snapshots - min(num_snapshots, len(summary_laps))
    col_width = 4
    col_height = 4
    fig3, axes = plt.subplots(1, num_cols, figsize=[num_cols * col_width, col_height])
    hmaps = []
    cbars = []

    sorted_normalized_ramp_snapshots = []
    max_ramp = np.max(ramp_snapshots)
    for i, lap in enumerate(summary_laps[:num_snapshots]):
        this_peak_locs = np.array([peak_loc_history[cell][lap] for cell in range(num_cells)])
        this_indexes = np.arange(num_cells)
        valid_indexes = np.where(~np.isnan(this_peak_locs))[0]
        sorted_subindexes = np.argsort(this_peak_locs[valid_indexes])
        this_sorted_ramps = []
        for cell in this_indexes[valid_indexes][sorted_subindexes]:
            ramp = np.interp(default_interp_x, binned_x, ramp_snapshots[lap][cell])
            this_sorted_ramps.append(ramp)
        sorted_normalized_ramp_snapshots.append(this_sorted_ramps)
        hm = axes[i + start_col].imshow(this_sorted_ramps, extent=(0., track_length, len(this_sorted_ramps), 0),
                                        aspect='auto',
                                        vmin=0., vmax=max_ramp)
        hmaps.append(hm)
        cbar = plt.colorbar(hm, ax=axes[i + start_col])
        cbar.ax.set_ylabel('Ramp amplitude (mV)', rotation=270)
        cbar.ax.get_yaxis().labelpad = 15
        cbars.append(cbar)
        axes[i + start_col].set_xticks(np.arange(0., track_length, 45.))
        axes[i + start_col].set_xlabel('Position (cm)')
        axes[i + start_col].set_ylabel('Cell index', labelpad=-15)
        axes[i + start_col].set_yticks([0, len(this_sorted_ramps) - 1])
        axes[i + start_col].set_yticklabels([1, len(this_sorted_ramps)])
        axes[i + start_col].set_title(summary_lap_labels[i], fontsize=mpl.rcParams['font.size'], y=1.025)

    hm3 = axes[3].imshow(population_representation_density_history, extent=(0., track_length, len(ramp_snapshots), 0),
                         aspect='auto')
    cbar3 = plt.colorbar(hm3, ax=axes[3])
    cbar3.ax.set_ylabel('Normalized population activity', rotation=270)
    cbar3.ax.get_yaxis().labelpad = 15
    axes[3].set_xticks(np.arange(0., track_length, 45.))
    axes[3].set_xlabel('Position (cm)')
    axes[3].set_ylabel('Lap')
    axes[3].set_yticks(range(num_laps), minor=True)
    axes[3].set_yticks(range(0, num_laps + 1, 5))
    axes[3].set_yticklabels(range(0, num_laps + 1, 5))
    axes[3].set_title('Summed\npopulation activity', fontsize=mpl.rcParams['font.size'], y=1.025)

    hm4 = axes[4].imshow(peak_locs_histogram_history, extent=(0., track_length, num_laps, 0),
                         aspect='auto', vmin=0.)  # , vmax=max(np.max(peak_locs_histogram_history), 0.12))
    hmaps.append(hm4)
    cbar4 = plt.colorbar(hm4, ax=axes[4])
    cbar4.ax.set_ylabel('Fraction of cells', rotation=270)
    cbar4.ax.get_yaxis().labelpad = 15
    cbars.append(cbar4)
    axes[4].set_xticks(np.arange(0., track_length, 45.))
    axes[4].set_xlabel('Position (cm)')
    axes[4].set_ylabel('Lap')
    axes[4].set_yticks(range(num_laps), minor=True)
    axes[4].set_yticks(range(0, num_laps + 1, 5))
    axes[4].set_yticklabels(range(0, num_laps + 1, 5))
    axes[4].set_title('Place field\npeak locations', fontsize=mpl.rcParams['font.size'], y=1.025)

    fig3.tight_layout(w_pad=0.8)

    context.update(locals())

    plt.show()


def plot_model_summary_figure(file_path):
    """

    :param file_path: str (path)
    """
    if not os.path.isfile(file_path):
        raise IOError('plot_model_summary_figure: invalid file path: %s' % file_path)
    exported_data_key = 'BTSP_population_history'
    with h5py.File(file_path, 'r') as f:
        if 'shared_context' not in f or exported_data_key not in f or 'enumerated' not in f[exported_data_key].attrs \
                or not f[exported_data_key].attrs['enumerated']:
            raise KeyError('plot_model_summary_figure: invalid file contents at path: %s' % file_path)
        binned_x = f['shared_context']['binned_x'][:]
        default_interp_x = f['shared_context']['default_interp_x'][:]
        track_length = f['shared_context'].attrs['track_length']
        num_groups = len(f[exported_data_key])
        for i in range(num_groups):
            group_id = str(i)
            group = f[exported_data_key][group_id]
            reward_locs_array = group.attrs['reward_locs_array']
            ramp_snapshots = group['ramp_snapshots'][:]
            population_representation_density_history = group['population_representation_density_history'][:]
            num_baseline_laps = group.attrs['num_baseline_laps']
            num_assay_laps = group.attrs['num_assay_laps']
            num_reward_laps = group.attrs['num_reward_laps']
            trial = group.attrs['trial']
            plot_population_history_snapshots(ramp_snapshots, population_representation_density_history,
                                              reward_locs_array, binned_x, default_interp_x, track_length, num_baseline_laps,
                                              num_assay_laps, num_reward_laps, trial)


def plot_plateau_modulation():
    """

    """
    import matplotlib as mpl
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['font.size'] = 11.
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['axes.titlepad'] = 2.
    mpl.rcParams['mathtext.default'] = 'regular'

    max_cols = 5
    col_width = 3.15
    col_height = 4
    fig6, axes6 = plt.subplots(1, max_cols, figsize=[max_cols * col_width, col_height])
    norm_basal_plateau_prob = context.basal_plateau_prob_f(context.basal_representation_xscale)
    norm_basal_plateau_prob /= np.max(norm_basal_plateau_prob)
    norm_basal_plateau_prob *= context.peak_basal_plateau_prob_per_lap
    norm_reward_plateau_prob = context.reward_plateau_prob_f(context.reward_representation_xscale)
    norm_reward_plateau_prob /= np.max(norm_reward_plateau_prob)
    norm_reward_plateau_prob *= context.peak_reward_plateau_prob_per_lap
    axes6[0].plot(context.basal_representation_xscale, norm_basal_plateau_prob, label='No reward', c='k')
    axes6[0].plot(context.reward_representation_xscale, norm_reward_plateau_prob, label='Reward', c='r')
    axes6[0].set_xticks([i * 0.25 for i in range(6)])
    axes6[0].set_xlim(0., 1.25)
    axes6[0].set_ylim(0., axes6[0].get_ylim()[1])
    axes6[0].set_xlabel('Normalized population activity')
    axes6[0].set_ylabel('Plateau probability per lap')
    axes6[0].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
    axes6[0].set_title('Modulation of plateau probability\nby feedback inhibition and reward',
                       fontsize=mpl.rcParams['font.size'], y=1.025)

    plateau_prob_ramp_modulation = context.plateau_prob_ramp_sensitivity_f(context.ramp_xscale)
    axes6[1].plot(context.ramp_xscale, plateau_prob_ramp_modulation * context.basal_plateau_prob_ramp_sensitivity + 1.,
                  label='No reward', c='k')
    axes6[1].plot(context.ramp_xscale, plateau_prob_ramp_modulation * context.reward_plateau_prob_ramp_sensitivity + 1.,
                  label='Reward', c='r')
    axes6[1].set_xlim(0., 10.)
    # axes6[1].set_ylim(0., 1.)
    axes6[1].set_xlabel('Ramp amplitude (mV)')
    axes6[1].set_ylabel('Normalized plateau probability')
    axes6[1].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
    axes6[1].set_title('Modulation of plateau probability\nby ramp amplitude',
                       fontsize=mpl.rcParams['font.size'], y=1.025)

    clean_axes(axes6)
    fig6.tight_layout(w_pad=0.8)
    plt.show()


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/simulate_biBTSP_synthetic_network_config.yaml')
@click.option("--trial", type=int, default=0)
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default='biBTSP_synthetic_network')
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
    python -i simulate_biBTSP_synthetic_network.py --plot --framework=serial --interactive

    To execute using MPI parallelism with 1 controller process and N - 1 worker processes:
    mpirun -n N python -i -m mpi4py.futures simulate_biBTSP_synthetic_network.py --plot --framework=mpi --interactive

    or interactively:

    To plot results previously exported to a file on a single process:
    python -i simulate_biBTSP_synthetic_network.py --plot-summary-figure --model-file-path=$PATH_TO_MODEL_FILE \
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
        print('getting here')
        config_worker()
    if plot_summary_figure:
        plot_model_summary_figure(model_file_path)
    elif not debug:
        simulate_network(export, plot)
    if plot:
        context.interface.apply(plt.show)
        plot_plateau_modulation()
        plt.show()

    if context.interactive:
        context.update(locals())
    else:
        context.interface.stop()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
