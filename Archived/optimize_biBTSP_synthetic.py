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

biBTSP_synthetic_alt: Single eligibility signal filter.
"""
__author__ = 'milsteina'
from biBTSP_utils import *
from nested.parallel import *
from nested.optimize_utils import *
import click

context = Context()


BTSP_model_name = 'synthetic'


def config_worker():
    """

    """
    context.data_file_path = context.output_dir + '/' + context.data_file_name
    init_context()


def init_context():
    """

    """
    if context.data_file_path is None or not os.path.isfile(context.data_file_path):
        raise IOError('optimize_biBTSP_%s: init_context: invalid data_file_path: %s' %
                      (BTSP_model_name, context.data_file_path))

    if 'weights_path_distance_threshold' not in context():
        context.weights_path_distance_threshold = 2.
    else:
        context.weights_path_distance_threshold = float(context.weights_path_distance_threshold)

    context.verbose = int(context.verbose)
    if 'plot' not in context():
        context.plot = False

    if 'truncate' not in context():
        context.truncate = 2.5

    with h5py.File(context.data_file_path, 'r') as f:
        dt = f['defaults'].attrs['dt']  # ms
        input_field_peak_rate = f['defaults'].attrs['input_field_peak_rate']  # Hz
        num_inputs = f['defaults'].attrs['num_inputs']
        track_length = f['defaults'].attrs['track_length']  # cm
        binned_dx = f['defaults'].attrs['binned_dx']  # cm
        generic_dx = f['defaults'].attrs['generic_dx']  # cm
        binned_x = f['defaults']['binned_x'][:]
        generic_x = f['defaults']['generic_x'][:]
        extended_x = f['defaults']['extended_x'][:]
        if 'default_run_vel' not in context() or context.default_run_vel is None:
            default_run_vel = f['defaults'].attrs['default_run_vel']  # cm/s
            generic_position_dt = f['defaults'].attrs['generic_position_dt']  # ms
            default_interp_dx = f['defaults'].attrs['default_interp_dx']  # cm
            generic_t = f['defaults']['generic_t'][:]
            default_interp_t = f['defaults']['default_interp_t'][:]
            default_interp_x = f['defaults']['default_interp_x'][:]
        else:
            default_run_vel = float(context.default_run_vel)
            generic_position_dt = generic_dx / default_run_vel * 1000.  # ms
            generic_t = np.arange(0., len(generic_x) * generic_position_dt, generic_position_dt)[:len(generic_x)]
            default_interp_t = np.arange(0., generic_t[-1], dt)
            default_interp_x = np.interp(default_interp_t, generic_t, generic_x)
            default_interp_dx = dt * default_run_vel / 1000.  # cm

        if 'input_field_width' not in context() or context.input_field_width is None:
            raise RuntimeError('optimize_biBTSP_%s: init context: missing required parameter: input_field_width' %
                               BTSP_model_name)
        context.input_field_width = float(context.input_field_width)
        input_field_width_key = str(int(context.input_field_width))
        if 'calibrated_input' not in f or input_field_width_key not in f['calibrated_input']:
            raise RuntimeError('optimize_biBTSP_%s: init context: data for input_field_width: %.1f not found in the '
                               'provided data_file_path: %s' %
                               (BTSP_model_name, float(context.input_field_width), context.data_file_path))
        input_field_width = f['calibrated_input'][input_field_width_key].attrs['input_field_width']  # cm
        input_rate_maps, peak_locs = \
            generate_spatial_rate_maps(binned_x, num_inputs, input_field_peak_rate, input_field_width, track_length)
        ramp_scaling_factor = f['calibrated_input'][input_field_width_key].attrs['ramp_scaling_factor']

    down_dt = 10.  # ms, to speed up optimization
    if 'num_induction_laps' not in context():
        num_induction_laps = 1
    else:
        num_induction_laps = context.num_induction_laps
    induction_dur = 300.  # ms
    context.update(locals())

    context.induction = None
    context.data_cache = defaultdict(dict)


def load_data(induction):
    """

    :param induction: int
    """
    induction = int(induction)
    if induction == context.induction:
        return

    if induction in context.data_cache:
        induction_context = context.data_cache[induction]
    else:
        induction_key = str(induction)
        induction_context = Context()

        this_induction_loc = context.induction_loc[induction_key]
        induction_context.mean_induction_start_loc = this_induction_loc
        induction_context.complete_induction_locs = [this_induction_loc for i in range(context.num_induction_laps)]
        induction_context.complete_induction_durs = [context.induction_dur for i in range(context.num_induction_laps)]
        this_induction_stop_index = np.where(context.default_interp_x >= this_induction_loc)[0][0] + \
                                    int(context.induction_dur / context.dt)

        induction_context.mean_induction_stop_loc = context.default_interp_x[this_induction_stop_index]

        induction_context.position = {key: [] for key in ['pre', 'induction', 'post']}
        induction_context.t = {key: [] for key in ['pre', 'induction', 'post']}
        induction_context.current = []

        induction_context.position['pre'].append(context.default_interp_x)
        induction_context.t['pre'].append(context.default_interp_t)
        for i in range(context.num_induction_laps):
            induction_context.position['induction'].append(context.default_interp_x)
            induction_context.t['induction'].append(context.default_interp_t)
        induction_context.position['post'].append(context.default_interp_x)
        induction_context.t['post'].append(context.default_interp_t)

        for i in range(context.num_induction_laps):
            start_index = np.where(context.default_interp_x >= induction_context.complete_induction_locs[i])[0][0]
            stop_index = np.where(context.default_interp_t >= context.default_interp_t[start_index] +
                                  induction_context.complete_induction_durs[i])[0][0]
            this_current = np.zeros_like(context.default_interp_t)
            this_current[start_index:stop_index] = 1.
            induction_context.current.append(this_current)

        induction_context.complete_position = np.array([])
        induction_context.complete_t = np.array([])
        running_dur = 0.
        running_length = 0.

        for group in (group for group in ['pre', 'induction', 'post'] if group in induction_context.position):
            for this_position, this_t in zip(induction_context.position[group], induction_context.t[group]):
                induction_context.complete_position = \
                    np.append(induction_context.complete_position, np.add(this_position, running_length))
                induction_context.complete_t = \
                    np.append(induction_context.complete_t, np.add(this_t, running_dur))
                running_length += context.track_length
                running_dur += len(this_t) * context.dt
        for i in range(len(induction_context.t['pre'])):
            induction_context.complete_t -= len(induction_context.t['pre'][i]) * context.dt
            induction_context.complete_position -= context.track_length

        induction_context.min_induction_t = \
            get_min_induction_t(induction_context.complete_t, induction_context.complete_position,
                                context.binned_x, context.track_length, induction_context.mean_induction_start_loc,
                                context.num_induction_laps)
        induction_context.clean_induction_t_indexes = \
            get_clean_induction_t_indexes(induction_context.min_induction_t, context.truncate * 1000.)

        induction_context.complete_run_vel = np.full_like(induction_context.complete_t, context.default_run_vel)
        induction_context.complete_run_vel_gate = np.ones_like(induction_context.complete_run_vel)
        induction_context.complete_run_vel_gate[np.where(induction_context.complete_run_vel <= 5.)[0]] = 0.

        # pre lap
        induction_context.induction_gate = np.zeros_like(context.default_interp_t)
        for this_current in induction_context.current:
            induction_context.induction_gate = np.append(induction_context.induction_gate, this_current)
        # post lap
        induction_context.induction_gate = \
            np.append(induction_context.induction_gate, np.zeros_like(context.default_interp_t))

        induction_context.induction_start_times = []
        induction_context.induction_stop_times = []
        running_position = 0.
        running_t = 0.
        for i in range(context.num_induction_laps):
            this_induction_start_index = \
                np.where(induction_context.complete_position >= this_induction_loc + running_position)[0][0]
            this_induction_start_time = induction_context.complete_t[this_induction_start_index]
            this_induction_stop_time = this_induction_start_time + context.induction_dur
            running_t += len(induction_context.t['induction'][i]) * context.dt
            induction_context.induction_start_times.append(this_induction_start_time)
            induction_context.induction_stop_times.append(this_induction_stop_time)
            running_position += context.track_length
        induction_context.induction_start_times = np.array(induction_context.induction_start_times)
        induction_context.induction_stop_times = np.array(induction_context.induction_stop_times)

        induction_context.target_ramp = {}
        induction_context.LSA_weights = {}
        if induction == 2:
            induction_loc_1 = context.induction_loc['1']
            if 1 in context.data_cache:
                induction_context.target_ramp['before'] = context.data_cache[1].target_ramp['after']
                induction_context.LSA_weights['before'] = context.data_cache[1].LSA_weights['after']
            else:
                induction_stop_index = np.where(context.default_interp_x >= induction_loc_1)[0][0] + \
                                         int(context.induction_dur / context.dt)
                induction_stop_loc = context.default_interp_x[induction_stop_index]

                target_ramp = get_target_synthetic_ramp(induction_loc_1, ramp_x=context.binned_x,
                                                        track_length=context.track_length,
                                                        target_peak_val=context.target_peak_val_1,
                                                        target_min_val=0., target_asymmetry=context.target_asymmetry_1,
                                                        target_peak_shift=context.target_peak_shift_1,
                                                        target_ramp_width=context.target_ramp_width_1)
                induction_context.target_ramp['before'], \
                induction_context.LSA_weights['before'], _, _ = \
                    get_delta_weights_LSA(target_ramp, ramp_x=context.binned_x, input_x=context.binned_x,
                                          interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                          peak_locs=context.peak_locs, ramp_scaling_factor=context.ramp_scaling_factor,
                                          induction_start_loc=induction_loc_1, induction_stop_loc=induction_stop_loc,
                                          track_length=context.track_length, target_range=context.target_range,
                                          bounds=(context.min_delta_weight, context.target_peak_delta_weight),
                                          verbose=context.verbose)

            # the shape of this target_ramp was estimated from gaussian regression from experimental data
            target_delta_ramp_2_dep = get_target_synthetic_ramp(induction_loc_1, ramp_x=context.binned_x,
                                                                track_length=context.track_length,
                                                                target_peak_val=-context.target_delta_peak_val_2,
                                                                target_min_val=0.,
                                                                target_asymmetry=context.target_asymmetry_2,
                                                                target_peak_shift=context.target_peak_shift_2,
                                                                target_ramp_width=context.target_ramp_width_2)
            target_delta_ramp_2_pot = get_target_synthetic_ramp(induction_context.mean_induction_start_loc,
                                                                ramp_x=context.binned_x,
                                                                track_length=context.track_length,
                                                                target_peak_val=context.target_peak_val_1,
                                                                target_min_val=0.,
                                                                target_asymmetry=context.target_asymmetry_1,
                                                                target_peak_shift=context.target_peak_shift_1,
                                                                target_ramp_width=context.target_ramp_width_1)
            target_delta_ramp_2 = np.subtract(target_delta_ramp_2_pot, target_delta_ramp_2_dep)
            target_ramp_2 = np.add(induction_context.target_ramp['before'], target_delta_ramp_2)
            induction_context.target_ramp['after'], \
            induction_context.LSA_weights['after'], _, _ = \
                get_delta_weights_LSA(target_ramp_2, ramp_x=context.binned_x, input_x=context.binned_x,
                                      interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                      peak_locs=context.peak_locs, ramp_scaling_factor=context.ramp_scaling_factor,
                                      induction_start_loc=induction_context.mean_induction_start_loc,
                                      induction_stop_loc=induction_context.mean_induction_stop_loc,
                                      track_length=context.track_length, target_range=context.target_range,
                                      bounds=(context.min_delta_weight, context.target_peak_delta_weight),
                                      verbose=context.verbose)
        else:
            induction_context.target_ramp['before'] = np.zeros_like(context.binned_x)
            induction_context.LSA_weights['before'] = np.zeros_like(context.peak_locs)
            induction_loc = induction_context.mean_induction_start_loc
            induction_stop_loc = induction_context.mean_induction_stop_loc

            target_ramp = get_target_synthetic_ramp(induction_loc, ramp_x=context.binned_x,
                                                    track_length=context.track_length,
                                                    target_peak_val=context.target_peak_val_1, target_min_val=0.,
                                                    target_asymmetry=context.target_asymmetry_1,
                                                    target_peak_shift=context.target_peak_shift_1,
                                                    target_ramp_width=context.target_ramp_width_1)
            induction_context.target_ramp['after'], \
            induction_context.LSA_weights['after'], _, _ = \
                get_delta_weights_LSA(target_ramp, ramp_x=context.binned_x, input_x=context.binned_x,
                                      interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                      peak_locs=context.peak_locs, ramp_scaling_factor=context.ramp_scaling_factor,
                                      induction_start_loc=induction_loc, induction_stop_loc=induction_stop_loc,
                                      track_length=context.track_length, target_range=context.target_range,
                                      bounds=(context.min_delta_weight, context.target_peak_delta_weight),
                                      verbose=context.verbose)
        context.data_cache[induction] = induction_context
    context.update(induction_context())

    context.complete_rate_maps = \
        get_complete_rate_maps(context.input_rate_maps, context.binned_x, context.position,
                               context.complete_run_vel_gate)
    context.down_t = np.arange(context.complete_t[0],
                               context.complete_t[-1] + context.down_dt / 2., context.down_dt)
    context.down_rate_maps = []
    for rate_map in context.complete_rate_maps:
        this_down_rate_map = np.interp(context.down_t, context.complete_t, rate_map)
        context.down_rate_maps.append(this_down_rate_map)
    context.down_induction_gate = np.interp(context.down_t, context.complete_t, context.induction_gate)
    context.induction = induction

    if context.verbose > 1:
        print('optimize_biBTSP_%s: process: %i loaded data for induction: %i' %
              (BTSP_model_name, os.getpid(), induction))


def update_model_params(x, local_context):
    """

    :param x: array
    :param local_context: :class:'Context'
    """
    if local_context is None:
        local_context = context
    local_context.update(param_array_to_dict(x, local_context.param_names))


def calculate_model_ramp(model_id=None, export=False, plot=False):
    """

    :param model_id: bool
    :param export: bool
    :param plot: bool
    :return: dict
    """
    local_signal_filter_t, local_signal_filter, global_filter_t, global_filter = \
        get_dual_exp_decay_signal_filters(context.local_signal_decay, context.global_signal_decay, context.down_dt)
    global_signal = get_global_signal(context.down_induction_gate, global_filter)
    global_signal_peak = np.max(global_signal)
    global_signal /= global_signal_peak
    local_signals = get_local_signal_population(local_signal_filter, 
                                                context.down_rate_maps / context.input_field_peak_rate)
    local_signal_peak = np.max(local_signals)
    local_signals /= local_signal_peak

    signal_xrange = np.linspace(0., 1., 10000)
    pot_rate = np.vectorize(scaled_single_sigmoid(
        context.f_pot_th, context.f_pot_th + context.f_pot_half_width, signal_xrange))
    dep_rate = np.vectorize(scaled_single_sigmoid(
        context.f_dep_th, context.f_dep_th + context.f_dep_half_width, signal_xrange))

    if plot and context.induction == 1:
        fig, axes = plt.subplots(1, 2)
        axes[0].plot(local_signal_filter_t / 1000., local_signal_filter / np.max(local_signal_filter), color='r',
                     label='Eligibility signal filter')
        axes[0].plot(global_filter_t / 1000., global_filter / np.max(global_filter), color='k',
                     label='Instructive signal filter')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Normalized amplitude')
        axes[0].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
        axes[0].set_xlim(-0.5, max(5000., local_signal_filter_t[-1], global_filter_t[-1]) / 1000.)

        dep_scale = context.k_dep / context.k_pot
        axes[1].plot(signal_xrange, pot_rate(signal_xrange), c='r', label='Potentiation rate')
        axes[1].plot(signal_xrange, dep_rate(signal_xrange) * dep_scale, c='c', label='Depression rate')
        axes[1].set_xlabel('Plasticity signal overlap (a.u.)')
        axes[1].set_ylabel('Normalized rate')
        axes[1].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
        clean_axes(axes)
        fig.tight_layout()
        fig.show()

    peak_weight = context.peak_delta_weight + 1.

    initial_delta_weights = context.LSA_weights['before']
    initial_ramp = context.target_ramp['before']

    # re-compute initial weights if they are out of the current weight bounds
    if context.induction == 2:
        if not np.all((context.min_delta_weight <= initial_delta_weights) &
                      (initial_delta_weights <= context.peak_delta_weight)):
            initial_ramp, initial_delta_weights, _, _ = \
                get_delta_weights_LSA(initial_ramp, ramp_x=context.binned_x, input_x=context.binned_x,
                                      interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                      peak_locs=context.peak_locs, ramp_scaling_factor=context.ramp_scaling_factor,
                                      induction_start_loc=context.mean_induction_start_loc,
                                      induction_stop_loc=context.mean_induction_stop_loc,
                                      track_length=context.track_length, target_range=context.target_range,
                                      bounds=(context.min_delta_weight, context.peak_delta_weight),
                                      initial_delta_weights=initial_delta_weights, verbose=context.verbose)
            if context.verbose > 1:
                print('Process: %i; model_id: %i; re-computed initial weights before induction: %i' %
                      (os.getpid(), model_id, context.induction))

            # discard model if current peak_delta_weight constraint reduces accuracy of initial_ramp
            if not 0.9 * np.max(context.target_ramp['before']) < np.max(initial_ramp) < \
                   1.1 * np.max(context.target_ramp['before']):
                if context.verbose > 0:
                    print(
                        'optimize_biBTSP_%s: calculate_model_ramp: pid: %i; model_id: %s: aborting - initial ramp is '
                        'inconsistent with value of peak_delta_weight: %.1f' %
                        (BTSP_model_name, os.getpid(), model_id, context.peak_delta_weight))
                    sys.stdout.flush()
                return dict()

    delta_weights_snapshots = [initial_delta_weights]
    current_ramp = initial_ramp
    ramp_snapshots = [current_ramp]
    initial_normalized_weights = np.divide(np.add(initial_delta_weights, 1.), peak_weight)
    current_normalized_weights = np.array(initial_normalized_weights)

    target_ramp = context.target_ramp['after']

    if plot:
        fig, axes = plt.subplots()
        fig.suptitle('Induction: %i' % (context.induction), y=1.)
        axes.set_xlabel('Time (s)')
        axes.set_ylabel('Integrated plasticity\nsignal overlap (a.u.)')

        fig2, axes2 = plt.subplots(1, 3, figsize=(10., 4.))
        fig2.suptitle('Induction: %i' % context.induction)
        axes2[0].plot(context.binned_x, initial_ramp, c='k', label='Before')
        axes2[0].plot(context.binned_x, target_ramp, c='r', label='Target')
        axes2[0].set_ylabel('Ramp amplitude (mV)')
        axes2[0].set_xlabel('Location (cm)')
        axes2[1].set_ylabel('Change in synaptic\nweight (normalized)')
        axes2[1].set_xlabel('Location (cm)')
        axes2[2].set_xlabel('Time relative to plateau onset (s)')
        axes2[2].set_ylabel('Change in ramp amplitude (mV)')
        axes2[2].plot(context.min_induction_t[context.clean_induction_t_indexes] / 1000.,
                     np.zeros_like(context.clean_induction_t_indexes), c='darkgrey', alpha=0.75,
                     zorder=1, linestyle='--')

    result = {}

    for induction_lap in range(len(context.induction_start_times)):
        start_time = context.induction_start_times[induction_lap]
        if induction_lap == len(context.induction_start_times) - 1:
            stop_time = context.down_t[-1]
        else:
            stop_time = context.induction_start_times[induction_lap + 1]
        indexes = np.where((context.down_t >= start_time) & (context.down_t < stop_time))[0]

        next_normalized_weights = []
        overlap = []
        for i, this_local_signal in enumerate(local_signals):
            this_signal_overlap = np.multiply(this_local_signal[indexes], global_signal[indexes])
            this_pot_rate = np.trapz(pot_rate(this_signal_overlap), dx=context.down_dt / 1000.)
            this_dep_rate = np.trapz(dep_rate(this_signal_overlap), dx=context.down_dt / 1000.)
            this_normalized_delta_weight = context.k_pot * this_pot_rate * (1. - current_normalized_weights[i]) - \
                                           context.k_dep * this_dep_rate * current_normalized_weights[i]
            this_next_normalized_weight = max(0., min(1., current_normalized_weights[i] + this_normalized_delta_weight))
            next_normalized_weights.append(this_next_normalized_weight)
            overlap.append(np.trapz(this_signal_overlap, dx=context.down_dt / 1000.))
        if plot:
            interp_overlap = np.interp(context.binned_x, context.peak_locs, overlap)
            axes.plot(context.min_induction_t[context.clean_induction_t_indexes] / 1000.,
                      interp_overlap[context.clean_induction_t_indexes],
                      label='Induction lap: %i' % (induction_lap + 1))
            # axes.plot(context.down_t[indexes], global_signal[indexes], label='Induction lap: %i' % (induction_lap + 1))
            axes2[1].plot(context.peak_locs, np.subtract(next_normalized_weights, current_normalized_weights),
                          label='Induction lap: %i' % (induction_lap + 1))
        current_normalized_weights = np.array(next_normalized_weights)
        current_delta_weights = np.subtract(np.multiply(current_normalized_weights, peak_weight), 1.)
        delta_weights_snapshots.append(current_delta_weights)
        current_ramp, discard_ramp_offset = \
            get_model_ramp(current_delta_weights, ramp_x=context.binned_x, input_x=context.binned_x,
                           input_rate_maps=context.input_rate_maps, ramp_scaling_factor=context.ramp_scaling_factor)

        if plot:
            axes2[0].plot(context.binned_x, current_ramp)

        if context.induction == 1 and induction_lap == 0:
            result['ramp_amp_after_first_plateau'] = np.max(current_ramp)
        ramp_snapshots.append(current_ramp)

    if plot:
        axes.legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
        clean_axes(axes)
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        fig.show()

        axes2[2].plot(context.min_induction_t[context.clean_induction_t_indexes] / 1000.,
                      np.subtract(current_ramp, initial_ramp)[context.clean_induction_t_indexes], c='k')
        axes2[0].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
        axes2[1].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
        clean_axes(axes2)
        fig2.tight_layout()
        fig2.subplots_adjust(top=0.9)
        fig2.show()

    delta_weights = np.subtract(current_delta_weights, initial_delta_weights)
    initial_weights = np.multiply(initial_normalized_weights, peak_weight)
    final_weights = np.add(current_delta_weights, 1.)

    if context.induction == 2:
        model_ramp, discard_delta_weights, model_ramp_offset, model_residual_score = \
            get_synthetic_residual_score(current_delta_weights, target_ramp, initial_ramp, ramp_x=context.binned_x,
                                         input_x=context.binned_x, interp_x=context.default_interp_x,
                                         input_rate_maps=context.input_rate_maps,
                                         ramp_scaling_factor=context.ramp_scaling_factor,
                                         induction_loc=context.mean_induction_start_loc,
                                         track_length=context.track_length, target_range=context.target_range,
                                         full_output=True)
    else:
        model_ramp, discard_delta_weights, model_ramp_offset, model_residual_score = \
            get_residual_score(current_delta_weights, target_ramp, ramp_x=context.binned_x,
                                         input_x=context.binned_x, interp_x=context.default_interp_x,
                                         input_rate_maps=context.input_rate_maps,
                                         ramp_scaling_factor=context.ramp_scaling_factor,
                                         induction_loc=context.mean_induction_start_loc,
                                         track_length=context.track_length, target_range=context.target_range,
                                         full_output=True)

    result['residual_score'] = model_residual_score

    ramp_amp, ramp_width, peak_shift, ratio, start_loc, peak_loc, end_loc, min_val, min_loc = {}, {}, {}, {}, {}, {}, \
                                                                                              {}, {}, {}

    ramp_amp['target'], ramp_width['target'], peak_shift['target'], ratio['target'], start_loc['target'], \
    peak_loc['target'], end_loc['target'], min_val['target'], min_loc['target'] = \
        calculate_ramp_features(ramp=target_ramp, induction_loc=context.mean_induction_start_loc,
                                binned_x=context.binned_x, interp_x=context.default_interp_x,
                                track_length=context.track_length)

    ramp_amp['model'], ramp_width['model'], peak_shift['model'], ratio['model'], start_loc['model'], \
    peak_loc['model'], end_loc['model'], min_val['model'], min_loc['model'] = \
        calculate_ramp_features(ramp=model_ramp, induction_loc=context.mean_induction_start_loc,
                                binned_x=context.binned_x, interp_x=context.default_interp_x,
                                track_length=context.track_length)

    if context.verbose > 0:
        print('Process: %i; induction: %i; model_id: %s:' % (os.getpid(), context.induction, str(model_id)))
        print('exp: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, peak_loc: %.1f, '
              'end_loc: %.1f, min_val: %.1f, min_loc: %.1f' %
              (ramp_amp['target'], ramp_width['target'], peak_shift['target'], ratio['target'], start_loc['target'],
               peak_loc['target'], end_loc['target'], min_val['target'], min_loc['target']))
        print('model: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, peak_loc: %.1f' 
              ', end_loc: %.1f, min_val: %.1f, min_loc: %.1f' %
              (ramp_amp['model'], ramp_width['model'], peak_shift['model'], ratio['model'], start_loc['model'],
               peak_loc['model'], end_loc['model'], min_val['model'], min_loc['model']))
        sys.stdout.flush()

    peak_index, min_index = {}, {}
    _, peak_index['target'], _, min_index['target'] = \
        get_indexes_from_ramp_bounds_with_wrap(context.binned_x, start_loc['target'], peak_loc['target'],
                                               end_loc['target'], min_loc['target'])
    _, peak_index['model'], _, min_index['model'] = \
        get_indexes_from_ramp_bounds_with_wrap(context.binned_x, start_loc['model'], peak_loc['model'],
                                               end_loc['model'], min_loc['model'])

    model_val_at_target_min_loc = model_ramp[min_index['target']]
    target_val_at_model_min_loc = target_ramp[min_index['model']]
    model_val_at_target_peak_loc = model_ramp[peak_index['target']]
    target_val_at_model_peak_loc = target_ramp[peak_index['model']]

    result['delta_val_at_target_peak'] = model_val_at_target_peak_loc - ramp_amp['target']
    result['delta_val_at_model_peak'] = ramp_amp['model'] - target_val_at_model_peak_loc
    result['delta_width'] = ramp_width['model'] - ramp_width['target']
    result['delta_peak_shift'] = peak_shift['model'] - peak_shift['target']
    result['delta_asymmetry'] = ratio['model'] - ratio['target']
    result['delta_val_at_target_min'] = model_val_at_target_min_loc - min_val['target']
    result['delta_val_at_model_min'] = min_val['model'] - target_val_at_model_min_loc

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

    local_peak_loc, local_peak_shift = {}, {}
    local_peak_loc['target'], local_peak_shift['target'] = \
        get_local_peak_shift(ramp=target_ramp, induction_loc=context.mean_induction_start_loc,
                             binned_x=context.binned_x, interp_x=context.default_interp_x,
                             track_length=context.track_length)
    local_peak_loc['model'], local_peak_shift['model'] = \
        get_local_peak_shift(ramp=model_ramp, induction_loc=context.mean_induction_start_loc,
                             binned_x=context.binned_x, interp_x=context.default_interp_x,
                             track_length=context.track_length)

    if export:
        with h5py.File(context.temp_output_path, 'a') as f:
            shared_context_key = 'shared_context'
            if shared_context_key not in f:
                f.create_group(shared_context_key)
                group = f[shared_context_key]
                group.create_dataset('peak_locs', compression='gzip', data=context.peak_locs)
                group.create_dataset('binned_x', compression='gzip', data=context.binned_x)
                group.create_dataset('signal_xrange', compression='gzip', data=signal_xrange)
                group.create_dataset('param_names', compression='gzip', data=np.asarray(context.param_names, dtype='S'))
                group.attrs['input_field_width'] = context.input_field_width
                group.attrs['ramp_scaling_factor'] = context.ramp_scaling_factor
            exported_data_key = 'exported_data'
            description = 'model_ramp_features'
            cell_key = 'synthetic'
            induction_key = str(context.induction)
            group = get_h5py_group(f, [model_id, exported_data_key, cell_key, induction_key, description], create=True)
            group.create_dataset('target_ramp', compression='gzip', data=target_ramp)
            group.create_dataset('initial_model_ramp', compression='gzip', data=initial_ramp)
            group.create_dataset('model_ramp', compression='gzip', data=model_ramp)
            group.create_dataset('model_weights', compression='gzip', data=final_weights)
            group.create_dataset('initial_weights', compression='gzip', data=initial_weights)
            group.create_dataset('global_signal', compression='gzip', data=global_signal)
            group.create_dataset('down_t', compression='gzip', data=context.down_t)
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

    # catch models with excessive fluctuations in weights across laps:
    result['weights_path_distance'] = \
        weights_path_distance_exceeds_threshold(delta_weights_snapshots, context.weights_path_distance_threshold,
                                                cumulative=True, return_value=True)

    return {context.induction: result}


def plot_model_summary_figure(export_file_path=None, exported_data_key=None, induction_lap=0):
    """

    :param export_file_path: str (path)
    :param exported_data_key: str
    :param induction_lap: int
    """
    import matplotlib as mpl
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['font.size'] = 12.
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['axes.titlepad'] = 2.
    mpl.rcParams['mathtext.default'] = 'regular'
    mpl.rcParams['axes.unicode_minus'] = True

    if export_file_path is None:
        raise IOError('plot_model_summary_figure: no export_file_path provided')
    elif not os.path.isfile(export_file_path):
        raise IOError('plot_model_summary_figure: invalid export_file_path: %s' % export_file_path)
    initial_weights = dict()
    initial_ramp = dict()
    model_ramp = dict()
    final_weights = dict()
    induction_start_loc = dict()
    induction_stop_loc = dict()
    induction_start_times = dict()
    induction_stop_times = dict()
    description = 'model_ramp_features'
    with h5py.File(export_file_path, 'r') as f:
        source = get_h5py_group(f, [exported_data_key, 'exported_data', 'synthetic'])
        group = source['1'][description]
        x = group['param_array'][:]
        if 'local_signal_peak' not in group.attrs or 'global_signal_peak' not in group.attrs:
            raise KeyError('plot_model_summary_figure: missing required attributes from file: %s' %
                           export_file_path)
        local_signal_peak = group.attrs['local_signal_peak']
        global_signal_peak = group.attrs['global_signal_peak']
        local_signal_filter_t = group['local_signal_filter_t'][:]
        local_signal_filter = group['local_signal_filter'][:]
        global_filter_t = group['global_filter_t'][:]
        global_filter = group['global_filter'][:]
        for induction_key in source:
            group = source[induction_key][description]
            initial_weights[int(induction_key)] = group['initial_weights'][:]
            initial_ramp[int(induction_key)] = group['initial_model_ramp'][:]
            model_ramp[int(induction_key)] = group['model_ramp'][:]
            final_weights[int(induction_key)] = group['model_weights'][:]
            induction_start_loc[int(induction_key)] = group.attrs['mean_induction_start_loc']
            induction_stop_loc[int(induction_key)] = group.attrs['mean_induction_stop_loc']
            induction_start_times[int(induction_key)] = group.attrs['induction_start_times']
            induction_stop_times[int(induction_key)] = group.attrs['induction_stop_times']

    load_data(1)
    this_min_induction_t = context.min_induction_t
    this_clean_induction_t_indexes = context.clean_induction_t_indexes
    this_induction_start_time = induction_start_times[1][0]
    induction = 1
    update_source_contexts(x)

    global_signal = np.divide(get_global_signal(context.down_induction_gate, global_filter), global_signal_peak)
    local_signals = \
        np.divide(get_local_signal_population(local_signal_filter,
                                              context.down_rate_maps / context.input_field_peak_rate),
                  local_signal_peak)

    signal_xrange = np.linspace(0., 1., 10000)
    pot_rate = np.vectorize(scaled_single_sigmoid(
        context.f_pot_th, context.f_pot_th + context.f_pot_half_width, signal_xrange))
    dep_rate = np.vectorize(scaled_single_sigmoid(
        context.f_dep_th, context.f_dep_th + context.f_dep_half_width, signal_xrange))

    fig, axes = plt.subplots(3, 3, figsize=(10., 11.))

    min_t_val = np.min(this_min_induction_t)
    max_t_val = np.max(this_min_induction_t)
    start_index = np.where(context.complete_t >= this_induction_start_time + min_t_val)[0][0]
    center_index = np.where(context.complete_t >= this_induction_start_time)[0][0]
    end_index = np.where(context.complete_t > this_induction_start_time + max_t_val)[0][0]
    this_t = context.complete_t[start_index:end_index] - this_induction_start_time
    example_input_index = np.where(context.peak_locs >=  context.complete_position[center_index])[0][0]
    example_input_rate_map = context.complete_rate_maps[example_input_index][start_index:end_index] / \
                             context.input_field_peak_rate
    xlim = (this_t[0] / 1000., this_t[-1] / 1000.)
    xticks = np.arange(-3., 4., 1.5)
    this_axis = axes[0][0]
    this_axis.plot(this_t / 1000., example_input_rate_map, c='k')
    this_axis.set_ylabel('Presynaptic firing\nrate (normalized)')
    this_axis.set_title(r'$R_i$', fontsize=mpl.rcParams['font.size'] + 2)
    this_axis.set_xlim(xlim)
    this_axis.set_xticks(xticks)

    this_axis = axes[2][0]
    this_axis.plot(this_t / 1000., context.induction_gate[start_index:end_index], c='k')
    this_axis.set_ylabel('Plateau potential\namplitude (normalized)')
    this_axis.set_ylim([-0.2, 1.2])
    this_axis.set_xlim(xlim)
    this_axis.set_xticks(xticks)
    this_axis.set_title(r'$P$', fontsize=mpl.rcParams['font.size'] + 2)
    this_axis.set_xlabel('Time relative to\nplateau onset (s)')

    this_axis = axes[0][2]
    this_axis.plot(local_signal_filter_t / 1000., local_signal_filter / np.max(local_signal_filter), color='k')
    this_axis.set_xlabel('Time (s)')
    this_axis.set_ylabel('Normalized amplitude')
    this_axis.set_xlim(-0.25, max(5000., local_signal_filter_t[-1], global_filter_t[-1]) / 1000.)
    this_axis.set_title(r'$E_i$ filter', fontsize=mpl.rcParams['font.size'] + 2)

    this_axis = axes[1][2]
    this_axis.plot(global_filter_t / 1000., global_filter / np.max(global_filter), color='k')
    this_axis.set_xlabel('Time (s)')
    this_axis.set_ylabel('Normalized amplitude')
    this_axis.set_xlim(-0.25, max(5000., local_signal_filter_t[-1], global_filter_t[-1]) / 1000.)
    this_axis.set_title(r'$I$ filter', fontsize=mpl.rcParams['font.size'] + 2)

    this_axis = axes[1][1]
    dep_scale = context.k_dep / context.k_pot
    this_axis.plot(signal_xrange, pot_rate(signal_xrange), c='r', label=r'$q^+$')
    this_axis.plot(signal_xrange, dep_rate(signal_xrange) * dep_scale, c='c', label=r'$q^-$')
    this_axis.set_xlabel('Plasticity signal overlap (a.u.)')
    this_axis.set_ylabel('dW/dt')
    this_axis.set_yticks([dep_scale, 1.])
    this_axis.set_yticklabels([r'$W \cdot k^-$', r'$(1-W) \cdot k^+$'])
    this_axis.set_xlim((0., 1.))
    this_axis.legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, fontsize=mpl.rcParams['font.size'])

    xlim = (np.min(this_min_induction_t[this_clean_induction_t_indexes] / 1000.),
            np.max(this_min_induction_t[this_clean_induction_t_indexes] / 1000.))
    this_axis = axes[2][1]
    this_axis.plot(this_min_induction_t[this_clean_induction_t_indexes] / 1000.,
                   np.subtract(model_ramp[induction], initial_ramp[induction])[
                       this_clean_induction_t_indexes], color='k')
    this_axis.plot(this_min_induction_t[this_clean_induction_t_indexes], np.zeros_like(this_clean_induction_t_indexes),
                   '--', c='grey', alpha=0.5)
    this_axis.set_xlabel('Time relative to\nplateau onset (s)')
    this_axis.set_ylabel('Change in ramp\namplitude (mV)')
    this_axis.set_ylim([-6., 12.])
    this_axis.set_xlim(xlim)
    this_axis.set_xticks(xticks)
    this_axis.set_title('Silent -> Place1', fontsize=mpl.rcParams['font.size'] + 2)

    this_axis = axes[2][2]
    load_data(2)
    this_min_induction_t = context.min_induction_t
    this_clean_induction_t_indexes = context.clean_induction_t_indexes
    induction = 2
    this_axis.plot(this_min_induction_t[this_clean_induction_t_indexes] / 1000.,
                   np.subtract(model_ramp[induction], initial_ramp[induction])[
                       this_clean_induction_t_indexes], color='k')
    this_axis.plot(this_min_induction_t[this_clean_induction_t_indexes], np.zeros_like(this_clean_induction_t_indexes),
                   '--', c='grey', alpha=0.5)
    this_axis.set_xlabel('Time relative to\nplateau onset (s)')
    this_axis.set_ylabel('Change in ramp\namplitude (mV)')
    this_axis.set_ylim([-6., 12.])
    this_axis.set_xlim(xlim)
    this_axis.set_xticks(xticks)
    this_axis.set_title('Place1 -> Place2', fontsize=mpl.rcParams['font.size'] + 2)

    clean_axes(axes)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5, wspace=0.4)
    fig.show()
    context.update(locals())


def get_args_static_model_ramp():
    """
    A nested map operation is required to compute model_ramp features. The arguments to be mapped are the same
    (static) for each set of parameters.
    :param x: array
    :return: list of list
    """
    return [[1, 2]]


def compute_features_model_ramp(x, induction=None, model_id=None, export=False, plot=False):
    """

    :param x: array
    :param induction: str
    :param model_id: int
    :param export: bool
    :param plot: bool
    :return: dict
    """
    load_data(induction)
    update_source_contexts(x, context)
    start_time = time.time()
    if context.disp:
        print('Process: %i: computing model_ramp features for induction: %s with x: %s' % \
              (os.getpid(), induction, ', '.join('%.3E' % i for i in x)))
        sys.stdout.flush()
    result = calculate_model_ramp(export=export, plot=plot, model_id=model_id)
    if context.disp:
        print('Process: %i: computing model_ramp features for induction: %s took %.1f s' % \
              (os.getpid(), induction, time.time() - start_time))
        sys.stdout.flush()
    return result


def filter_features_model_ramp(primitives, current_features, model_id=None, export=False):
    """

    :param primitives: list of dict (each dict contains results from a single simulation)
    :param current_features: dict
    :param model_id: int
    :param export: bool
    :return: dict
    """
    features = {}
    groups = ['induction1', 'induction2']
    grouped_feature_names = ['delta_val_at_target_peak', 'delta_val_at_model_peak', 'delta_width', 'delta_peak_shift',
                             'delta_asymmetry', 'delta_min_loc', 'delta_val_at_target_min', 'delta_val_at_model_min',
                             'residual_score']
    feature_names = ['ramp_amp_after_first_plateau', 'weights_path_distance']
    for this_result_dict in primitives:
        if not this_result_dict:
            if context.verbose > 0:
                print('optimize_biBTSP_%s: filter_features_model_ramp: pid: %i; model failed' %
                      (BTSP_model_name, os.getpid()))
                sys.stdout.flush()
            return dict()
        for induction in this_result_dict:
            induction = int(induction)
            group = 'induction' + str(induction)
            for feature_name in grouped_feature_names:
                key = group + '_' + feature_name
                if key not in features:
                    features[key] = []
                features[key].append(this_result_dict[induction][feature_name])
            for feature_name in feature_names:
                if feature_name in this_result_dict[induction]:
                    if feature_name not in features:
                        features[feature_name] = []
                    features[feature_name].append(this_result_dict[induction][feature_name])

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


def get_objectives(features, model_id=None, export=False):
    """

    :param features: dict
    :param model_id: int
    :param export: bool
    :return: tuple of dict
    """
    objectives = {}

    grouped_feature_names = ['residual_score']
    groups = ['induction1', 'induction2']
    for feature_name in grouped_feature_names:
        for group in groups:
            objective_name = group + '_' + feature_name
            if objective_name in features:
                objectives[objective_name] = features[objective_name]

    feature_name = 'ramp_amp_after_first_plateau'
    if feature_name in context.objective_names and feature_name in features:
        if features[feature_name] < context.target_val[feature_name]:
            objectives[feature_name] = ((features[feature_name] - context.target_val[feature_name]) /
                                        context.target_range[feature_name]) ** 2.
        else:
            objectives[feature_name] = 0.

    feature_name = 'weights_path_distance'
    if feature_name in context.objective_names and feature_name in features:
        objectives[feature_name] = (features[feature_name] / context.target_range[feature_name]) ** 2.

    for objective_name in context.objective_names:
        if objective_name not in objectives:
            return dict(), dict()
        else:
            objectives[objective_name] = np.mean(objectives[objective_name])

    return features, objectives


def get_features_interactive(interface, x, model_id=None, plot=False):
    """

    :param interface: :class: 'IpypInterface', 'MPIFuturesInterface', 'ParallelContextInterface', or 'SerialInterface'
    :param x:
    :param model_id: int
    :param plot:
    :return: dict
    """
    features = {}
    args = interface.execute(get_args_static_model_ramp)
    group_size = len(args[0])
    sequences = [[x] * group_size] + args + [[model_id] * group_size] + [[context.export] * group_size] + \
                [[plot] * group_size]
    primitives = interface.map(compute_features_model_ramp, *sequences)
    new_features = interface.execute(filter_features_model_ramp, primitives, features, model_id, context.export)
    features.update(new_features)

    return features


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_biBTSP_synthetic_config.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--verbose", type=int, default=2)
@click.option("--plot", is_flag=True)
@click.option("--interactive", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--plot-summary-figure", is_flag=True)
@click.option("--exported-data-key", type=str, default=None)
@click.pass_context
def main(cli, config_file_path, output_dir, export, export_file_path, label, verbose, plot, interactive, debug,
         plot_summary_figure, exported_data_key):
    """
    To execute on a single process on one cell from the experimental dataset:
    python -i optimize_biBTSP_synthetic.py --plot --framework=serial --interactive

    To execute using MPI parallelism with 1 controller process and N - 1 worker processes:
    mpirun -n N python -i -m mpi4py.futures optimize_biBTSP_synthetic.py --plot --framework=mpi --interactive

    To optimize the models by running many instances in parallel:
    mpirun -n N python -m mpi4py.futures -m nested.optimize --config-file-path=$PATH_TO_CONFIG_FILE --disp --export \
        --framework=mpi --pop_size=200 --path_length=3 --max_iter=50

    To plot results previously exported to a file on a single process:
    python -i optimize_biBTSP_synthetic.py --plot-summary-figure --model-file-path=$PATH_TO_MODEL_FILE \
        --framework=serial --interactive

    :param cli: contains unrecognized args as list of str
    :param config_file_path: str (path)
    :param output_dir: str (path)
    :param export: bool
    :param export_file_path: str
    :param label: str
    :param verbose: int
    :param plot: bool
    :param interactive: bool
    :param debug: bool
    :param plot_summary_figure: bool
    :param exported_data_key: str
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    kwargs = get_unknown_click_arg_dict(cli.args)
    context.disp = verbose > 0

    context.interface = get_parallel_interface(source_file=__file__, source_package=__package__, **kwargs)
    context.interface.start(disp=context.disp)
    context.interface.ensure_controller()
    config_optimize_interactive(__file__, config_file_path=config_file_path, output_dir=output_dir,
                                export=export, export_file_path=export_file_path, label=label,
                                disp=context.disp, interface=context.interface, verbose=verbose, plot=plot, **kwargs)

    if plot_summary_figure:
        context.interface.execute(plot_model_summary_figure, export_file_path, exported_data_key)
        context.interface.execute(plt.show)
    elif not debug:
        model_id = 0

        features = get_features_interactive(context.interface, context.x0_array, model_id=model_id, plot=plot)
        features, objectives = context.interface.execute(get_objectives, features, model_id, context.export)
        if export:
            if 'model_key' in context() and context.model_key is not None:
                model_label = context.model_key
            else:
                model_label = 'x0'
            legend = {'model_labels': [model_label], 'export_keys': [context.exported_data_key],
                      'source': context.config_file_path}
            merge_exported_data(context, export_file_path=context.export_file_path,
                                output_dir=context.output_dir, legend=legend, verbose=context.disp)

        sys.stdout.flush()
        print('model_id: %i; model_labels: %s' % (model_id, model_label))
        print('params:')
        pprint.pprint(context.x0_dict)
        print('features:')
        pprint.pprint(features)
        print('objectives:')
        pprint.pprint(objectives)
        sys.stdout.flush()
        time.sleep(.1)

        if plot:
            context.interface.apply(plt.show)

    if context.interactive:
        context.update(locals())
    else:
        context.interface.stop()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
