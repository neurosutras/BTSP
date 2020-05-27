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
2) Activity at each synapse generates long duration 'local plasticity signals', or 'eligibility traces' for synaptic
plasticity.
3) Dendritic plateaus generate a long duration 'global plasticity signal', or an 'instructive trace' for synaptic
plasticity.
4) Changes in weight at each synapse are integrated over periods of nonzero overlap between eligibility and instructive
signals, and updated once per lap.

Features/assumptions of voltage-dependent model D:
1) Dendritic plateaus generate a global instructive signal that provides a necessary cofactor required to convert
plasticity eligibility signals at each synapse into either increases or decreases in synaptic strength.
2) Activity at each synapse generates a local potentiation eligibility signal that, in conjunction with the global
gating signal, can activate a forward process to increase synaptic strength. The amplitude of this eligibility signal
is inversely proportional to the postsynaptic voltage at the time of a presynaptic spike.
3) Activity at each synapse generates a local depression eligibility signal that, in conjunction with the global
gating signal, can activate a reverse process to decrease synaptic strength. The amplitude of this eligibility signal
is proportional to the postsynaptic voltage at the time of a presynaptic spike.
4) global_signals are pooled across all cells and normalized to a peak value of 1.
5) local_signals are pooled across all cells and normalized to a peak value of 1.
6) f_pot represents the "sensitivity" of the forward process to the presence of the pot_signal. The transformation
f_pot has the flexibility to be any segment of a sigmoid (so can be linear, exponential rise, or saturating).
7) f_dep represents the "sensitivity" of the reverse process to the presence of the dep_signal. The transformation
f_dep has the flexibility to be any segment of a sigmoid (so can be linear, exponential rise, or saturating).

biBTSP_VD_D: Single eligibility signal filter. Sigmoidal f_pot and f_dep.
L_pot ~ local_signal_filter(pre_rate * (1. - V / V_max))
L_dep ~ local_signal_filter(pre_rate *  V / V_max)
dW/dt ~ k_pot * f_pot(L_pot * global_signal) - k_dep * f_dep(L_dep * global_signal)

"""
__author__ = 'milsteina'
from biBTSP_utils import *
from nested.parallel import *
from nested.optimize_utils import *
import click

context = Context()


BTSP_model_name = 'VD_D'


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
        default_run_vel = f['defaults'].attrs['default_run_vel']  # cm/s
        generic_position_dt = f['defaults'].attrs['generic_position_dt']  # ms
        default_interp_dx = f['defaults'].attrs['default_interp_dx']  # cm
        binned_x = f['defaults']['binned_x'][:]
        generic_x = f['defaults']['generic_x'][:]
        generic_t = f['defaults']['generic_t'][:]
        default_interp_t = f['defaults']['default_interp_t'][:]
        default_interp_x = f['defaults']['default_interp_x'][:]
        extended_x = f['defaults']['extended_x'][:]

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

        all_data_keys = [(int(cell_id), int(induction)) for cell_id in f['data'] for induction in f['data'][cell_id]]
        if 'data_keys' not in context() or context.data_keys is None:
            if 'cell_id' not in context() or context.cell_id == 'all' or context.cell_id is None:
                context.data_keys = all_data_keys
            else:
                context.data_keys = \
                    [(int(context.cell_id), int(induction)) for induction in f['data'][str(context.cell_id)]]
        else:
            context.data_keys = [(int(cell_id), int(induction))
                                 for cell_id in [cell_id for cell_id in context.data_keys if str(cell_id) in f['data']]
                                 for induction in f['data'][str(cell_id)]]
        spont_cell_id_list = [int(cell_id) for cell_id in f['data'] if f['data'][cell_id].attrs['spont']]

    if context.verbose > 1:
        print('optimize_biBTSP_%s: pid: %i; processing the following data_keys: %s' %
              (BTSP_model_name, os.getpid(), str(context.data_keys)))
    largest_signal_amplitude_data_keys = \
        [(6, 1), (18, 1), (23, 1), (24, 1), (25, 1), (31, 2), (46, 2), (5, 2), (16, 1)]
    down_dt = 10.  # ms, to speed up optimization
    context.update(locals())
    context.cell_id = None
    context.induction = None
    context.data_cache = defaultdict(dict)


def import_data(cell_id, induction):
    """

    :param cell_id: int
    :param induction: int
    """
    cell_id = int(cell_id)
    induction = int(induction)
    if cell_id == context.cell_id and induction == context.induction:
        return

    if cell_id in context.data_cache and induction in context.data_cache[cell_id]:
        induction_context = context.data_cache[cell_id][induction]
    else:
        cell_key = str(cell_id)
        induction_key = str(induction)

        induction_context = Context()

        with h5py.File(context.data_file_path, 'r') as f:
            if cell_key not in f['data'] or induction_key not in f['data'][cell_key]:
                raise KeyError('optimize_biBTSP_%s: no data found for cell_id: %s, induction: %s' %
                               (BTSP_model_name, cell_key, induction_key))
            data_group = f['data'][cell_key][induction_key]
            induction_context.induction_locs = data_group.attrs['induction_locs']
            induction_context.induction_durs = data_group.attrs['induction_durs']
            induction_context.exp_ramp_raw = {}
            induction_context.exp_ramp = {}
            if 'before' in data_group['raw']['exp_ramp']:
                induction_context.exp_ramp_raw['before'] = data_group['raw']['exp_ramp']['before'][:]
            induction_context.position = {}
            induction_context.t = {}
            induction_context.current = []
            for category in data_group['processed']['position']:
                induction_context.position[category] = []
                induction_context.t[category] = []
                for i in range(len(data_group['processed']['position'][category])):
                    lap_key = str(i)
                    induction_context.position[category].append(
                        data_group['processed']['position'][category][lap_key][:])
                    induction_context.t[category].append(data_group['processed']['t'][category][lap_key][:])
            for i in range(len(data_group['processed']['current'])):
                lap_key = str(i)
                induction_context.current.append(data_group['processed']['current'][lap_key][:])
            induction_context.mean_position = data_group['processed']['mean_position'][:]
            induction_context.mean_t = data_group['processed']['mean_t'][:]
            induction_context.min_induction_t = data_group['processed']['min_induction_t'][:]
            induction_context.clean_induction_t_indexes = \
                get_clean_induction_t_indexes(induction_context.min_induction_t, context.truncate * 1000.)

            if 'before' in data_group['processed']['exp_ramp']:
                induction_context.exp_ramp['before'] = data_group['processed']['exp_ramp']['before'][:]
            induction_context.LSA_ramp = {}
            induction_context.LSA_ramp_offset = {}
            calibrated_input_group = f['calibrated_input'][context.input_field_width_key][cell_key][induction_key]
            if 'before' in calibrated_input_group['LSA_ramp']:
                induction_context.LSA_ramp['before'] = calibrated_input_group['LSA_ramp']['before'][:]
                induction_context.LSA_ramp_offset['before'] = \
                    calibrated_input_group['LSA_ramp']['before'].attrs['ramp_offset']
            induction_context.LSA_weights = {}
            induction_context.LSA_weights['before'] = calibrated_input_group['LSA_weights']['before'][:]
            induction_context.complete_run_vel = data_group['complete']['run_vel'][:]
            induction_context.complete_run_vel_gate = data_group['complete']['run_vel_gate'][:]
            induction_context.complete_position = data_group['complete']['position'][:]
            induction_context.complete_t = data_group['complete']['t'][:]
            induction_context.induction_gate = data_group['complete']['induction_gate'][:]
            induction_context.exp_ramp_raw['after'] = data_group['raw']['exp_ramp']['after'][:]
            induction_context.exp_ramp['after'] = data_group['processed']['exp_ramp']['after'][:]
            induction_context.LSA_ramp['after'] = calibrated_input_group['LSA_ramp']['after'][:]
            induction_context.LSA_ramp_offset['after'] = calibrated_input_group['LSA_ramp']['after'].attrs[
                'ramp_offset']
            induction_context.LSA_weights['after'] = calibrated_input_group['LSA_weights']['after'][:]
            induction_context.peak_ramp_amp = 0.
            for this_induction_key in f['data'][cell_key]:
                if 'before' in f['data'][cell_key][this_induction_key]['processed']['exp_ramp']:
                    induction_context.peak_ramp_amp = \
                        max(induction_context.peak_ramp_amp,
                            np.max(f['data'][cell_key][this_induction_key]['processed']['exp_ramp']['before'][:]))
                if 'after' in f['data'][cell_key][this_induction_key]['processed']['exp_ramp']:
                    induction_context.peak_ramp_amp = \
                        max(induction_context.peak_ramp_amp,
                            np.max(f['data'][cell_key][this_induction_key]['processed']['exp_ramp']['after'][:]))
        context.data_cache[cell_id][induction] = induction_context
    context.update(induction_context())

    context.mean_induction_start_loc = np.mean(context.induction_locs)
    context.mean_induction_dur = np.mean(context.induction_durs)
    mean_induction_start_index = \
        np.where(context.mean_position >= context.mean_induction_start_loc)[0][0]
    mean_induction_stop_index = \
        np.where(context.mean_t >= context.mean_t[mean_induction_start_index] +
                 context.mean_induction_dur)[0][0]
    context.mean_induction_stop_loc = context.mean_position[mean_induction_stop_index]
    induction_start_times = []
    induction_stop_times = []
    track_start_times = []
    track_stop_times = []
    running_position = 0.
    running_t = 0.
    for i, (this_induction_loc, this_induction_dur) in enumerate(
            zip(context.induction_locs, context.induction_durs)):
        this_induction_start_index = \
            np.where(context.complete_position >= this_induction_loc + running_position)[0][0]
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
    context.complete_rate_maps = \
        get_complete_rate_maps(context.input_rate_maps, context.binned_x, context.position,
                               context.complete_run_vel_gate)
    context.down_t = np.arange(context.complete_t[0], context.complete_t[-1] + context.down_dt / 2., context.down_dt)
    context.down_rate_maps = []
    for rate_map in context.complete_rate_maps:
        this_down_rate_map = np.interp(context.down_t, context.complete_t, rate_map)
        context.down_rate_maps.append(this_down_rate_map)
    context.down_induction_gate = np.interp(context.down_t, context.complete_t,
                                                      context.induction_gate)
    context.cell_id = cell_id
    context.induction = induction

    if context.verbose > 1:
        print('optimize_biBTSP_%s: process: %i loaded data for cell: %i, induction: %i' %
              (BTSP_model_name, os.getpid(), cell_id, induction))


def update_model_params(x, local_context):
    """

    :param x: array
    :param local_context: :class:'Context'
    """
    if local_context is None:
        local_context = context
    local_context.update(param_array_to_dict(x, local_context.param_names))


def get_args_static_signal_amplitudes():
    """
    A nested map operation is required to compute model signal amplitudes. The arguments to be mapped are the same
    (static) for each set of parameters.
    :return: list of list
    """
    # return list(zip(*context.all_data_keys))
    return list(zip(*context.largest_signal_amplitude_data_keys))


def compute_features_signal_amplitudes(x, cell_id=None, induction=None, model_id=None, export=False, plot=False):
    """

    :param x: array
    :param cell_id: int
    :param induction: int
    :param model_id: int or str
    :param export: bool
    :param plot: bool
    :return: dict
    """
    import_data(cell_id, induction)
    update_source_contexts(x, context)
    if context.verbose > 1:
        print('Process: %i: computing signal_amplitude features for model_id: %s, cell_id: %i, induction: %i with x: '
              '%s' % (os.getpid(), model_id, context.cell_id, context.induction, ', '.join('%.3E' % i for i in x)))
        sys.stdout.flush()
    start_time = time.time()
    local_signal_filter_t, local_signal_filter, global_filter_t, global_filter = \
        get_dual_exp_decay_signal_filters(context.local_signal_decay, context.global_signal_decay, context.down_dt)
    global_signal = get_global_signal(context.down_induction_gate, global_filter)
    local_signals = get_local_signal_population(local_signal_filter,
                                                context.down_rate_maps / context.input_field_peak_rate)

    result = {'local_signal_peak': np.max(local_signals),
              'global_signal_peak': np.max(global_signal)}
    if context.verbose > 1:
        print('Process: %i: computing signal_amplitude features for model_id: %s, cell_id: %i, induction: %i took '
              '%.1f s' % (os.getpid(), model_id, context.cell_id, context.induction, time.time() - start_time))
        sys.stdout.flush()
    return {cell_id: {induction: result}}


def filter_features_signal_amplitudes(primitives, current_features, model_id=None, export=False, plot=False):
    """

    :param primitives: list of dict (each dict contains results from a single simulation)
    :param current_features: dict
    :param model_id: int or str
    :param export: bool
    :param plot: bool
    :return: dict
    """
    local_signal_peaks = []
    global_signal_peaks = []
    data_key_list = []
    for this_dict in primitives:
        for cell_id in this_dict:
            for induction_id in this_dict[cell_id]:
                global_signal_peaks.append(this_dict[cell_id][induction_id]['global_signal_peak'])
                local_signal_peaks.append(this_dict[cell_id][induction_id]['local_signal_peak'])
                data_key_list.append((cell_id, induction_id))
    if context.verbose > 0:
        index = np.argmax(global_signal_peaks)
        this_data_key = data_key_list[index]
        if this_data_key not in context.largest_signal_amplitude_data_keys:
            print('WARNING: cell: %i, induction: %i has largest global_signal_peak' %
                  (this_data_key[0], this_data_key[1]))
        index = np.argmax(local_signal_peaks)
        this_data_key = data_key_list[index]
        if this_data_key not in context.largest_signal_amplitude_data_keys:
            print('WARNING: cell: %i, induction: %i has largest local_signal_peak' %
                  (this_data_key[0], this_data_key[1]))
        sys.stdout.flush()
    if plot:
        fig, axes = plt.subplots(1)
        hist, edges = np.histogram(local_signal_peaks, bins=min(10, len(primitives)), density=True)
        bin_width = edges[1] - edges[0]
        axes.plot(edges[:-1] + bin_width / 2., hist * bin_width, c='r', label='Local eligibility signals')
        axes.set_xlabel('Peak eligibility signal amplitudes (a.u.)')
        axes.set_ylabel('Probability')
        axes.set_title('Eligibility signal amplitude distribution')
        axes.legend(loc='best', frameon=False, framealpha=0.5)
        clean_axes(axes)
        fig.tight_layout()
        fig.show()

        hist, edges = np.histogram(global_signal_peaks, bins=min(10, len(primitives)), density=True)
        bin_width = edges[1] - edges[0]
        fig, axes = plt.subplots(1)
        axes.plot(edges[:-1] + bin_width / 2., hist * bin_width, c='k')
        axes.set_xlabel('Peak instructive signal amplitudes (a.u.)')
        axes.set_ylabel('Probability')
        axes.set_title('Instructive signal amplitude distribution')
        clean_axes(axes)
        fig.tight_layout()
        fig.show()
    signal_amplitude_features = {'local_signal_max': np.max(local_signal_peaks),
                                 'global_signal_max': np.max(global_signal_peaks)
                                 }
    if context.verbose > 1:
        print('Process: %i: signal_amplitude features; model_id: %s; local_signal_max: %.2E, global_signal_max: %.2E' %
              (os.getpid(), model_id, signal_amplitude_features['local_signal_max'],
               signal_amplitude_features['global_signal_max']))
    return signal_amplitude_features


def calculate_model_ramp(local_signal_peak=None, global_signal_peak=None, model_id=None, export=False, plot=False):
    """

    :param local_signal_peak: float
    :param global_signal_peak: float
    :param model_id: int or str
    :param export: bool
    :param plot: bool
    :return: dict
    """
    local_signal_filter_t, local_signal_filter, global_filter_t, global_filter = \
        get_dual_exp_decay_signal_filters(context.local_signal_decay, context.global_signal_decay,
                                          context.down_dt)
    global_signal = np.divide(get_global_signal(context.down_induction_gate, global_filter), global_signal_peak)

    # BCM-like voltage dependence; linear
    signal_xrange = np.linspace(0., 1., 10000)
    pot_phi = np.vectorize(lambda x: min(1., max(0., (1. - x))))
    dep_phi = np.vectorize(lambda x: min(1., max(0., x)))

    pot_rate = np.vectorize(scaled_single_sigmoid(
        context.f_pot_th, context.f_pot_th + context.f_pot_half_width, signal_xrange))
    dep_rate = np.vectorize(scaled_single_sigmoid(
        context.f_dep_th, context.f_dep_th + context.f_dep_half_width, signal_xrange))

    if plot and context.induction == 1:
        fig, axes = plt.subplots(1, 3, figsize=(10., 4.))
        axes[0].plot(signal_xrange, pot_phi(signal_xrange), c='r', label='Potentiation')
        axes[0].plot(signal_xrange, dep_phi(signal_xrange), c='c', label='Depression')
        axes[0].set_xlabel('Ramp amplitude (normalized)')
        axes[0].set_ylabel('Eligibility modulation factor')
        axes[0].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
        dep_scale = context.k_dep / context.k_pot
        axes[2].plot(signal_xrange, pot_rate(signal_xrange), c='r', label='Potentiation rate')
        axes[2].plot(signal_xrange, dep_rate(signal_xrange) * dep_scale, c='c', label='Depression rate')
        axes[2].set_xlabel('Plasticity signal overlap (a.u.)')
        axes[2].set_ylabel('Normalized rate')
        axes[2].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
        axes[1].plot(local_signal_filter_t / 1000., local_signal_filter / np.max(local_signal_filter), color='r',
                     label='Eligibility signal filter')
        axes[1].plot(global_filter_t / 1000., global_filter / np.max(global_filter), color='k',
                     label='Instructive signal filter')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Normalized amplitude')
        axes[1].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
        axes[1].set_xlim(-0.5, max(5000., local_signal_filter_t[-1], global_filter_t[-1]) / 1000.)
        clean_axes(axes)
        fig.tight_layout()
        fig.show()

    initial_delta_weights = context.LSA_weights['before']

    if 'before' in context.exp_ramp:
        initial_ramp = context.LSA_ramp['before']
    else:
        initial_ramp, discard_ramp_offset = \
            get_model_ramp(initial_delta_weights, ramp_x=context.binned_x, input_x=context.binned_x,
                           input_rate_maps=context.input_rate_maps, ramp_scaling_factor=context.ramp_scaling_factor)

    current_delta_weights = initial_delta_weights
    delta_weights_snapshots = [current_delta_weights]
    current_ramp = initial_ramp
    ramp_snapshots = [current_ramp]

    target_ramp = context.exp_ramp['after']

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(10., 4.))
        fig.suptitle('Induction: %i' % (context.induction), y=1.)
        axes[0].set_ylabel('Instructive signal amplitude (a.u.)')
        axes[0].set_xlabel('Time (s)')
        axes[1].set_ylabel('Integrated plasticity signal overlap (a.u.)')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_title('Potentiation')
        axes[2].set_ylabel('Integrated plasticity signal overlap (a.u.)')
        axes[2].set_title('Depression')
        axes[2].set_xlabel('Time (s)')

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

    this_peak_ramp_amp = context.peak_ramp_amp + context.plateau_delta_depo

    for induction_lap in range(len(context.induction_start_times)):
        current_complete_ramp = get_complete_ramp(current_ramp, context.binned_x, context.position,
                                                  context.complete_run_vel_gate, context.induction_gate,
                                                  this_peak_ramp_amp)
        current_complete_normalized_ramp = \
            np.divide(np.interp(context.down_t, context.complete_t, current_complete_ramp), this_peak_ramp_amp)
        pot_eligibility_signals = np.divide(get_voltage_dependent_eligibility_signal_population(
            local_signal_filter, current_complete_normalized_ramp, pot_phi,
            context.down_rate_maps / context.input_field_peak_rate), local_signal_peak)
        dep_eligibility_signals = np.divide(get_voltage_dependent_eligibility_signal_population(
            local_signal_filter, current_complete_normalized_ramp, dep_phi,
            context.down_rate_maps / context.input_field_peak_rate), local_signal_peak)

        start_time = context.induction_start_times[induction_lap]
        if induction_lap == len(context.induction_start_times) - 1:
            stop_time = context.down_t[-1]
        else:
            stop_time = context.induction_start_times[induction_lap + 1]
        indexes = np.where((context.down_t >= start_time) & (context.down_t < stop_time))[0]

        next_delta_weights = []
        pot_overlap = []
        dep_overlap = []
        for i, (this_pot_signal, this_dep_signal) in \
                enumerate(zip(pot_eligibility_signals, dep_eligibility_signals)):
            this_pot_signal_overlap = np.multiply(this_pot_signal[indexes], global_signal[indexes])
            this_dep_signal_overlap = np.multiply(this_dep_signal[indexes], global_signal[indexes])
            this_pot_rate = np.trapz(pot_rate(this_pot_signal_overlap), dx=context.down_dt / 1000.)
            this_dep_rate = np.trapz(dep_rate(this_dep_signal_overlap), dx=context.down_dt / 1000.)
            this_delta_weight = context.k_pot * this_pot_rate - context.k_dep * this_dep_rate
            next_delta_weights.append(max(current_delta_weights[i] + this_delta_weight, -1.))
            pot_overlap.append(np.trapz(this_pot_signal_overlap, dx=context.down_dt / 1000.))
            dep_overlap.append(np.trapz(this_dep_signal_overlap, dx=context.down_dt / 1000.))
        if plot:
            interp_pot_overlap = np.interp(context.binned_x, context.peak_locs, pot_overlap)
            interp_dep_overlap = np.interp(context.binned_x, context.peak_locs, dep_overlap)
            axes[1].plot(context.min_induction_t[context.clean_induction_t_indexes] / 1000.,
                         interp_pot_overlap[context.clean_induction_t_indexes])
            axes[2].plot(context.min_induction_t[context.clean_induction_t_indexes] / 1000.,
                         interp_dep_overlap[context.clean_induction_t_indexes],
                         label='Induction lap: %i' % (induction_lap + 1))
            axes[0].plot(context.down_t[indexes] / 1000., global_signal[indexes])
            axes2[1].plot(context.peak_locs, np.subtract(next_delta_weights, current_delta_weights),
                          label='Induction lap: %i' % (induction_lap + 1))
        current_delta_weights = np.array(next_delta_weights)
        delta_weights_snapshots.append(current_delta_weights)
        current_ramp, discard_ramp_offset = \
            get_model_ramp(current_delta_weights, ramp_x=context.binned_x, input_x=context.binned_x,
                           input_rate_maps=context.input_rate_maps, ramp_scaling_factor=context.ramp_scaling_factor)

        if plot:
            axes2[0].plot(context.binned_x, current_ramp)
        ramp_snapshots.append(current_ramp)

    if plot:
        axes[2].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
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
    initial_weights = np.add(initial_delta_weights, 1.)
    final_weights = np.add(current_delta_weights, 1.)

    model_ramp, discard_delta_weights, model_ramp_offset, model_residual_score = \
        get_residual_score(current_delta_weights, target_ramp, ramp_x=context.binned_x, input_x=context.binned_x,
                           interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                           ramp_scaling_factor=context.ramp_scaling_factor,
                           induction_loc=context.mean_induction_start_loc, track_length=context.track_length,
                           target_range=context.target_range, full_output=True)

    result = {'residual_score': model_residual_score}

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
        print('Process: %i; model_id: %s; cell: %i; induction: %i:' %
              (os.getpid(), model_id, context.cell_id, context.induction))
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
            cell_key = str(context.cell_id)
            induction_key = str(context.induction)
            group = get_h5py_group(f, [model_id, exported_data_key, cell_key, induction_key, description], create=True)
            if 'before' in context.exp_ramp:
                group.create_dataset('initial_exp_ramp', compression='gzip', data=context.exp_ramp['before'])
            group.create_dataset('target_ramp', compression='gzip', data=target_ramp)
            group.create_dataset('initial_model_ramp', compression='gzip', data=initial_ramp)
            group.create_dataset('model_ramp', compression='gzip', data=model_ramp)
            group.create_dataset('model_weights', compression='gzip', data=final_weights)
            group.create_dataset('initial_weights', compression='gzip', data=initial_weights)
            group.create_dataset('global_signal', compression='gzip', data=global_signal)
            group.create_dataset('down_t', compression='gzip', data=context.down_t)
            group.create_dataset('pot_rate', compression='gzip', data=pot_rate(signal_xrange))
            group.create_dataset('dep_rate', compression='gzip', data=dep_rate(signal_xrange))
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

    # catch models with excessive fluctuations in weights across laps:
    result['weights_path_distance'] = \
        weights_path_distance_exceeds_threshold(delta_weights_snapshots, context.weights_path_distance_threshold,
                                                cumulative=True, return_value=True)

    return {context.cell_id: {context.induction: result}}


def plot_model_summary_supp_figure(cell_id, export_file_path=None, exported_data_key=None, induction_lap=0):
    """

    :param cell_id: int
    :param export_file_path: str (path)
    :param exported_data_key: str
    :param induction_lap: int
    """
    if (cell_id, 2) not in context.data_keys:
        raise KeyError('plot_model_summary_figure: cell_id: %i, induction: 2 not found' % cell_id)
    if export_file_path is None:
        raise IOError('plot_model_summary_figure: no export_file_path provided')
    elif not os.path.isfile(export_file_path):
        raise IOError('plot_model_summary_figure: invalid export_file_path: %s' % export_file_path)
    with h5py.File(export_file_path, 'r') as f:
        source = get_h5py_group(f, [exported_data_key, 'exported_data', str(cell_id), '2', 'model_ramp_features'])
        x = source['param_array'][:]
        if 'local_signal_peak' not in source.attrs or 'global_signal_peak' not in source.attrs:
            raise KeyError('plot_model_summary_figure: missing required attributes for cell_id: %i, '
                           'induction 2; from file: %s' % (cell_id, export_file_path))
        local_signal_peak = source.attrs['local_signal_peak']
        global_signal_peak = source.attrs['global_signal_peak']
        local_signal_filter_t = source['local_signal_filter_t'][:]
        local_signal_filter = source['local_signal_filter'][:]
        global_filter_t = source['global_filter_t'][:]
        global_filter = source['global_filter'][:]
        initial_weights = source['initial_weights'][:]
        initial_ramp = source['initial_model_ramp'][:]
        model_ramp = source['model_ramp'][:]
        ramp_snapshots = []
        for lap in range(len(source['ramp_snapshots'])):
            ramp_snapshots.append(source['ramp_snapshots'][str(lap)][:])
        delta_weights_snapshots = []
        for lap in range(len(source['delta_weights_snapshots'])):
            delta_weights_snapshots.append(source['delta_weights_snapshots'][str(lap)][:])
        final_weights = source['model_weights'][:]

    import_data(cell_id, 2)
    update_source_contexts(x)

    initial_exp_ramp = context.exp_ramp['before']
    target_ramp = context.exp_ramp['after']

    global_signal = np.divide(get_global_signal(context.down_induction_gate, global_filter), global_signal_peak)

    signal_xrange = np.linspace(0., 1., 10000)
    pot_phi = np.vectorize(lambda x: min(1., max(0., (1. - x))))
    dep_phi = np.vectorize(lambda x: min(1., max(0., x)))

    pot_rate = np.vectorize(scaled_single_sigmoid(
        context.f_pot_th, context.f_pot_th + context.f_pot_half_width, signal_xrange))
    dep_rate = np.vectorize(scaled_single_sigmoid(
        context.f_dep_th, context.f_dep_th + context.f_dep_half_width, signal_xrange))

    this_peak_ramp_amp = context.peak_ramp_amp + context.plateau_delta_depo

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

    target_time_delay = 2500.

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
    distant_depressing_indexes = input_sample_indexes[relative_indexes]
    if np.any(distant_depressing_indexes):
        relative_index = np.argmin(np.subtract(final_weights, initial_weights)[distant_depressing_indexes])
        this_example_index = distant_depressing_indexes[relative_index]
    else:
        relative_index = np.argmin(np.subtract(final_weights, initial_weights)[input_sample_indexes])
        this_example_index = input_sample_indexes[relative_index]
    example_input_dict['Depressing input example'] = this_example_index

    import matplotlib as mpl
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['font.size'] = 10.
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['axes.titlepad'] = 2.
    mpl.rcParams['mathtext.default'] = 'regular'
    mpl.rcParams['axes.unicode_minus'] = True
    from matplotlib.pyplot import cm

    fig, axes = plt.subplots(4, 3, figsize=(8, 10))

    this_axis = axes[0][0]
    voltage_range = np.linspace(0., 1., 10000)
    this_axis.plot(voltage_range, pot_phi(voltage_range), c='c', label='Potentiation')
    this_axis.plot(voltage_range, dep_phi(voltage_range), c='r', label='Depression')
    this_axis.set_xlabel('Normalized voltage')
    this_axis.set_ylabel('Modulation factor')
    this_axis.set_xlim(0., 1.)
    this_axis.set_title('Modulation of synaptic eligibility\nsignals by postsynaptic voltage',
                        fontsize=mpl.rcParams['font.size'])
    this_axis.legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)

    this_axis = axes[3][0]
    ymax = 0.
    for color, label, ramp in zip(['darkgrey', 'k'], ['Before induction 2', 'After induction 2'],
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
    this_axis.legend(loc=(0.05, 0.95), frameon=False, framealpha=0.5, handlelength=1,
                     fontsize=mpl.rcParams['font.size'])
    this_axis.set_xticks(np.arange(0., context.track_length, 45.))

    this_axis = axes[1][0]
    xmax = max(5000., local_signal_filter_t[-1], global_filter_t[-1]) / 1000.
    xmax = math.ceil(xmax)
    this_axis.plot(local_signal_filter_t / 1000., local_signal_filter / np.max(local_signal_filter), color='gray',
                   label='Synaptic\neligibility signal')
    this_axis.plot(global_filter_t / 1000., global_filter / np.max(global_filter), color='k',
                   label='Dendritic\ninstructive signal')
    this_axis.set_xlabel('Time (s)')
    this_axis.set_ylabel('Normalized\namplitude')
    this_axis.set_ylim(0., this_axis.get_ylim()[1])
    this_axis.set_xlim(-0.5, xmax)
    this_axis.set_title('Plasticity signal kinetics', fontsize=mpl.rcParams['font.size'])
    this_axis.legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, fontsize=mpl.rcParams['font.size'])

    this_axis = axes[2][0]
    dep_scale = context.k_dep / context.k_pot
    this_axis.plot(signal_xrange, pot_rate(signal_xrange), label='Potentiation', c='c')
    this_axis.plot(signal_xrange, dep_rate(signal_xrange) * dep_scale, label='Depression', c='r')
    this_axis.set_xlabel('Normalized signal overlap')
    this_axis.set_ylabel('Normalized rate')
    this_axis.set_ylim(0., this_axis.get_ylim()[1])
    # this_axis.set_xlim(0., 1.)
    this_axis.set_title('Nonlinear sensitivity to\nplasticity signal overlap', fontsize=mpl.rcParams['font.size'])
    this_axis.legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, fontsize=mpl.rcParams['font.size'])

    example_pot_signals = dict()
    example_dep_signals = dict()
    example_net_dwdt = dict()
    example_pre_rates = dict()

    current_ramp = ramp_snapshots[induction_lap]
    current_complete_ramp = get_complete_ramp(current_ramp, context.binned_x, context.position,
                                              context.complete_run_vel_gate, context.induction_gate,
                                              this_peak_ramp_amp)

    start_time = context.induction_start_times[induction_lap]
    if induction_lap == len(context.induction_start_times) - 1:
        stop_time = context.down_t[-1]
    else:
        stop_time = context.induction_start_times[induction_lap + 1]
    indexes = np.where((context.down_t >= start_time) & (context.down_t <= stop_time))

    this_current_complete_ramp = np.interp(context.down_t, context.complete_t, current_complete_ramp)
    this_current_ramp = this_current_complete_ramp[indexes]
    this_normalized_current_complete_ramp = np.divide(this_current_complete_ramp, this_peak_ramp_amp)
    this_pot_phi = pot_phi(this_normalized_current_complete_ramp)
    this_dep_phi = dep_phi(this_normalized_current_complete_ramp)

    this_t = context.down_t[indexes] / 1000.
    this_global_signal = global_signal[indexes]

    for name, i in viewitems(example_input_dict):
        this_down_rate_map = context.down_rate_maps[i]
        example_pre_rates[name] = this_down_rate_map[indexes]

        this_pot_signal = np.divide(get_local_signal(np.multiply(this_down_rate_map, this_pot_phi), local_signal_filter,
                                                     context.down_dt)[indexes], local_signal_peak)
        this_dep_signal = np.divide(get_local_signal(np.multiply(this_down_rate_map, this_dep_phi), local_signal_filter,
                                                     context.down_dt)[indexes], local_signal_peak)
        example_pot_signals[name] = this_pot_signal
        example_dep_signals[name] = this_dep_signal
        this_pot_rate = pot_rate(np.multiply(this_pot_signal, this_global_signal))
        this_dep_rate = dep_rate(np.multiply(this_dep_signal, this_global_signal))
        this_net_weight_rate = context.k_pot * this_pot_rate - context.k_dep * this_dep_rate
        example_net_dwdt[name] = this_net_weight_rate

    colors = ['r', 'c']

    axes[0][1].get_shared_x_axes().join(axes[0][1], axes[0][2], axes[1][1], axes[1][2], axes[2][1], axes[2][2])
    axes[0][1].get_shared_y_axes().join(axes[0][1], axes[0][2])
    axes[1][1].get_shared_y_axes().join(axes[1][1], axes[1][2])
    # axes[2][1].get_shared_y_axes().join(axes[2][1], axes[2][2])

    ymax1 = np.max(this_global_signal)
    axes0_1_right = [axes[0][1].twinx(), axes[0][2].twinx()]
    axes0_1_right[0].get_shared_y_axes().join(axes0_1_right[0], axes0_1_right[1])
    for i, (name, index) in enumerate(viewitems(example_input_dict)):
        this_rate_map = example_pre_rates[name]
        this_pot_signal = example_pot_signals[name]
        this_dep_signal = example_dep_signals[name]
        this_net_dwdt = example_net_dwdt[name]
        ymax1 = max(ymax1, np.max(this_pot_signal), np.max(this_dep_signal))
        axes[0][i + 1].plot(this_t, this_rate_map, c=colors[i], linewidth=1., label='Presynaptic firing rate')
        axes0_1_right[i].plot(this_t, this_current_ramp, c='k', linewidth=1., label='Postsynaptic voltage')
        axes[0][i + 1].set_title('%s:' % name, fontsize=mpl.rcParams['font.size'], y=1.4)
        axes[0][i + 1].set_xlabel('Time (s)')
        axes[0][i + 1].legend(loc=(0., 1.15), frameon=False, framealpha=0.5, handlelength=1,
                              fontsize=mpl.rcParams['font.size'])
        axes0_1_right[i].legend(loc=(0., 1.0), frameon=False, framealpha=0.5, handlelength=1,
                                fontsize=mpl.rcParams['font.size'])

        axes[1][i + 1].plot(this_t, this_pot_signal, c=colors[i], label='Synaptic potentiation eligibility signal')
        axes[1][i + 1].plot(this_t, this_dep_signal, '--', c=colors[i], label='Synaptic depression eligibility signal')
        axes[1][i + 1].plot(this_t, this_global_signal, c='k', label='Dendritic instructive signal', linewidth=0.75)
        axes[1][i + 1].fill_between(this_t, 0., np.minimum(this_pot_signal, this_global_signal), alpha=0.5,
                                    facecolor=colors[i], label='Signal overlap')
        axes[1][i + 1].fill_between(this_t, 0., np.minimum(this_dep_signal, this_global_signal), alpha=0.5,
                                    facecolor=colors[i])
        axes[1][i + 1].set_xlabel('Time (s)')
        axes[1][i + 1].legend(loc=(0., 1.0), frameon=False, framealpha=0.5, handlelength=1,
                              fontsize=mpl.rcParams['font.size'])

        axes[2][i + 1].plot(this_t, this_net_dwdt, c=colors[i])
        axes[2][i + 1].set_xlabel('Time (s)')

    xmin = max(-5., np.min(this_t))
    xmax = np.max(this_t)
    axes[0][1].set_xlim(xmin, xmax)
    axes[0][1].set_xticks(np.round((np.arange(xmin, xmax, 5.) / 5.)) * 5.)
    axes[0][2].set_xticks(np.round((np.arange(xmin, xmax, 5.) / 5.)) * 5.)
    axes[1][1].set_xticks(np.round((np.arange(xmin, xmax, 5.) / 5.)) * 5.)
    axes[1][2].set_xticks(np.round((np.arange(xmin, xmax, 5.) / 5.)) * 5.)
    axes[2][1].set_xticks(np.round((np.arange(xmin, xmax, 5.) / 5.)) * 5.)
    axes[2][2].set_xticks(np.round((np.arange(xmin, xmax, 5.) / 5.)) * 5.)

    ymax0_left = context.input_field_peak_rate / 0.9
    ymax0_right = this_peak_ramp_amp / 0.9
    ymax1 /= 0.9
    axes[0][1].set_ylim([0., ymax0_left])
    # axes[0][2].set_ylim([0., ymax1_left])
    axes0_1_right[0].set_ylim([0., ymax0_right])
    # axes0_1_right[1].set_ylim([0., ymax0_right])
    axes[1][1].set_ylim([0., ymax1])
    # axes[1][2].set_ylim([0., ymax1])
    ymax2_1 = np.max(np.abs(axes[2][1].get_ylim())) / 0.85
    axes[2][1].set_ylim([-ymax2_1, ymax2_1])
    ymax2_2 = np.max(np.abs(axes[2][2].get_ylim())) / 0.85
    axes[2][2].set_ylim([-ymax2_2, ymax2_2])

    bar_loc0 = ymax0_left * 0.95
    bar_loc1 = ymax1 * 0.95
    bar_loc2_1 = ymax2_1 * 0.95
    bar_loc2_2 = ymax2_2 * 0.95
    axes[0][1].hlines(bar_loc0,
                      xmin=context.induction_start_times[induction_lap] / 1000.,
                      xmax=context.induction_stop_times[induction_lap] / 1000., linewidth=2)
    axes[0][2].hlines(bar_loc0,
                      xmin=context.induction_start_times[induction_lap] / 1000.,
                      xmax=context.induction_stop_times[induction_lap] / 1000., linewidth=2)
    axes[1][1].hlines(bar_loc1,
                      xmin=context.induction_start_times[induction_lap] / 1000.,
                      xmax=context.induction_stop_times[induction_lap] / 1000., linewidth=2)
    axes[1][2].hlines(bar_loc1,
                      xmin=context.induction_start_times[induction_lap] / 1000.,
                      xmax=context.induction_stop_times[induction_lap] / 1000., linewidth=2)
    axes[2][1].hlines(bar_loc2_1,
                      xmin=context.induction_start_times[induction_lap] / 1000.,
                      xmax=context.induction_stop_times[induction_lap] / 1000., linewidth=2)
    axes[2][2].hlines(bar_loc2_2,
                      xmin=context.induction_start_times[induction_lap] / 1000.,
                      xmax=context.induction_stop_times[induction_lap] / 1000., linewidth=2)

    axes[0][1].set_ylabel('Firing rate (Hz)')
    axes0_1_right[0].set_ylabel('Ramp\namplitude (mV)', rotation=-90, labelpad=20)
    axes0_1_right[0].set_yticks(np.arange(0., np.max(this_current_ramp), 5.))
    axes0_1_right[1].set_yticks(np.arange(0., np.max(this_current_ramp), 5.))
    axes[0][1].set_yticks(np.arange(0., context.input_field_peak_rate + 1., 10.))
    axes[0][2].set_yticks(np.arange(0., context.input_field_peak_rate + 1., 10.))
    axes[1][1].set_ylabel('Plasticity signal\namplitude')
    axes[1][1].set_yticks(np.arange(0., ymax1, 0.2))
    axes[1][2].set_yticks(np.arange(0., ymax1, 0.2))
    # yrange2 = np.round((np.arange(-ymax2, ymax2, 0.2) / 0.2)) * 0.2
    # axes[2][1].set_yticks(yrange2)
    # axes[2][2].set_yticks(yrange2)
    axes[2][1].set_ylabel('Rate of change\nin synaptic weight')

    for row in range(3):
        for col in range(1, 3):
            for label in axes[row][col].get_xticklabels():
                label.set_visible(True)
    clean_twin_right_axes(axes0_1_right)

    ymax3 = 0.
    for col, weights in zip(range(1, 3), [initial_weights, final_weights]):
        this_axis = axes[3][col]
        this_max_rate_map = np.zeros_like(context.input_rate_maps[0])
        for i in (index for index in input_sample_indexes if index not in viewvalues(example_input_dict)):
            rate_map = np.array(context.input_rate_maps[i])
            rate_map *= weights[i] * context.ramp_scaling_factor
            ymax3 = max(ymax3, np.max(rate_map))
            this_axis.plot(context.binned_x, rate_map, c='gray', zorder=0, linewidth=0.75)  # , alpha=0.5)
        for i, (name, index) in enumerate(viewitems(example_input_dict)):
            rate_map = np.array(context.input_rate_maps[index])
            rate_map *= weights[index] * context.ramp_scaling_factor
            ymax3 = max(ymax3, np.max(rate_map))
            this_axis.plot(context.binned_x, rate_map, c=colors[i], zorder=1, label=name)
        this_axis.set_xlim(0., context.track_length)
        this_axis.set_xticks(np.arange(0., context.track_length, 45.))
        this_axis.set_xlabel('Position (cm)')
    axes[3][1].legend(loc=(-0.1, 1.), frameon=False, framealpha=0.5, handlelength=1, fontsize=mpl.rcParams['font.size'])
    axes[3][1].set_ylabel('Input\namplitude (mV)')
    ymax3 = math.ceil(10. * ymax3 / 0.95) / 10.
    bar_loc = ymax3 * 0.95
    for col in range(1, 3):
        this_axis = axes[3][col]
        this_axis.set_ylim(0., ymax3)
        this_axis.hlines(bar_loc, xmin=context.mean_induction_start_loc, xmax=context.mean_induction_stop_loc)

    clean_axes(axes)
    fig.suptitle('Voltage-dependent model D (cell %i)' % cell_id,
                 fontsize=mpl.rcParams['font.size'], x=0.02, ha='left')
    fig.subplots_adjust(left=0.1, hspace=1.075, wspace=0.7, right=0.955, top=0.925, bottom=0.05)
    fig.show()

    # Alternative plots
    fig, axes = plt.subplots(4, 3, figsize=(8, 10))

    axes[0][0].get_shared_x_axes().join(axes[0][0], axes[0][1], axes[0][2], axes[1][1], axes[1][2])
    axes[0][0].get_shared_y_axes().join(axes[0][0], axes[0][1], axes[1][1])

    bar_loc = max(10., np.max(model_ramp) + 1., np.max(target_ramp) + 1.) * 0.95
    delta_weights = np.subtract(final_weights, initial_weights)
    peak_weight = np.max(np.abs(delta_weights))
    axes[0][2].plot(context.peak_locs, delta_weights, c='k')
    axes[0][2].axhline(y=0., linestyle='--', c='grey')
    axes[0][2].hlines(peak_weight * 1.05, xmin=context.mean_induction_start_loc, xmax=context.mean_induction_stop_loc)
    axes[0][1].plot(context.binned_x, initial_ramp, label='Before induction 2', c='darkgrey')
    axes[0][1].plot(context.binned_x, model_ramp, label='After induction 2', c='k')
    axes[0][1].hlines(bar_loc, xmin=context.mean_induction_start_loc, xmax=context.mean_induction_stop_loc)
    axes[0][2].set_ylabel('Change in\nsynaptic weight')
    axes[0][2].set_xlabel('Location (cm)')
    axes[0][1].set_ylabel('Ramp\namplitude (mV)')
    axes[0][1].set_xlabel('Location (cm)')
    axes[0][1].set_xticks(np.arange(0., context.track_length, 45.))
    axes[0][2].set_xticks(np.arange(0., context.track_length, 45.))
    # axes[0][1].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
    axes[0][1].set_ylim([min(-1., np.min(model_ramp) - 1., np.min(target_ramp) - 1.),
                         max(10., np.max(model_ramp) + 1., np.max(target_ramp) + 1.)])
    axes[0][2].set_ylim([-peak_weight, peak_weight * 1.1])
    axes[0][1].set_title('Model fit', fontsize=mpl.rcParams['font.size'], pad=10.)

    axes[0][0].plot(context.binned_x, initial_exp_ramp, label='Before induction 2', c='darkgrey')
    axes[0][0].plot(context.binned_x, target_ramp, label='After induction 2', c='k')
    axes[0][0].hlines(bar_loc, xmin=context.mean_induction_start_loc, xmax=context.mean_induction_stop_loc)
    axes[0][0].set_ylabel('Ramp\namplitude (mV)')
    axes[0][0].set_xlabel('Location (cm)')
    axes[0][0].set_xticks(np.arange(0., context.track_length, 45.))
    axes[0][0].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
    axes[0][0].set_ylim([min(-1., np.min(model_ramp) - 1., np.min(target_ramp) - 1.),
                         max(10., np.max(model_ramp) + 1., np.max(target_ramp) + 1.)])
    axes[0][0].set_title('Experimental data', fontsize=mpl.rcParams['font.size'], pad=10.)

    cmap = cm.jet
    axes[1][1].plot(context.binned_x, ramp_snapshots[0], c='k')
    axes[1][1].set_ylabel('Ramp\namplitude (mV)')
    axes[1][1].set_xlabel('Location (cm)')
    axes[1][2].set_ylabel('Change in\nsynaptic weight')
    axes[1][2].set_xlabel('Location (cm)')

    norm = mpl.colors.Normalize(vmin=0, vmax=len(ramp_snapshots), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    for i in range(1, len(ramp_snapshots)):
        current_ramp = ramp_snapshots[i]
        current_delta_weights = np.subtract(delta_weights_snapshots[i], delta_weights_snapshots[i - 1])
        axes[1][1].plot(context.binned_x, current_ramp, c=mapper.to_rgba(i))
        axes[1][2].plot(context.peak_locs, current_delta_weights, c=mapper.to_rgba(i))
    mapper.set_array([])
    cbar = fig.colorbar(mapper, ax=axes[1][2])
    cbar.set_label('Induction lap #', rotation=270., labelpad=12.)
    cbar.set_ticks(np.arange(1., len(ramp_snapshots), 2))

    clean_axes(axes)
    fig.suptitle('Voltage-dependent model D (cell %i)' % cell_id,
                 fontsize=mpl.rcParams['font.size'], x=0.02, ha='left')
    fig.subplots_adjust(left=0.1, hspace=1.075, wspace=0.7, right=0.955, top=0.925, bottom=0.05)
    fig.show()

    context.update(locals())


def plot_model_summary_figure(cell_id, export_file_path=None, exported_data_key=None, induction_lap=0,
                              target_min_delay=2750., target_delay_range=250.):
    """

    :param cell_id: int
    :param export_file_path: str (path)
    :param exported_data_key: str
    :param induction_lap: int
    :param target_min_delay
    :param target_delay_range
    """
    if (cell_id, 2) not in context.data_keys:
        raise KeyError('plot_model_summary_figure: cell_id: %i, induction: 2 not found' % cell_id)
    if export_file_path is None:
        raise IOError('plot_model_summary_figure: no export_file_path provided')
    elif not os.path.isfile(export_file_path):
        raise IOError('plot_model_summary_figure: invalid export_file_path: %s' % export_file_path)
    with h5py.File(export_file_path, 'r') as f:
        source = get_h5py_group(f, [exported_data_key, 'exported_data', str(cell_id), '2', 'model_ramp_features'])
        x = source['param_array'][:]
        if 'local_signal_peak' not in source.attrs or 'global_signal_peak' not in source.attrs:
            raise KeyError('plot_model_summary_figure: missing required attributes for cell_id: %i, '
                           'induction 2; from file: %s' % (cell_id, export_file_path))
        local_signal_peak = source.attrs['local_signal_peak']
        global_signal_peak = source.attrs['global_signal_peak']
        local_signal_filter_t = source['local_signal_filter_t'][:]
        local_signal_filter = source['local_signal_filter'][:]
        global_filter_t = source['global_filter_t'][:]
        global_filter = source['global_filter'][:]
        initial_weights = source['initial_weights'][:]
        initial_ramp = source['initial_model_ramp'][:]
        model_ramp = source['model_ramp'][:]
        ramp_snapshots = []
        for lap in range(len(source['ramp_snapshots'])):
            ramp_snapshots.append(source['ramp_snapshots'][str(lap)][:])
        delta_weights_snapshots = []
        for lap in range(len(source['delta_weights_snapshots'])):
            delta_weights_snapshots.append(source['delta_weights_snapshots'][str(lap)][:])
        final_weights = source['model_weights'][:]

    import_data(cell_id, 2)
    update_source_contexts(x, context)

    initial_exp_ramp = context.exp_ramp['before']
    target_ramp = context.exp_ramp['after']

    global_signal = np.divide(get_global_signal(context.down_induction_gate, global_filter), global_signal_peak)

    signal_xrange = np.linspace(0., 1., 10000)
    pot_phi = np.vectorize(lambda x: min(1., max(0., (1. - x))))
    dep_phi = np.vectorize(lambda x: min(1., max(0., x)))

    pot_rate = np.vectorize(scaled_single_sigmoid(
        context.f_pot_th, context.f_pot_th + context.f_pot_half_width, signal_xrange))
    dep_rate = np.vectorize(scaled_single_sigmoid(
        context.f_dep_th, context.f_dep_th + context.f_dep_half_width, signal_xrange))

    this_peak_ramp_amp = context.peak_ramp_amp + context.plateau_delta_depo

    input_sample_indexes = np.arange(len(context.peak_locs))

    example_input_dict = {}

    sample_time_delays = []
    mean_induction_start_time_index = np.where(context.mean_position > context.mean_induction_start_loc)[0][0]
    mean_induction_start_time = context.mean_t[mean_induction_start_time_index]
    for index in input_sample_indexes:
        this_peak_loc = context.peak_locs[index]
        this_time_index = np.where(context.mean_position > this_peak_loc)[0][0]
        this_delay = context.mean_t[this_time_index] - mean_induction_start_time
        sample_time_delays.append(this_delay)
    sample_time_delays = np.array(sample_time_delays)

    relative_indexes = np.where((sample_time_delays > target_min_delay) &
                                (sample_time_delays <= target_min_delay + target_delay_range) &
                                (final_weights[input_sample_indexes] < initial_weights[input_sample_indexes]))[0]
    distant_depressing_indexes = input_sample_indexes[relative_indexes]
    if np.any(distant_depressing_indexes):
        relative_index = np.argmin(np.subtract(final_weights, initial_weights)[distant_depressing_indexes])
        depressing_example_index = distant_depressing_indexes[relative_index]
    else:
        relative_index = np.argmin(np.subtract(final_weights, initial_weights)[input_sample_indexes])
        depressing_example_index = input_sample_indexes[relative_index]

    import matplotlib as mpl
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['font.size'] = 12.
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['axes.titlepad'] = 2.
    mpl.rcParams['mathtext.default'] = 'regular'
    mpl.rcParams['axes.unicode_minus'] = True
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(3, figsize=(5, 7.5))

    current_weights = np.add(delta_weights_snapshots[induction_lap], 1.)
    current_ramp = ramp_snapshots[induction_lap]
    current_complete_ramp = get_complete_ramp(current_ramp, context.binned_x, context.position,
                                              context.complete_run_vel_gate, context.induction_gate,
                                              this_peak_ramp_amp)

    start_time = context.induction_start_times[induction_lap]
    if induction_lap == len(context.induction_start_times) - 1:
        stop_time = context.down_t[-1]
    else:
        stop_time = context.induction_start_times[induction_lap + 1]
    indexes = np.where((context.down_t >= start_time) & (context.down_t <= stop_time))

    this_current_ramp = np.interp(context.down_t, context.complete_t, current_complete_ramp)[indexes]
    this_normalized_current_ramp = np.divide(this_current_ramp, this_peak_ramp_amp)
    this_t = context.down_t[indexes] / 1000.
    this_global_signal = global_signal[indexes]

    example_pre_rate = context.down_rate_maps[depressing_example_index][indexes]
    example_pot_elig_signal = \
        np.divide(get_local_signal(np.multiply(example_pre_rate, pot_phi(this_normalized_current_ramp)),
                                   local_signal_filter, context.down_dt),
                  local_signal_peak)
    example_dep_elig_signal = \
        np.divide(get_local_signal(np.multiply(example_pre_rate, dep_phi(this_normalized_current_ramp)),
                                   local_signal_filter, context.down_dt),
                  local_signal_peak)
    example_pot_rate = context.k_pot * pot_rate(np.multiply(example_pot_elig_signal, this_global_signal))
    example_dep_rate = context.k_dep * dep_rate(np.multiply(example_dep_elig_signal, this_global_signal))
    example_net_dwdt = np.subtract(example_pot_rate, example_dep_rate)

    axes[0].get_shared_x_axes().join(axes[0], axes[1], axes[2])
    ymax1 = max(np.max(this_global_signal), np.max(example_pot_elig_signal), np.max(example_dep_elig_signal))
    axes0_right = axes[0].twinx()

    axes[0].plot(this_t, example_pre_rate, c='r', linewidth=1., label='Presynaptic\nfiring rate')
    axes0_right.plot(this_t, this_current_ramp, c='grey', linewidth=1., label='Postsynaptic $V_{m}$')
    handles, labels = axes[0].get_legend_handles_labels()
    handles_right, labels_right = axes0_right.get_legend_handles_labels()
    handles.extend(handles_right)
    labels.extend(labels_right)
    handles.append(Line2D([0], [0], color='k'))
    labels.append('Postsynaptic\nplateau')
    leg = axes[0].legend(handles=handles, labels=labels, loc=(0., 1.), frameon=False, framealpha=0.5, handlelength=1,
                         fontsize=mpl.rcParams['font.size'])
    for line in leg.get_lines():
        line.set_linewidth(2.)

    axes[1].plot(this_t, example_pot_elig_signal, c='r', linewidth=1., label='Potentiation $signal_{eligibility}$')
    axes[1].plot(this_t, example_dep_elig_signal, c='c', linewidth=1.,
                 label='Depression $signal_{eligibility}$')
    axes[1].plot(this_t, this_global_signal, c='k', linewidth=1., label='Dendritic $signal_{instructive}$')
    axes[1].fill_between(this_t, 0., np.minimum(example_pot_elig_signal, this_global_signal), alpha=0.5,
                         facecolor='r', edgecolor='none', label='Potentiation signal overlap')
    axes[1].fill_between(this_t, 0., np.minimum(example_dep_elig_signal, this_global_signal),
                         alpha=0.5, facecolor='c', edgecolor='none', label='Depression signal overlap')
    leg = axes[1].legend(loc=(0., 1.0), frameon=False, framealpha=0.5, handlelength=1,
                         fontsize=mpl.rcParams['font.size'])
    for line in leg.get_lines():
        line.set_linewidth(2.)

    axes[2].plot(this_t, example_pot_rate, c='c', linewidth=1., label='Potentiation rate')
    axes[2].plot(this_t, example_dep_rate, c='r', linewidth=1., label='Depression rate')
    axes[2].plot(this_t, example_net_dwdt, c='k', linewidth=1., label='Net dW/dt')
    axes[2].set_xlabel('Time (s)')
    leg = axes[2].legend(loc=(0., 1.0), frameon=False, framealpha=0.5, handlelength=1,
                         fontsize=mpl.rcParams['font.size'])
    for line in leg.get_lines():
        line.set_linewidth(2.)

    xmin = max(-5., np.min(this_t))
    xmax = np.max(this_t)
    axes[0].set_xlim(xmin, xmax)
    axes[0].set_xticks(np.round((np.arange(xmin, xmax, 5.) / 5.)) * 5.)
    axes[1].set_xticks(np.round((np.arange(xmin, xmax, 5.) / 5.)) * 5.)
    axes[2].set_xticks(np.round((np.arange(xmin, xmax, 5.) / 5.)) * 5.)

    ymax0_left = context.input_field_peak_rate / 0.85
    ymax0_right = this_peak_ramp_amp / 0.85

    axes[0].set_ylim([0., ymax0_left])
    axes0_right.set_ylim([0., ymax0_right])
    axes[1].set_ylim([0., axes[1].get_ylim()[1]])

    bar_loc0 = ymax0_left * 0.95
    axes[0].hlines(bar_loc0, xmin=context.induction_start_times[induction_lap] / 1000.,
                   xmax=context.induction_stop_times[induction_lap] / 1000., linewidth=2.)

    axes[0].set_ylabel('Firing rate (Hz)')
    axes0_right.set_ylabel('Ramp\namplitude (mV)', rotation=-90, labelpad=30)
    axes0_right.set_yticks(np.arange(0., np.max(this_current_ramp), 5.))
    axes[0].set_yticks(np.arange(0., context.input_field_peak_rate + 1., 10.))
    axes[1].set_ylabel('Plasticity signal\namplitude')
    axes[2].set_ylabel('Rate of change\nin synaptic weight')

    clean_twin_right_axes([axes0_right])
    clean_axes(axes)
    fig.suptitle('Voltage-dependent model D (cell %i)' % cell_id,
                 fontsize=mpl.rcParams['font.size'], x=0.02, ha='left')
    fig.subplots_adjust(left=0.25, hspace=0.8, right=0.8, top=0.8, bottom=0.1)
    fig.show()

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
                                model_id=None, export=False, plot=False):
    """

    :param x: array
    :param cell_id: int
    :param induction: int
    :param local_signal_peak: float
    :param global_signal_peak: float
    :param model_id: int or str
    :param export: bool
    :param plot: bool
    :return: dict
    """
    import_data(cell_id, induction)
    update_source_contexts(x, context)
    start_time = time.time()
    if context.disp:
        print('Process: %i: computing model_ramp_features for model_id: %s, cell_id: %i, induction: %i with x: %s' %
              (os.getpid(), model_id, context.cell_id, context.induction, ', '.join('%.3E' % i for i in x)))
        sys.stdout.flush()
    result = calculate_model_ramp(local_signal_peak=local_signal_peak, global_signal_peak=global_signal_peak,
                                  model_id=model_id, export=export, plot=plot)
    if context.disp:
        print('Process: %i: computing model_ramp_features for model_id: %s, cell_id: %i, induction: %i took %.1f s' %
              (os.getpid(), model_id, context.cell_id, context.induction, time.time() - start_time))
        sys.stdout.flush()
    return result


def filter_features_model_ramp(primitives, current_features, model_id=None, export=False):
    """

    :param primitives: list of dict (each dict contains results from a single simulation)
    :param current_features: dict
    :param model_id: int or str
    :param export: bool
    :return: dict
    """
    features = {}
    groups = ['spont', 'exp1', 'exp2', 'exp3']
    grouped_feature_names = ['delta_val_at_target_peak', 'delta_val_at_model_peak', 'delta_width', 'delta_peak_shift',
                             'delta_asymmetry', 'delta_min_loc', 'delta_val_at_target_min', 'delta_val_at_model_min',
                             'residual_score']
    feature_names = ['weights_path_distance']
    for this_result_dict in primitives:
        if not this_result_dict:
            if context.verbose > 0:
                print('optimize_biBTSP_%s: filter_features_model_ramp: pid: %i; model_id: %s; model failed' %
                      (BTSP_model_name, os.getpid(), model_id))
                sys.stdout.flush()
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


def get_objectives(features, model_id=None, export=False):
    """

    :param features: dict
    :param model_id: int or str
    :param export: bool
    :return: tuple of dict
    """
    objectives = {}

    grouped_feature_names = ['residual_score']
    groups = ['spont', 'exp1', 'exp2', 'exp3']
    for feature_name in grouped_feature_names:
        for group in groups:
            objective_name = group + '_' + feature_name
            if objective_name in features:
                objectives[objective_name] = features[objective_name]

    feature_name = 'weights_path_distance'
    if feature_name in context.objective_names and feature_name in features:
        objectives[feature_name] = (features[feature_name] / context.target_range[feature_name]) ** 2.

    for objective_name in context.objective_names:
        if objective_name not in objectives:
            objectives[objective_name] = 0.
        else:
            objectives[objective_name] = np.mean(objectives[objective_name])

    return features, objectives


def run_tests():

    model_id = 0
    if 'model_key' in context() and context.model_key is not None:
        model_label = context.model_key
    else:
        model_label = 'test'

    features = {}
    args = context.interface.execute(get_args_static_signal_amplitudes)
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[model_id] * group_size] + [[context.export] * group_size]
    primitives = context.interface.map(compute_features_signal_amplitudes, *sequences)
    new_features = context.interface.execute(filter_features_signal_amplitudes, primitives, features, model_id,
                                             context.export, context.plot)
    features.update(new_features)

    args = context.interface.execute(get_args_dynamic_model_ramp, context.x0_array, features)
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[model_id] * group_size] + \
                [[context.export] * group_size] + [[context.plot] * group_size]
    primitives = context.interface.map(compute_features_model_ramp, *sequences)
    new_features = context.interface.execute(filter_features_model_ramp, primitives, features, model_id, context.export)
    features.update(new_features)

    features, objectives = context.interface.execute(get_objectives, features, model_id, context.export)
    if context.export:
        merge_exported_data(context, param_arrays=[context.x0_array],
                            model_ids=[model_id], model_labels=[model_label], features=[features],
                            objectives=[objectives], export_file_path=context.export_file_path,
                            verbose=context.verbose > 1)
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

    if context.plot:
        context.interface.apply(plt.show)

    return features


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_biBTSP_VD_D_cli_config.yaml')
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
    To execute on a single process on cell from the experimental dataset with cell_id==1:
    python -i optimize_biBTSP_VD_D.py --cell_id=1 --plot --framework=serial --interactive

    To analyze and export data to using MPI parallelism with 1 controller process and N - 1 worker processes:
    mpirun -n N python -i -m mpi4py.futures -m nested.analyze --config-file-path=$PATH_TO_CONFIG_YAML \
        --param-file-path=$PATH_TO_PARAM_YAML --model-key=$VALID_KEY_IN_PARAM_YAML --disp --cell_id=1
        --framework=mpi --export

    To plot results previously exported to a file on a single process:
    python -i optimize_biBTSP_VD_D.py --param_file_path=$PATH_TO_PARAM_YAML --model_key=$VALID_KEY_IN_PARAM_YAML \
        --cell_id=1 --framework=serial --export-file-path-$PATH_TO_EXPORTED_DATA_HDF5 --plot-summary-figure \
        --model-label=$VALID_KEY_IN_EXPORT_FILE

    To optimize the models by running many instances in parallel:
    mpirun -n N python -i -m mpi4py.futures -m nested.optimize --config-file-path=$PATH_TO_CONFIG_YAML --disp
        --framework=mpi --cell_id=1 --pop_size=200 --path_length=3 --max_iter=50 --label=cell1

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
    if label is None:
        if 'input_field_width' in kwargs:
            label = '%scm' % str(int(float(kwargs['input_field_width'])))
        if 'cell_id' in kwargs:
            if label is None:
                label = 'cell%i' % int(kwargs['cell_id'])
            else:
                label += '_cell%i' % int(kwargs['cell_id'])

    context.interface = get_parallel_interface(source_file=__file__, source_package=__package__, **kwargs)
    context.interface.start(disp=context.disp)
    context.interface.ensure_controller()
    config_optimize_interactive(__file__, config_file_path=config_file_path, output_dir=output_dir,
                                export=export, export_file_path=export_file_path, label=label,
                                disp=context.disp, interface=context.interface, verbose=verbose, plot=plot, **kwargs)

    if plot_summary_figure:
        if 'cell_id' not in kwargs:
            raise RuntimeError('optimize_biBTSP_%s: missing required parameter: cell_id' % BTSP_model_name)
        context.interface.execute(plot_model_summary_figure, int(context.kwargs['cell_id']), export_file_path,
                                  exported_data_key)
        context.interface.execute(plot_model_summary_supp_figure, int(context.kwargs['cell_id']), export_file_path,
                                  exported_data_key)
    elif not debug:
        run_tests()

    if not context.interactive:
        context.interface.stop()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
