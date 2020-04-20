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

Features/assumptions of weight-dependent model B:
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

biBTSP_WD_D: Single eligibility signal filter. Sigmoidal f_pot and f_dep.
"""
__author__ = 'milsteina'
from biBTSP_utils import *
from nested.parallel import *
from nested.optimize_utils import *
import click

context = Context()


BTSP_model_name = 'WD_D'


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
        allow_offset_cell_ids = [int(cell_id) for cell_id in f['data'] if '2' in f['data'][cell_id] and
                                 ('1' not in f['data'][cell_id] or
                                  'before' not in f['data'][cell_id]['1']['raw']['exp_ramp'])]
    if context.verbose > 1:
        print('optimize_biBTSP_%s: pid: %i; processing the following data_keys: %s' %
              (BTSP_model_name, os.getpid(), str(context.data_keys)))
    self_consistent_cell_ids = [cell_id for (cell_id, induction) in context.data_keys if induction == 1 and
                                (cell_id, 2) in context.data_keys]
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

            # TODO: exp_ramp_vs_t reflects mean, rather than min delay to plateau across laps
            induction_context.exp_ramp_vs_t = {'after': data_group['processed']['exp_ramp_vs_t']['after'][:]}
            if 'before' in data_group['processed']['exp_ramp']:
                induction_context.exp_ramp['before'] = data_group['processed']['exp_ramp']['before'][:]
                induction_context.exp_ramp_vs_t['before'] = data_group['processed']['exp_ramp_vs_t']['before'][:]
            induction_context.LSA_ramp = {}
            induction_context.LSA_ramp_offset = {}
            calibrated_input_group = f['calibrated_input'][context.input_field_width_key][cell_key][induction_key]
            if 'before' in calibrated_input_group['LSA_ramp']:
                induction_context.LSA_ramp['before'] = calibrated_input_group['LSA_ramp']['before'][:]
                induction_context.LSA_ramp_offset['before'] = calibrated_input_group['LSA_ramp']['before'].attrs[
                    'ramp_offset']
            induction_context.LSA_weights = {}
            induction_context.LSA_weights['before'] = calibrated_input_group['LSA_weights']['before'][:]
            induction_context.complete_run_vel = data_group['complete']['run_vel'][:]
            induction_context.complete_run_vel_gate = data_group['complete']['run_vel_gate'][:]
            induction_context.complete_position = data_group['complete']['position'][:]
            induction_context.complete_t = data_group['complete']['t'][:]
            induction_context.induction_gate = data_group['complete']['induction_gate'][:]
            # if both induction 1 and 2 are in dataset, for self-consistency, use same target_ramp
            if induction == 1 and '2' in f['data'][cell_key]:
                data_group = f['data'][cell_key]['2']
                calibrated_input_group = f['calibrated_input'][context.input_field_width_key][cell_key]['2']
                induction_context.exp_ramp_raw['after'] = data_group['raw']['exp_ramp']['before'][:]
                induction_context.exp_ramp['after'] = data_group['processed']['exp_ramp']['before'][:]
                induction_context.LSA_ramp['after'] = calibrated_input_group['LSA_ramp']['before'][:]
                induction_context.LSA_ramp_offset['after'] = calibrated_input_group['LSA_ramp']['before'].attrs[
                    'ramp_offset']
                induction_context.LSA_weights['after'] = calibrated_input_group['LSA_weights']['before'][:]
            else:
                induction_context.exp_ramp_raw['after'] = data_group['raw']['exp_ramp']['after'][:]
                induction_context.exp_ramp['after'] = data_group['processed']['exp_ramp']['after'][:]
                induction_context.LSA_ramp['after'] = calibrated_input_group['LSA_ramp']['after'][:]
                induction_context.LSA_ramp_offset['after'] = calibrated_input_group['LSA_ramp']['after'].attrs[
                    'ramp_offset']
                induction_context.LSA_weights['after'] = calibrated_input_group['LSA_weights']['after'][:]
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
    context.down_t = np.arange(context.complete_t[0],
                                         context.complete_t[-1] + context.down_dt / 2., context.down_dt)
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
        calculate_ramp_features(ramp=context.exp_ramp['after'], induction_loc=context.mean_induction_start_loc,
                                binned_x=context.binned_x, interp_x=context.default_interp_x,
                                track_length=context.track_length)
    start_index, peak_index, end_index, min_index = \
        get_indexes_from_ramp_bounds_with_wrap(context.binned_x, start_loc, peak_loc, end_loc, min_loc)
    axes[1][0].scatter(context.binned_x[[start_index, peak_index, end_index]],
                       context.exp_ramp['after'][[start_index, peak_index, end_index]])
    start_index, peak_index, end_index, min_index = \
        get_indexes_from_ramp_bounds_with_wrap(context.mean_position, start_loc, peak_loc, end_loc, min_loc)
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
    fig.show()


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
    for i in range(1, len(pretty_position)):
        if pretty_position[i] - pretty_position[i - 1] < -context.track_length / 2.:
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
    fig.show()


def get_args_static_signal_amplitudes():
    """
    A nested map operation is required to compute model signal amplitudes. The arguments to be mapped are the same
    (static) for each set of parameters.
    :return: list of list
    """
    return list(zip(*context.all_data_keys))


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
        get_dual_exp_decay_signal_filters(context.local_signal_decay, context.global_signal_decay,
                                          context.down_dt, plot)
    global_signal = get_global_signal(context.down_induction_gate, global_filter)
    local_signals = get_local_signal_population(local_signal_filter, context.down_rate_maps, context.down_dt)

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
    if context.verbose > 1:
        index = np.argmax(global_signal_peaks)
        this_data_key = data_key_list[index]
        if this_data_key[0] not in [24] or this_data_key[1] != 1:
            print('cell: %i, induction: %i has largest global_signal_peak' % (this_data_key[0], this_data_key[1]))
        index = np.argmax(local_signal_peaks)
        this_data_key = data_key_list[index]
    if this_data_key[0] not in [6, 18, 23, 24, 25] or this_data_key[1] != 1:
        print('cell: %i, induction: %i has largest local_signal_peak' % (this_data_key[0], this_data_key[1]))
    if plot:
        fig, axes = plt.subplots(1)
        hist, edges = np.histogram(local_signal_peaks, bins=min(10, len(primitives)), density=True)
        bin_width = edges[1] - edges[0]
        axes.plot(edges[:-1] + bin_width / 2., hist * bin_width, c='r', label='Local eligibility signals')
        axes.set_xlabel('Peak local plasticity signal amplitudes (a.u.)')
        axes.set_ylabel('Probability')
        axes.set_title('Local signal amplitude distribution')
        axes.legend(loc='best', frameon=False, framealpha=0.5)
        clean_axes(axes)
        fig.tight_layout()
        fig.show()

        hist, edges = np.histogram(global_signal_peaks, bins=min(10, len(primitives)), density=True)
        bin_width = edges[1] - edges[0]
        fig, axes = plt.subplots(1)
        axes.plot(edges[:-1] + bin_width / 2., hist * bin_width, c='k')
        axes.set_xlabel('Peak global plasticity signal amplitudes (a.u.)')
        axes.set_ylabel('Probability')
        axes.set_title('Global signal amplitude distribution')
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
                                          context.down_dt, plot)
    global_signal = np.divide(get_global_signal(context.down_induction_gate, global_filter), global_signal_peak)
    local_signals = \
        np.divide(get_local_signal_population(local_signal_filter, context.down_rate_maps, context.down_dt),
                  local_signal_peak)

    signal_xrange = np.linspace(0., 1., 10000)
    pot_rate = np.vectorize(scaled_single_sigmoid(
        context.f_pot_th, context.f_pot_th + context.f_pot_peak, signal_xrange))
    dep_rate = np.vectorize(scaled_single_sigmoid(
        context.f_dep_th, context.f_dep_th + context.f_dep_peak, signal_xrange))

    if plot:
        fig, axes = plt.subplots(1)
        dep_scale = context.k_dep / context.k_pot
        axes.plot(signal_xrange, pot_rate(signal_xrange), c='c', label='Potentiation rate')
        axes.plot(signal_xrange, dep_rate(signal_xrange) * dep_scale, c='r', label='Depression rate')
        axes.set_xlabel('Normalized signal overlap (a.u.)')
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
                    get_delta_weights_LSA(context.exp_ramp['before'], ramp_x=context.binned_x, input_x=context.binned_x,
                                          interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                          peak_locs=context.peak_locs, ramp_scaling_factor=context.ramp_scaling_factor,
                                          induction_start_loc=context.mean_induction_start_loc,
                                          induction_stop_loc=context.mean_induction_stop_loc,
                                          track_length=context.track_length, target_range=context.target_range,
                                          bounds=(context.min_delta_weight, context.peak_delta_weight),
                                          initial_delta_weights=initial_delta_weights, verbose=context.verbose)
                if context.verbose > 1:
                    print('Process: %i; re-computed initial weights: model_id: %s, cell_id: %i, before induction: %i,'
                          ' ramp_offset: %.3f' % (os.getpid(), model_id, context.cell_id, context.induction,
                                                  initial_ramp_offset))
            else:
                initial_ramp = context.LSA_ramp['before']
            initial_ramp_offset = None
        else:
            initial_ramp, discard_ramp_offset = \
                get_model_ramp(initial_delta_weights, ramp_x=context.binned_x, input_x=context.binned_x,
                               input_rate_maps=context.input_rate_maps, ramp_scaling_factor=context.ramp_scaling_factor)
            initial_ramp_offset = context.LSA_ramp_offset['after']
    else:
        if context.cell_id in context.allow_offset_cell_ids:
            allow_offset = True
        if not np.all((context.min_delta_weight <= initial_delta_weights) &
                      (initial_delta_weights <= context.peak_delta_weight)):
            initial_ramp, initial_delta_weights, initial_ramp_offset, discard_residual_score = \
                get_delta_weights_LSA(context.exp_ramp['before'], ramp_x=context.binned_x, input_x=context.binned_x,
                                      interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                      peak_locs=context.peak_locs, ramp_scaling_factor=context.ramp_scaling_factor,
                                      induction_start_loc=context.mean_induction_start_loc,
                                      induction_stop_loc=context.mean_induction_stop_loc,
                                      track_length=context.track_length, target_range=context.target_range,
                                      bounds=(context.min_delta_weight, context.peak_delta_weight),
                                      initial_delta_weights=initial_delta_weights, allow_offset=allow_offset,
                                      verbose=context.verbose)
            if context.verbose > 1:
                print('Process: %i; re-computed initial weights: model_id: %s, cell_id: %i, before induction: %i, '
                      'ramp_offset: %.3f' % (os.getpid(), model_id, context.cell_id, context.induction,
                                             initial_ramp_offset))
        else:
            initial_ramp = context.LSA_ramp['before']
            initial_ramp_offset = context.LSA_ramp_offset['before']
        allow_offset = False

    delta_weights_snapshots = [initial_delta_weights]
    current_ramp = initial_ramp
    ramp_snapshots = [current_ramp]
    initial_normalized_weights = np.divide(np.add(initial_delta_weights, 1.), peak_weight)
    current_normalized_weights = np.array(initial_normalized_weights)

    target_ramp = context.exp_ramp['after']

    if plot:
        fig, axes = plt.subplots()
        fig.suptitle('Induction: %i' % context.induction)
        axes.plot(context.down_t / 1000., global_signal)
        axes.set_ylabel('Instructive signal')
        axes.set_xlabel('Time (s)')

        fig2, axes2 = plt.subplots(1, 2, sharex=True)
        fig2.suptitle('Induction: %i' % context.induction)
        axes2[0].plot(context.binned_x, initial_ramp, c='k', label='Before')
        axes2[0].set_ylabel('Ramp amplitude (mV)')
        axes2[0].set_xlabel('Location (cm)')
        axes2[1].set_ylabel('Change in synaptic weight')
        axes2[1].set_xlabel('Location (cm)')

    for induction_lap in range(len(context.induction_start_times)):
        if induction_lap == 0:
            start_time = context.down_t[0]
        else:
            start_time = context.induction_stop_times[induction_lap - 1]
        if induction_lap == len(context.induction_start_times) - 1:
            stop_time = context.down_t[-1]
        else:
            stop_time = context.induction_start_times[induction_lap + 1]
        indexes = np.where((context.down_t >= start_time) & (context.down_t <= stop_time))

        next_normalized_weights = []
        for i, this_local_signal in enumerate(local_signals):
            this_pot_rate = np.trapz(pot_rate(np.multiply(this_local_signal[indexes], global_signal[indexes])),
                                     dx=context.down_dt / 1000.)
            this_dep_rate = np.trapz(dep_rate(np.multiply(this_local_signal[indexes], global_signal[indexes])),
                                     dx=context.down_dt / 1000.)
            this_normalized_delta_weight = context.k_pot * this_pot_rate * (1. - current_normalized_weights[i]) - \
                                           context.k_dep * this_dep_rate * current_normalized_weights[i]
            this_next_normalized_weight = max(0., min(1., current_normalized_weights[i] + this_normalized_delta_weight))
            next_normalized_weights.append(this_next_normalized_weight)
        if plot:
            axes2[1].plot(context.peak_locs,
                          np.multiply(np.subtract(next_normalized_weights, current_normalized_weights), peak_weight),
                          label='Induction lap: %i' % (induction_lap + 1))
        current_normalized_weights = np.array(next_normalized_weights)
        current_delta_weights = np.subtract(np.multiply(current_normalized_weights, peak_weight), 1.)
        delta_weights_snapshots.append(current_delta_weights)
        current_ramp, discard_ramp_offset = \
            get_model_ramp(current_delta_weights, ramp_x=context.binned_x, input_x=context.binned_x,
                           input_rate_maps=context.input_rate_maps, ramp_scaling_factor=context.ramp_scaling_factor,
                           allow_offset=allow_offset, impose_offset=initial_ramp_offset)

        if plot:
            axes2[0].plot(context.binned_x, current_ramp)
        ramp_snapshots.append(current_ramp)

    if plot:
        axes2[1].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
        clean_axes(axes)
        clean_axes(axes2)
        fig.tight_layout()
        fig2.tight_layout()
        fig.show()
        fig2.show()

    delta_weights = np.subtract(current_delta_weights, initial_delta_weights)
    initial_weights = np.multiply(initial_normalized_weights, peak_weight)
    final_weights = np.add(current_delta_weights, 1.)

    if context.induction == 1:
        initial_ramp_offset = None
        if 'before' in context.exp_ramp:
            allow_offset = False
        else:
            allow_offset = True
    try:
        model_ramp, discard_delta_weights, model_ramp_offset, model_residual_score = \
            get_residual_score(current_delta_weights, target_ramp, ramp_x=context.binned_x, input_x=context.binned_x,
                               interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                               ramp_scaling_factor=context.ramp_scaling_factor,
                               induction_loc=context.mean_induction_start_loc, track_length=context.track_length,
                               target_range=context.target_range, allow_offset=allow_offset,
                               impose_offset=initial_ramp_offset, full_output=True)
    except Exception as e:
        print('optimize_biBTSP_%s: compute_features_model_ramp: pid: %i; model_id: %s; Exception was generated while '
              'evaluating cell_id: %i, induction: %i with x:' %
              (BTSP_model_name, os.getpid(), model_id, context.cell_id, context.induction))
        pprint.pprint(context.x_array)
        pprint.pprint('current_ramp has np.nan: %s' % np.any(np.isnan(current_ramp)))
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise e

    if allow_offset and context.induction == 1:
        initial_ramp, discard_ramp_offset = subtract_baseline(initial_ramp, model_ramp_offset)

    result = {'residual_score': model_residual_score}

    if context.cell_id in context.allow_offset_cell_ids and context.induction == 1:
        LSA_delta_weights = context.LSA_weights['after']
        if not np.all((context.min_delta_weight <= LSA_delta_weights) &
                      (LSA_delta_weights <= context.peak_delta_weight)):
            LSA_ramp, LSA_delta_weights, LSA_ramp_offset, LSA_residual_score = \
                get_delta_weights_LSA(context.exp_ramp['after'], ramp_x=context.binned_x, input_x=context.binned_x,
                                      interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                      peak_locs=context.peak_locs, ramp_scaling_factor=context.ramp_scaling_factor,
                                      induction_start_loc=context.mean_induction_start_loc,
                                      induction_stop_loc=context.mean_induction_stop_loc,
                                      track_length=context.track_length, target_range=context.target_range,
                                      bounds=(context.min_delta_weight, context.peak_delta_weight),
                                      initial_delta_weights=LSA_delta_weights, allow_offset=allow_offset,
                                      impose_offset=initial_ramp_offset, verbose=context.verbose)
            if context.verbose > 1:
                print('Process: %i; re-computed LSA weights: model_id: %s, cell_id: %i, after induction: %i, '
                      'ramp_offset: %.3f' % (os.getpid(), model_id, context.cell_id, context.induction,
                                             LSA_ramp_offset))
        else:
            LSA_ramp, LSA_delta_weights, LSA_ramp_offset, LSA_residual_score = \
                get_residual_score(LSA_delta_weights, target_ramp, ramp_x=context.binned_x, input_x=context.binned_x,
                                   interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                   ramp_scaling_factor=context.ramp_scaling_factor,
                                   induction_loc=context.mean_induction_start_loc, track_length=context.track_length,
                                   target_range=context.target_range, allow_offset=allow_offset,
                                   impose_offset=initial_ramp_offset, full_output=True)

        result['self_consistent_delta_residual_score'] = max(0., model_residual_score - LSA_residual_score)
    else:
        LSA_ramp = None

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

    if LSA_ramp is not None:
        ramp_amp['LSA'], ramp_width['LSA'], peak_shift['LSA'], ratio['LSA'], start_loc['LSA'], \
        peak_loc['LSA'], end_loc['LSA'], min_val['LSA'], min_loc['LSA'] = \
            calculate_ramp_features(ramp=LSA_ramp, induction_loc=context.mean_induction_start_loc,
                                    binned_x=context.binned_x, interp_x=context.default_interp_x,
                                    track_length=context.track_length)

    if context.verbose > 0:
        print('Process: %i; model_id: %s; cell: %i; induction: %i:' %
              (os.getpid(), model_id, context.cell_id, context.induction))
        print('exp: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, peak_loc: %.1f, '
              'end_loc: %.1f, min_val: %.1f, min_loc: %.1f' %
              (ramp_amp['target'], ramp_width['target'], peak_shift['target'], ratio['target'], start_loc['target'],
               peak_loc['target'], end_loc['target'], min_val['target'], min_loc['target']))
        if LSA_ramp is not None:
            print('LSA: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, ' 
                  'peak_loc: %.1f, end_loc: %.1f, min_val: %.1f, min_loc: %.1f, ramp_offset: %.3f' %
                  (ramp_amp['LSA'], ramp_width['LSA'], peak_shift['LSA'], ratio['LSA'], start_loc['LSA'],
                   peak_loc['LSA'], end_loc['LSA'], min_val['LSA'], min_loc['LSA'], LSA_ramp_offset))
        print('model: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, peak_loc: %.1f' 
              ', end_loc: %.1f, min_val: %.1f, min_loc: %.1f, ramp_offset: %.3f' %
              (ramp_amp['model'], ramp_width['model'], peak_shift['model'], ratio['model'], start_loc['model'],
               peak_loc['model'], end_loc['model'], min_val['model'], min_loc['model'], model_ramp_offset))
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
    if plot:
        bar_loc = max(10., np.max(model_ramp) + 1., np.max(target_ramp) + 1.) * 0.95
        fig, axes = plt.subplots(2)
        axes[1].plot(context.peak_locs, delta_weights)
        axes[1].hlines(context.peak_delta_weight * 1.05, xmin=context.mean_induction_start_loc,
                       xmax=context.mean_induction_stop_loc)
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
        axes[1].set_ylim([-context.peak_delta_weight * 1.05, context.peak_delta_weight * 1.1])
        clean_axes(axes)
        fig.suptitle('Cell_id: %i, Induction: %i' % (context.cell_id, context.induction))
        fig.tight_layout()
        fig.show()

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
            if LSA_ramp is not None:
                group.create_dataset('LSA_ramp', compression='gzip', data=LSA_ramp)
                group.create_dataset('LSA_weights', compression='gzip', data=np.add(LSA_delta_weights, 1.))
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
    if weights_path_distance_exceeds_threshold(delta_weights_snapshots, context.weights_path_distance_threshold):
        if context.verbose > 0:
            print('optimize_biBTSP_%s: calculate_model_ramp: pid: %i; model_id: %s; aborting - excessive fluctuations '
                  'in weights across laps; cell_id: %i, induction: %i' %
                  (BTSP_model_name, os.getpid(), model_id, context.cell_id, context.induction))
        return dict()

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
    local_signals = \
        np.divide(get_local_signal_population(local_signal_filter, context.down_rate_maps, context.down_dt),
                  local_signal_peak)

    signal_xrange = np.linspace(0., 1., 10000)
    pot_rate = np.vectorize(scaled_single_sigmoid(
        context.f_pot_th, context.f_pot_th + context.f_pot_peak, signal_xrange))
    dep_rate = np.vectorize(scaled_single_sigmoid(
        context.f_dep_th, context.f_dep_th + context.f_dep_peak, signal_xrange))

    peak_weight = context.peak_delta_weight + 1.
    peak_ramp_amp = np.max(ramp_snapshots) + 5.

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
    this_axis.set_xlabel('Normalized eligibility signal')
    this_axis.set_ylabel('Normalized rate')
    this_axis.set_ylim(0., this_axis.get_ylim()[1])
    # this_axis.set_xlim(0., 1.)
    this_axis.set_title('Nonlinear sensitivity to\nsynaptic eligibility signals', fontsize=mpl.rcParams['font.size'])
    this_axis.legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, fontsize=mpl.rcParams['font.size'])

    example_local_signals = dict()
    example_net_dwdt = dict()
    example_pre_rates = dict()

    current_weights = np.add(delta_weights_snapshots[induction_lap], 1.)
    current_ramp = ramp_snapshots[induction_lap]
    current_complete_ramp = get_complete_ramp(current_ramp, context.binned_x, context.position,
                                              context.complete_run_vel_gate, context.induction_gate, peak_ramp_amp)

    if induction_lap == 0:
        start_time = context.down_t[0]
    else:
        start_time = context.induction_stop_times[induction_lap - 1]
    if induction_lap == len(context.induction_start_times) - 1:
        stop_time = context.down_t[-1]
    else:
        stop_time = context.induction_start_times[induction_lap + 1]
    indexes = np.where((context.down_t >= start_time) & (context.down_t <= stop_time))

    this_current_ramp = np.interp(context.down_t, context.complete_t, current_complete_ramp)[indexes]
    this_t = context.down_t[indexes] / 1000.
    this_global_signal = global_signal[indexes]

    for name, i in viewitems(example_input_dict):
        example_pre_rates[name] = context.down_rate_maps[i][indexes]
        this_local_signal = local_signals[i]
        example_local_signals[name] = this_local_signal[indexes]
        this_pot_rate = np.multiply(pot_rate(this_local_signal[indexes]), this_global_signal)
        this_dep_rate = np.multiply(dep_rate(this_local_signal[indexes]), this_global_signal)
        this_net_weight_rate = context.k_pot * this_pot_rate * (peak_weight - current_weights[i]) - \
                               context.k_dep * this_dep_rate * current_weights[i]
        example_net_dwdt[name] = this_net_weight_rate

    colors = ['c', 'r']

    axes[0][1].get_shared_x_axes().join(axes[0][1], axes[0][2], axes[1][1], axes[1][2], axes[2][1], axes[2][2])
    axes[0][1].get_shared_y_axes().join(axes[0][1], axes[0][2])
    axes[1][1].get_shared_y_axes().join(axes[1][1], axes[1][2])
    axes[2][1].get_shared_y_axes().join(axes[2][1], axes[2][2])

    ymax1 = np.max(this_global_signal)
    ymax2 = 0.
    axes0_1_right = [axes[0][1].twinx(), axes[0][2].twinx()]
    axes0_1_right[0].get_shared_y_axes().join(axes0_1_right[0], axes0_1_right[1])
    for i, (name, index) in enumerate(viewitems(example_input_dict)):
        this_rate_map = example_pre_rates[name]
        this_local_signal = example_local_signals[name]
        this_net_dwdt = example_net_dwdt[name]
        ymax1 = max(ymax1, np.max(this_local_signal))
        axes[0][i + 1].plot(this_t, this_rate_map, c=colors[i], linewidth=1., label='Presynaptic firing rate')
        axes0_1_right[i].plot(this_t, this_current_ramp, c='k', linewidth=1., label='Postsynaptic voltage')
        axes[0][i + 1].set_title('%s:' % name, fontsize=mpl.rcParams['font.size'], y=1.4)
        axes[0][i + 1].set_xlabel('Time (s)')
        axes[0][i + 1].legend(loc=(0., 1.15), frameon=False, framealpha=0.5, handlelength=1,
                              fontsize=mpl.rcParams['font.size'])
        axes0_1_right[i].legend(loc=(0., 1.0), frameon=False, framealpha=0.5, handlelength=1,
                                fontsize=mpl.rcParams['font.size'])

        axes[1][i + 1].plot(this_t, this_local_signal, c=colors[i], label='Synaptic eligibility signal')
        axes[1][i + 1].plot(this_t, this_global_signal, c='k', label='Dendritic instructive signal', linewidth=0.75)
        axes[1][i + 1].fill_between(this_t, 0., np.minimum(this_local_signal, this_global_signal), alpha=0.5,
                                    facecolor=colors[i], label='Signal overlap')
        axes[1][i + 1].set_xlabel('Time (s)')
        axes[1][i + 1].legend(loc=(0., 1.0), frameon=False, framealpha=0.5, handlelength=1,
                              fontsize=mpl.rcParams['font.size'])

        axes[2][i + 1].plot(this_t, this_net_dwdt, c=colors[i])
        axes[2][i + 1].set_xlabel('Time (s)')
        ymax2 = max(ymax2, np.max(np.abs(this_net_dwdt)))

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
    ymax0_right = peak_ramp_amp / 0.9
    ymax1 /= 0.9
    ymax2 /= 0.85
    axes[0][1].set_ylim([0., ymax0_left])
    # axes[0][2].set_ylim([0., ymax1_left])
    axes0_1_right[0].set_ylim([0., ymax0_right])
    # axes0_1_right[1].set_ylim([0., ymax0_right])
    axes[1][1].set_ylim([0., ymax1])
    # axes[1][2].set_ylim([0., ymax1])
    axes[2][1].set_ylim([-ymax2, ymax2])
    # axes[2][2].set_ylim([-ymax2, ymax2])

    bar_loc0 = ymax0_left * 0.95
    bar_loc1 = ymax1 * 0.95
    bar_loc2 = ymax2 * 0.95
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
    axes[2][1].hlines(bar_loc2,
                      xmin=context.induction_start_times[induction_lap] / 1000.,
                      xmax=context.induction_stop_times[induction_lap] / 1000., linewidth=2)
    axes[2][2].hlines(bar_loc2,
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
    yrange2 = np.round((np.arange(-ymax2, ymax2, 0.2) / 0.2)) * 0.2
    axes[2][1].set_yticks(yrange2)
    axes[2][2].set_yticks(yrange2)
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
    fig.suptitle('Synaptic resource-limited model B (cell %i)' % cell_id,
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

    for w in np.linspace(0., 1., 10):
        net_delta_weight = pot_rate(signal_xrange) * (1. - w) - dep_rate(signal_xrange) * dep_scale * w
        axes[1][0].plot(signal_xrange, net_delta_weight, c=cmap(w))
    axes[1][0].axhline(y=0., linestyle='--', c='grey')
    axes[1][0].set_xlabel('Normalized eligibility signal')
    axes[1][0].set_ylabel('Normalized rate')
    axes[1][0].set_title('Rate of change\nin synaptic weight', fontsize=mpl.rcParams['font.size'], pad=10.)
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1][0])
    cbar.set_label('Initial synaptic\nweight (normalized)', rotation=270., labelpad=20.)

    clean_axes(axes)
    fig.suptitle('Weight-dependent model B (cell %i)' % cell_id,
                 fontsize=mpl.rcParams['font.size'], x=0.02, ha='left')
    fig.subplots_adjust(left=0.1, hspace=1.075, wspace=0.7, right=0.955, top=0.925, bottom=0.05)
    fig.show()

    context.update(locals())


def plot_model_summary_figure(cell_id, model_file_path, induction_lap=0, target_min_delay=2750.,
                              target_delay_range=250.):
    """

    :param cell_id: int
    :param model_file_path: str (path)
    :param induction_lap: int
    :param target_min_delay
    :param target_delay_range
    """
    if (cell_id, 2) not in context.data_keys:
        raise KeyError('plot_model_summary_figure: cell_id: %i, induction: 2 not found' % cell_id)
    if not os.path.isfile(model_file_path):
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
        for lap in range(len(group['ramp_snapshots'])):
            ramp_snapshots.append(group['ramp_snapshots'][str(lap)][:])
        delta_weights_snapshots = []
        for lap in range(len(group['delta_weights_snapshots'])):
            delta_weights_snapshots.append(group['delta_weights_snapshots'][str(lap)][:])
        final_weights = group['model_weights'][:]

    import_data(cell_id, 2)
    print(x)
    update_source_contexts(x, context)

    initial_exp_ramp = context.exp_ramp['before']  # context.exp_ramp_raw['before']
    target_ramp = context.exp_ramp['after']  # context.exp_ramp_raw['after']

    global_signal = np.divide(get_global_signal(context.down_induction_gate, global_filter), global_signal_peak)
    local_signals = \
        np.divide(get_local_signal_population(local_signal_filter, context.down_rate_maps, context.down_dt),
                  local_signal_peak)

    signal_xrange = np.linspace(0., 1., 10000)
    pot_rate = np.vectorize(scaled_single_sigmoid(
        context.f_pot_th, context.f_pot_th + context.f_pot_peak, signal_xrange))
    dep_rate = np.vectorize(scaled_single_sigmoid(
        context.f_dep_th, context.f_dep_th + context.f_dep_peak, signal_xrange))

    peak_weight = context.peak_delta_weight + 1.
    peak_ramp_amp = np.max(ramp_snapshots) + 5.

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

    fig, axes = plt.subplots(3, figsize=(4, 8.5))

    current_weights = np.add(delta_weights_snapshots[induction_lap], 1.)
    current_ramp = ramp_snapshots[induction_lap]
    current_complete_ramp = get_complete_ramp(current_ramp, context.binned_x, context.position,
                                              context.complete_run_vel_gate, context.induction_gate, peak_ramp_amp)

    if induction_lap == 0:
        start_time = context.down_t[0]
    else:
        start_time = context.induction_stop_times[induction_lap - 1]
    if induction_lap == len(context.induction_start_times) - 1:
        stop_time = context.down_t[-1]
    else:
        stop_time = context.induction_start_times[induction_lap + 1]
    indexes = np.where((context.down_t >= start_time) & (context.down_t <= stop_time))

    this_current_ramp = np.interp(context.down_t, context.complete_t, current_complete_ramp)[indexes]
    this_t = context.down_t[indexes] / 1000.
    this_global_signal = global_signal[indexes]

    example_pre_rate = context.down_rate_maps[depressing_example_index][indexes]
    example_current_normalized_weight = current_weights[depressing_example_index] / peak_weight
    example_local_signal = local_signals[depressing_example_index][indexes]
    example_pot_elig_signal = example_local_signal * (1. - example_current_normalized_weight)
    example_dep_elig_signal = example_local_signal * example_current_normalized_weight
    example_pot_rate = peak_weight * context.k_pot * np.multiply(pot_rate(example_pot_elig_signal), this_global_signal)
    example_dep_rate = peak_weight * context.k_dep * np.multiply(dep_rate(example_dep_elig_signal), this_global_signal)
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

    axes[1].plot(this_t, example_pot_elig_signal, c='c', linewidth=1., label='Potentiation $signal_{eligibility}$')
    axes[1].plot(this_t, example_dep_elig_signal, c='r', linewidth=1.,
                 label='Depression $signal_{eligibility}$')
    axes[1].plot(this_t, this_global_signal, c='k', linewidth=1., label='Dendritic $signal_{instructive}$')
    axes[1].fill_between(this_t, 0., np.minimum(example_pot_elig_signal, this_global_signal), alpha=0.5,
                         facecolor='c', edgecolor='none', label='Potentiation signal overlap')
    axes[1].fill_between(this_t, 0., np.minimum(example_dep_elig_signal, this_global_signal),
                         alpha=0.5, facecolor='r', edgecolor='none', label='Depression signal overlap')
    leg = axes[1].legend(loc=(0., 1.0), frameon=False, framealpha=0.5, handlelength=1,
                         fontsize=mpl.rcParams['font.size'])
    for line in leg.get_lines():
        line.set_linewidth(2.)

    axes[2].plot(this_t, example_pot_rate, c='c', linewidth=1., label='Potentiation rate')
    axes[2].plot(this_t, example_dep_rate, c='r', linewidth=1., label='Depression rate')
    axes[2].plot(this_t, example_net_dwdt, c='k', linewidth=1., label='Net dW/dt')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend(loc=(0., 1.0), frameon=False, framealpha=0.5, handlelength=1, fontsize=mpl.rcParams['font.size'])

    xmin = max(-5., np.min(this_t))
    xmax = np.max(this_t)
    axes[0].set_xlim(xmin, xmax)
    axes[0].set_xticks(np.round((np.arange(xmin, xmax, 5.) / 5.)) * 5.)
    axes[1].set_xticks(np.round((np.arange(xmin, xmax, 5.) / 5.)) * 5.)
    axes[2].set_xticks(np.round((np.arange(xmin, xmax, 5.) / 5.)) * 5.)

    ymax0_left = context.input_field_peak_rate / 0.85
    ymax0_right = peak_ramp_amp / 0.85

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
    fig.suptitle('History-dependent model B (cell %i)' % cell_id,
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
    groups = ['spont', 'exp1', 'exp2']
    grouped_feature_names = ['delta_val_at_target_peak', 'delta_val_at_model_peak', 'delta_width', 'delta_peak_shift',
                             'delta_asymmetry', 'delta_min_loc', 'delta_val_at_target_min', 'delta_val_at_model_min',
                             'residual_score']
    feature_names = ['self_consistent_delta_residual_score']
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
              default='config/optimize_biBTSP_WD_D_cli_config.yaml')
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
    python -i optimize_biBTSP_WD_D.py --cell_id=1 --plot --framework=serial --interactive

    To analyze and export data to using MPI parallelism with 1 controller process and N - 1 worker processes:
    mpirun -n N python -i -m mpi4py.futures -m nested.analyze --config-file-path=$PATH_TO_CONFIG_YAML \
        --param-file-path=$PATH_TO_PARAM_YAML --model-key=$VALID_KEY_IN_PARAM_YAML --disp --cell_id=1
        --framework=mpi --export

    To plot results previously exported to a file on a single process:
    python -i optimize_biBTSP_WD_D.py --param_file_path=$PATH_TO_PARAM_YAML --model_key=$VALID_KEY_IN_PARAM_YAML \
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
        print(int(context.kwargs['cell_id']), export_file_path, exported_data_key)
        #context.interface.execute(plot_model_summary_figure, int(context.kwargs['cell_id']), export_file_path,
        #                          exported_data_key)
        context.interface.execute(plot_model_summary_supp_figure, int(context.kwargs['cell_id']), export_file_path,
                                  exported_data_key)
    elif not debug:
        run_tests()

    if not context.interactive:
        context.interface.stop()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
