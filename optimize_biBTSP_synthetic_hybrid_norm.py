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

Features/assumptions of hybrid model (dependent on both voltage and weight):
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
f_pot is linear.
7) f_dep represents the "sensitivity" of the reverse process to the presence of the local_signal. The transformation
f_dep has the flexibility to be any segment of a sigmoid (so can be linear, exponential rise, or saturating).

biBTSP_synthetic_hybrid_norm:
Spine depolarization is modulated by local dendritic depolarization.
Eligibility signal filter acts on spine depolarization.
Gain function f_pot is linear and f_dep is sigmoidal.

"""
__author__ = 'milsteina'
from biBTSP_utils import *
from nested.parallel import *
from nested.optimize_utils import *
import click

context = Context()


BTSP_model_name = 'synthetic_hybrid_norm'


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

    input_field_peak_rate = 1.
    ramp_scaling_factor = 0.14970976921836665

    with h5py.File(context.data_file_path, 'r') as f:
        dt = f['defaults'].attrs['dt']  # ms
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

    down_dt = 10.  # ms, to speed up optimization
    if 'num_induction_laps' not in context():
        num_induction_laps = 1
    else:
        num_induction_laps = context.num_induction_laps
    induction_dur = 300.  # ms
    context.update(locals())

    context.induction = None
    context.data_cache = defaultdict(dict)


def load_data(induction, condition='control'):
    """

    :param induction: int
    :param condition: str
    """
    induction = int(induction)
    context.condition = str(condition)

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

        induction_context.target_ramp = defaultdict(dict)
        induction_context.LSA_weights = defaultdict(dict)
        induction_context.ramp_offset = dict()
        if induction == 2:
            if 1 in context.data_cache:
                induction_context.target_ramp['before']['control'] = \
                    context.data_cache[1].target_ramp['after']['control']
                induction_context.LSA_weights['before']['control'] = \
                    context.data_cache[1].LSA_weights['after']['control']
            else:
                induction_loc = context.induction_loc['1']
                induction_stop_index = np.where(context.default_interp_x >= induction_loc)[0][0] + \
                                         int(context.induction_dur / context.dt)
                induction_stop_loc = context.default_interp_x[induction_stop_index]

                target_ramp = get_target_synthetic_ramp(induction_loc, ramp_x=context.binned_x,
                                                        track_length=context.track_length,
                                                        target_peak_val=context.target_peak_val_1,
                                                        target_min_val=0., target_asymmetry=1.1,
                                                        target_peak_shift=context.target_peak_shift_1,
                                                        target_ramp_width=context.target_ramp_width)
                induction_context.target_ramp['before']['control'], \
                induction_context.LSA_weights['before']['control'], _, _ = \
                    get_delta_weights_LSA(target_ramp, ramp_x=context.binned_x, input_x=context.binned_x,
                                          interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                          peak_locs=context.peak_locs, ramp_scaling_factor=context.ramp_scaling_factor,
                                          induction_start_loc=induction_loc, induction_stop_loc=induction_stop_loc,
                                          track_length=context.track_length, target_range=context.target_range,
                                          bounds=(context.min_delta_weight, context.target_peak_delta_weight),
                                          verbose=context.verbose)

            # the shape of this target_ramp was estimated from gaussian regression from experimental data
            target_delta_ramp_2_dep = get_target_synthetic_ramp(110., ramp_x=context.binned_x,
                                                                track_length=context.track_length, target_peak_val=4.5,
                                                                target_min_val=0., target_asymmetry=0.6,
                                                                target_peak_shift=0., target_ramp_width=90.)
            target_delta_ramp_2_pot = get_target_synthetic_ramp(induction_context.mean_induction_start_loc,
                                                                ramp_x=context.binned_x,
                                                                track_length=context.track_length, target_peak_val=8.,
                                                                target_min_val=0., target_asymmetry=1.7,
                                                                target_peak_shift=-5., target_ramp_width=144.)
            target_delta_ramp_2 = np.subtract(target_delta_ramp_2_pot, target_delta_ramp_2_dep)
            target_ramp_2 = np.add(induction_context.target_ramp['before']['control'], target_delta_ramp_2)
            induction_context.target_ramp['after']['control'], \
            induction_context.LSA_weights['after']['control'], _, _ = \
                get_delta_weights_LSA(target_ramp_2, ramp_x=context.binned_x, input_x=context.binned_x,
                                      interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                      peak_locs=context.peak_locs, ramp_scaling_factor=context.ramp_scaling_factor,
                                      induction_start_loc=induction_context.mean_induction_start_loc,
                                      induction_stop_loc=induction_context.mean_induction_stop_loc,
                                      track_length=context.track_length, target_range=context.target_range,
                                      bounds=(context.min_delta_weight, context.target_peak_delta_weight),
                                      verbose=context.verbose)
            induction_context.target_ramp['after']['hyper'] = induction_context.target_ramp['after']['control']
            induction_context.ramp_offset['control'] = 0.
            induction_context.ramp_offset['hyper'] = context.target_ramp_offset_2_hyper

            if context.plot and context.condition == 'control':
                x_start = induction_context.mean_induction_start_loc
                x_end = induction_context.mean_induction_stop_loc
                ylim = 13.
                fig, axes = plt.subplots(2)
                axes[0].plot(context.binned_x, induction_context.target_ramp['after']['control'],
                             label='Control', color='k')
                axes[0].hlines(ylim + 0.2, xmin=x_start, xmax=x_end, linewidth=2, colors='k')
                axes[0].set_xlabel('Location (cm)')
                axes[0].set_ylabel('Ramp amplitude (mV)')
                axes[0].set_xlim([0., context.track_length])
                axes[0].set_ylim([-5., 15.])
                axes[0].legend(loc='best', frameon=False, framealpha=0.5)
                axes[0].set_title('Target: Induction 2')

                axes[1].plot(context.binned_x, np.ones_like(context.binned_x) *
                             induction_context.ramp_offset['control'], c='k', label='Control')
                axes[1].plot(context.binned_x, np.ones_like(context.binned_x) *
                             induction_context.ramp_offset['hyper'], c='c', label='Hyper')
                axes[1].hlines(context.max_dend_depo * 1.05, xmin=x_start, xmax=x_end, linewidth=2, colors='k')
                axes[1].set_xlabel('Location (cm)')
                axes[1].set_ylabel('Vm offset (mV)')
                axes[1].set_xlim([0., context.track_length])
                axes[1].set_ylim([context.min_dend_depo * 1.1, context.max_dend_depo * 1.1])
                clean_axes(axes)
                fig.tight_layout()
                fig.show()
        else:
            induction_context.target_ramp['before']['control'] = np.zeros_like(context.binned_x)
            induction_context.LSA_weights['before']['control'] = np.zeros_like(context.peak_locs)
            induction_loc = induction_context.mean_induction_start_loc
            induction_stop_loc = induction_context.mean_induction_stop_loc

            control_target_ramp = get_target_synthetic_ramp(induction_loc, ramp_x=context.binned_x,
                                                    track_length=context.track_length,
                                                    target_peak_val=context.target_peak_val_1, target_min_val=0.,
                                                    target_asymmetry=1.1,
                                                    target_peak_shift=context.target_peak_shift_1,
                                                    target_ramp_width=context.target_ramp_width)
            induction_context.target_ramp['after']['control'], \
            induction_context.LSA_weights['after']['control'], _, _ = \
                get_delta_weights_LSA(control_target_ramp, ramp_x=context.binned_x, input_x=context.binned_x,
                                      interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                      peak_locs=context.peak_locs, ramp_scaling_factor=context.ramp_scaling_factor,
                                      induction_start_loc=induction_loc, induction_stop_loc=induction_stop_loc,
                                      track_length=context.track_length, target_range=context.target_range,
                                      bounds=(context.min_delta_weight, context.target_peak_delta_weight),
                                      verbose=context.verbose)

            depo_target_ramp = control_target_ramp * context.target_peak_val_1_depo / context.target_peak_val_1
            induction_context.target_ramp['after']['depo'], \
            induction_context.LSA_weights['after']['depo'], _, _ = \
                get_delta_weights_LSA(depo_target_ramp, ramp_x=context.binned_x, input_x=context.binned_x,
                                      interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                      peak_locs=context.peak_locs, ramp_scaling_factor=context.ramp_scaling_factor,
                                      induction_start_loc=induction_loc, induction_stop_loc=induction_stop_loc,
                                      track_length=context.track_length, target_range=context.target_range,
                                      bounds=(context.min_delta_weight, context.target_peak_delta_weight),
                                      verbose=context.verbose)

            hyper_target_ramp = control_target_ramp * context.target_peak_val_1_hyper / context.target_peak_val_1
            induction_context.target_ramp['after']['hyper'], \
            induction_context.LSA_weights['after']['hyper'], _, _ = \
                get_delta_weights_LSA(hyper_target_ramp, ramp_x=context.binned_x, input_x=context.binned_x,
                                      interp_x=context.default_interp_x, input_rate_maps=context.input_rate_maps,
                                      peak_locs=context.peak_locs, ramp_scaling_factor=context.ramp_scaling_factor,
                                      induction_start_loc=induction_loc, induction_stop_loc=induction_stop_loc,
                                      track_length=context.track_length, target_range=context.target_range,
                                      bounds=(context.min_delta_weight, context.target_peak_delta_weight),
                                      verbose=context.verbose)

            induction_context.ramp_offset['control'] = 0.
            induction_context.ramp_offset['depo'] = context.target_ramp_offset_1_depo
            induction_context.ramp_offset['hyper'] = context.target_ramp_offset_1_hyper

            if context.plot and context.condition == 'control':
                x_start = induction_context.mean_induction_start_loc
                x_end = induction_context.mean_induction_stop_loc
                ylim = 13.
                fig, axes = plt.subplots(2)
                axes[0].plot(context.binned_x, induction_context.target_ramp['after']['control'], label='Control',
                             color='k')
                axes[0].plot(context.binned_x, induction_context.target_ramp['after']['depo'], label='Depo',
                             color='c')
                axes[0].plot(context.binned_x, induction_context.target_ramp['after']['hyper'], label='Hyper',
                             color='r')
                axes[0].hlines(ylim + 0.2, xmin=x_start, xmax=x_end, linewidth=2, colors='k')
                axes[0].set_xlabel('Location (cm)')
                axes[0].set_ylabel('Ramp amplitude (mV)')
                axes[0].set_xlim([0., context.track_length])
                axes[0].set_ylim([-5., 15.])
                axes[0].legend(loc='best', frameon=False, framealpha=0.5)
                axes[0].set_title('Target: Induction 1')

                axes[1].plot(context.binned_x, np.ones_like(context.binned_x) *
                             induction_context.ramp_offset['control'], c='k', label='Control')
                axes[1].plot(context.binned_x, np.ones_like(context.binned_x) *
                             induction_context.ramp_offset['depo'], c='r', label='Depo')
                axes[1].plot(context.binned_x, np.ones_like(context.binned_x) *
                             induction_context.ramp_offset['hyper'], c='c', label='Hyper')
                axes[1].hlines(context.max_dend_depo * 1.05, xmin=x_start, xmax=x_end, linewidth=2, colors='k')
                axes[1].set_xlabel('Location (cm)')
                axes[1].set_ylabel('Vm offset (mV)')
                axes[1].set_xlim([0., context.track_length])
                axes[1].set_ylim([context.min_dend_depo * 1.1, context.max_dend_depo * 1.1])
                clean_axes(axes)
                fig.tight_layout()
                fig.show()

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
        print('optimize_biBTSP_%s: process: %i loaded data for induction: %i, condition: %s' %
              (BTSP_model_name, os.getpid(), induction, condition))


def update_model_params(x, local_context):
    """

    :param x: array
    :param local_context: :class:'Context'
    """
    if local_context is None:
        local_context = context
    local_context.update(param_array_to_dict(x, local_context.param_names))


def get_spine_depo_f(dend_depo):
    """

    :param dend_depo: float
    :return: callable
    """
    dend_depo_mod = -context.dend_depo_mod_range / (context.max_dend_depo - context.min_dend_depo) * dend_depo
    this_spine_th = min(1., max(0., context.spine_th + dend_depo_mod))
    return np.vectorize(scaled_single_sigmoid(this_spine_th, this_spine_th + context.spine_half_width))


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

    signal_xrange = np.linspace(0., 1., 10000)
    dend_depo_range = np.linspace(context.min_dend_depo, context.max_dend_depo, 10000)
    pot_rate = lambda x: x
    dep_rate = np.vectorize(scaled_single_sigmoid(
        context.f_dep_th, context.f_dep_th + context.f_dep_half_width, signal_xrange))

    this_dend_depo = context.ramp_offset[context.condition]
    if context.condition == 'control':
        this_spine_th = context.spine_th
    elif context.condition == 'depo':
        this_spine_th = context.spine_th_depo
    elif context.condition == 'hyper':
        this_spine_th = context.spine_th_hyper

    get_spine_depo = np.vectorize(scaled_single_sigmoid(this_spine_th, this_spine_th + context.spine_half_width))

    if plot and context.induction == 1 and context.condition == 'control':
        fig, axes = plt.subplots(1, 3, figsize=(10., 4.))
        axes[0].plot(signal_xrange, get_spine_depo(signal_xrange), c='k', label='Control')
        this_spine_depo_f = np.vectorize(scaled_single_sigmoid(context.spine_th_depo,
                                                               context.spine_th_depo + context.spine_half_width))
        axes[0].plot(signal_xrange, this_spine_depo_f(signal_xrange), c='r', label='Depolarized')
        this_spine_depo_f = np.vectorize(scaled_single_sigmoid(context.spine_th_hyper,
                                                               context.spine_th_hyper + context.spine_half_width))
        axes[0].plot(signal_xrange, this_spine_depo_f(signal_xrange), c='c', label='Hyperpolarized')
        axes[0].set_xlabel('Presynaptic firing rate (normalized)')
        axes[0].set_ylabel('Spine depolarization (normalized)')
        axes[0].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)

        axes[1].plot(local_signal_filter_t / 1000., local_signal_filter / np.max(local_signal_filter), color='r',
                  label='Eligibility signal filter')
        axes[1].plot(global_filter_t / 1000., global_filter / np.max(global_filter), color='k',
                  label='Instructive signal filter')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Normalized amplitude')
        axes[1].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
        axes[1].set_xlim(-0.5, max(5000., local_signal_filter_t[-1], global_filter_t[-1]) / 1000.)

        dep_scale = context.k_dep / context.k_pot
        axes[2].plot(signal_xrange, pot_rate(signal_xrange), c='r', label='Potentiation rate')
        axes[2].plot(signal_xrange, dep_rate(signal_xrange) * dep_scale, c='c', label='Depression rate')
        axes[2].set_xlabel('Plasticity signal overlap (a.u.)')
        axes[2].set_ylabel('Normalized rate')
        axes[2].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
        clean_axes(axes)
        fig.tight_layout()
        fig.show()

    peak_weight = context.peak_delta_weight + 1.

    initial_delta_weights = context.LSA_weights['before']['control']
    initial_ramp = context.target_ramp['before']['control']

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
            if not 0.9 * np.max(context.target_ramp['before']['control']) < np.max(initial_ramp) < \
                   1.1 * np.max(context.target_ramp['before']['control']):
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

    target_ramp = context.target_ramp['after'][context.condition]

    if plot:
        fig, axes = plt.subplots()
        fig.suptitle('Induction: %i (%s)' % (context.induction, context.condition), y=1.)
        axes.set_xlabel('Time (s)')
        axes.set_ylabel('Integrated plasticity\nsignal overlap (a.u.)')

        fig2, axes2 = plt.subplots(1, 3, figsize=(10., 4.))
        fig2.suptitle('Induction: %i (%s)' % (context.induction, context.condition))
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
        for i, (this_rate_map, this_current_normalized_weight) in \
                enumerate(zip(context.down_rate_maps, current_normalized_weights)):
            this_normalized_rate_map = this_rate_map / context.input_field_peak_rate
            this_spine_depo = get_spine_depo(this_normalized_rate_map)
            this_local_signal = get_local_signal(this_spine_depo, local_signal_filter)
            this_signal_overlap = np.multiply(this_local_signal[indexes], global_signal[indexes])
            this_pot_rate = np.trapz(pot_rate(this_signal_overlap), dx=context.down_dt / 1000.)
            this_dep_rate = np.trapz(dep_rate(this_signal_overlap), dx=context.down_dt / 1000.)
            this_normalized_delta_weight = context.k_pot * this_pot_rate * (1. - this_current_normalized_weight) - \
                                           context.k_dep * this_dep_rate * this_current_normalized_weight
            this_next_normalized_weight = \
                max(0., min(1., this_current_normalized_weight + this_normalized_delta_weight))
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

        if context.induction == 1 and context.condition == 'control' and induction_lap == 0:
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
        print('Process: %i; induction: %i; condition: %s:' % (os.getpid(), context.induction, context.condition))
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
                group.create_dataset('dend_depo_range', compression='gzip', data=dend_depo_range)
                group.create_dataset('param_names', compression='gzip', data=np.asarray(context.param_names, dtype='S'))
                group.attrs['input_field_width'] = context.input_field_width
                group.attrs['ramp_scaling_factor'] = context.ramp_scaling_factor
            exported_data_key = 'exported_data'
            description = 'model_ramp_features'
            cell_key = 'synthetic'
            induction_key = str(context.induction)
            condition_key = str(context.condition)
            group = get_h5py_group(f, [model_id, exported_data_key, cell_key, induction_key, condition_key,
                                       description], create=True)
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
            group.create_dataset('dend_depo', compression='gzip', data=this_dend_depo)
            group.attrs['local_signal_peak'] = local_signal_peak
            group.attrs['global_signal_peak'] = global_signal_peak
            group.attrs['mean_induction_start_loc'] = context.mean_induction_start_loc
            group.attrs['mean_induction_stop_loc'] = context.mean_induction_stop_loc
            group.attrs['induction_start_times'] = context.induction_start_times
            group.attrs['induction_stop_times'] = context.induction_stop_times
            # group.attrs['track_start_times'] = context.track_start_times
            # group.attrs['track_stop_times'] = context.track_stop_times
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

    return {context.induction: {context.condition: result}}


def plot_model_summary_figure(export_file_path=None, exported_data_key=None, induction_lap=0):
    """

    :param export_file_path: str (path)
    :param exported_data_key: str
    :param induction_lap: int
    """
    if export_file_path is None:
        raise IOError('plot_model_summary_figure: no export_file_path provided')
    elif not os.path.isfile(export_file_path):
        raise IOError('plot_model_summary_figure: invalid export_file_path: %s' % export_file_path)
    initial_weights = defaultdict(dict)
    initial_ramp = defaultdict(dict)
    model_ramp = defaultdict(dict)
    final_weights = defaultdict(dict)
    ramp_offset = defaultdict(dict)
    induction_start_loc = defaultdict(dict)
    induction_stop_loc = defaultdict(dict)
    induction_start_times = defaultdict(dict)
    induction_stop_times = defaultdict(dict)
    description = 'model_ramp_features'
    with h5py.File(export_file_path, 'r') as f:
        source = get_h5py_group(f, [exported_data_key, 'exported_data', 'synthetic'])
        group = source['1']['control'][description]
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
            for condition in source[induction_key]:
                group = source[induction_key][condition][description]
                initial_weights[int(induction_key)][condition] = group['initial_weights'][:]
                initial_ramp[int(induction_key)][condition] = group['initial_model_ramp'][:]
                model_ramp[int(induction_key)][condition] = group['model_ramp'][:]
                final_weights[int(induction_key)][condition] = group['model_weights'][:]
                ramp_offset[int(induction_key)][condition] = group['ramp_offset'][:]
                induction_start_loc[int(induction_key)][condition] = group.attrs['mean_induction_start_loc']
                induction_stop_loc[int(induction_key)][condition] = group.attrs['mean_induction_stop_loc']
                induction_start_times[int(induction_key)][condition] = group.attrs['induction_start_times']
                induction_stop_times[int(induction_key)][condition] = group.attrs['induction_stop_times']

    load_data(1)
    load_data(2)
    update_source_contexts(x)

    global_signal = np.divide(get_global_signal(context.down_induction_gate, global_filter), global_signal_peak)
    signal_xrange = np.linspace(0., 1., 10000)
    vrange = np.linspace(context.min_delta_ramp, context.peak_delta_ramp, 10000)
    pot_rate = np.vectorize(scaled_single_sigmoid(
        context.f_pot_th, context.f_pot_th + context.f_pot_half_width, signal_xrange))
    dep_rate = np.vectorize(scaled_single_sigmoid(
        context.f_dep_th, context.f_dep_th + context.f_dep_half_width, signal_xrange))
    phi_pot = np.vectorize(lambda x: 1.)
    phi_dep = np.vectorize(lambda x: 1.)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    dep_scale = context.k_dep / context.k_pot
    axes[0].plot(vrange, phi_pot(vrange), c='r', label='Potentiation')
    axes[0].plot(vrange, phi_dep(vrange), c='c', label='Depression')
    axes[0].set_xlabel('Relative ramp amplitude')
    axes[0].set_ylabel('Modulation factor')
    axes[0].set_xlim((context.min_delta_ramp, context.peak_delta_ramp))
    axes[0].set_title('Voltage modulation', fontsize=mpl.rcParams['font.size'])
    axes[0].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, fontsize=mpl.rcParams['font.size'])

    xmax = max(5000., local_signal_filter_t[-1], global_filter_t[-1]) / 1000.
    xmax = math.ceil(xmax)
    axes[1].plot(local_signal_filter_t / 1000., local_signal_filter / np.max(local_signal_filter), color='gray',
                   label='Synaptic\neligibility signal')
    axes[1].plot(global_filter_t / 1000., global_filter / np.max(global_filter), color='k',
                   label='Dendritic\ninstructive signal')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Normalized amplitude')
    axes[1].set_ylim(0., axes[1].get_ylim()[1])
    axes[1].set_xlim(-0.5, xmax)
    axes[1].set_title('Plasticity signal kinetics', fontsize=mpl.rcParams['font.size'])
    axes[1].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, fontsize=mpl.rcParams['font.size'])

    axes[2].plot(signal_xrange, pot_rate(signal_xrange), c='r', label='Potentiation')
    axes[2].plot(signal_xrange, dep_rate(signal_xrange) * dep_scale, c='c', label='Depression')
    axes[2].set_xlabel('Normalized eligibility\nsignal amplitude (a.u.)')
    axes[2].set_ylabel('Normalized rate')
    axes[2].set_xlim((0., 1.))
    # axes[2].set_ylim((0., 1.))
    axes[2].set_title('Rate of change in\nsynaptic weight', fontsize=mpl.rcParams['font.size'])
    axes[2].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, fontsize=mpl.rcParams['font.size'])
    clean_axes(axes)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.35)
    fig.show()

    ordered_conditions = {1: ['control', 'depo', 'hyper'], 2: ['control', 'hyper']}
    condition_labels = {'control': 'Control', 'depo': 'Depolarized', 'hyper': 'Hyperpolarized'}
    condition_colors = {'control': 'k', 'depo': 'c', 'hyper': 'r'}

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    ylim = 13.
    for col, induction in enumerate(range(1, 3)):
        for condition in ordered_conditions[induction]:
            label = condition_labels[condition]
            color = condition_colors[condition]
            if condition == 'control':
                x_start = induction_start_loc[induction][condition]
                x_end = induction_stop_loc[induction][condition]
                axes[0][col].hlines(ylim + 0.2, xmin=x_start, xmax=x_end, linewidth=2, colors='k')
                axes[0][col].set_xlabel('Location (cm)')
                axes[0][col].set_ylabel('Change in ramp\namplitude (mV)')
                axes[0][col].set_xlim([0., context.track_length])
                axes[0][col].set_ylim([-5., 15.])
                axes[0][col].set_title('Induction %i' % induction, fontsize=mpl.rcParams['font.size'])
                axes[1][col].set_xlabel('Location (cm)')
                axes[1][col].set_ylabel('Vm offset (mV)')
                axes[1][col].set_xlim([0., context.track_length])
                axes[1][col].set_ylim([-45., 15.])
            axes[0][col].plot(context.binned_x, np.subtract(model_ramp[induction][condition],
                                                            initial_ramp[induction][condition]),
                              label=label, color=color)

            this_ramp_offset = ramp_offset[induction][condition]
            interp_ramp_offset = np.zeros_like(context.generic_x)
            offset_indexes = np.where(this_ramp_offset != 0)[0]
            if len(offset_indexes) > 0:
                if offset_indexes[0] == 0:
                    offset_start_loc = 0.
                else:
                    offset_start_loc = context.binned_x[offset_indexes[0]]
                if offset_indexes[-1] == len(context.binned_x) - 1:
                    offset_stop_loc = context.track_length
                else:
                    offset_stop_loc = context.binned_x[offset_indexes[-1]]
                offset_val = this_ramp_offset[offset_indexes[0]]
                indexes = np.where((context.generic_x >= offset_start_loc) &
                                   (context.generic_x <= offset_stop_loc))[0]
                interp_ramp_offset[indexes] = offset_val
            axes[1][col].plot(context.generic_x, interp_ramp_offset, c=color)

    axes[0][0].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, fontsize=mpl.rcParams['font.size'])
    axes[0][1].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, fontsize=mpl.rcParams['font.size'])
    clean_axes(axes)
    fig.tight_layout()
    fig.show()
    """
    resolution = 10
    input_sample_indexes = np.arange(0, len(context.peak_locs), resolution)

    example_input_dict = {}

    sample_time_delays = []
    for index in input_sample_indexes:
        this_peak_loc = context.peak_locs[index]
        this_delay = (this_peak_loc - context.mean_induction_start_loc) / context.default_run_vel / 1000.
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
    mpl.rcParams['font.size'] = 12.
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['axes.titlepad'] = 2.
    mpl.rcParams['mathtext.default'] = 'regular'
    mpl.rcParams['axes.unicode_minus'] = True
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
    this_axis.legend(loc=(0.05, 0.95), frameon=False, framealpha=0.5, handlelength=1,
                     fontsize=mpl.rcParams['font.size'])
    this_axis.set_xticks(np.arange(0., context.track_length, 45.))

    this_axis = fig.add_subplot(gs0[0, 2])
    axes.append(this_axis)
    xmax = max(5000., local_signal_filter_t[-1], global_filter_t[-1]) / 1000.
    xmax = math.ceil(xmax)
    this_axis.plot(local_signal_filter_t / 1000., local_signal_filter / np.max(local_signal_filter), color='gray',
                   label='Synaptic\neligibility signal')
    this_axis.plot(global_filter_t / 1000., global_filter / np.max(global_filter), color='k',
                   label='Dendritic\ngating signal')
    this_axis.set_xlabel('Time (s)')
    this_axis.set_ylabel('Normalized amplitude')
    this_axis.set_ylim(0., this_axis.get_ylim()[1])
    this_axis.set_xlim(-0.5, xmax)
    this_axis.set_title('Plasticity signal kinetics', fontsize=mpl.rcParams['font.size'])
    this_axis.legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, fontsize=mpl.rcParams['font.size'])

    this_axis = fig.add_subplot(gs0[0, 3])
    axes.append(this_axis)
    dep_scale = context.k_dep / context.k_pot
    this_axis.plot(signal_xrange, pot_rate(signal_xrange), label='Potentiation', c='c')
    this_axis.plot(signal_xrange, dep_rate(signal_xrange) * dep_scale, label='Depression', c='r')
    this_axis.set_xlabel('Normalized eligibility signal')
    this_axis.set_ylabel('Normalized rate')
    this_axis.set_ylim(0., this_axis.get_ylim()[1])
    this_axis.set_xlim(0., 1.)
    this_axis.set_title('State transition rates', fontsize=mpl.rcParams['font.size'])
    this_axis.legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, fontsize=mpl.rcParams['font.size'])
    clean_axes(axes)

    colors = ['c', 'r']
    """
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
    """
    axes2 = []
    ymax = 0.
    for row, weights in zip(range(1, 3), [initial_weights, final_weights]):
        this_axis = fig.add_subplot(gs0[row, 0])
        axes2.append(this_axis)
        this_max_rate_map = np.zeros_like(context.input_rate_maps[0])
        for i in (index for index in input_sample_indexes if index not in viewvalues(example_input_dict)):
            rate_map = np.array(context.input_rate_maps[i])
            rate_map *= weights[i] * context.ramp_scaling_factor
            ymax = max(ymax, np.max(rate_map))
            this_axis.plot(context.binned_x, rate_map, c='gray', zorder=0, linewidth=0.75)  # , alpha=0.5)
        for i, (name, index) in enumerate(viewitems(example_input_dict)):
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

    fig2, axes2 = plt.subplots(1, 2, sharex=True)
    fig2.suptitle('Induction: %i' % context.induction)
    axes2[0].plot(context.binned_x, ramp_snapshots[0], c='k', label='Before')
    axes2[0].set_ylabel('Ramp amplitude (mV)')
    axes2[0].set_xlabel('Location (cm)')
    axes2[1].set_ylabel('Change in synaptic weight')
    axes2[1].set_xlabel('Location (cm)')
    for i in range(1, len(ramp_snapshots)):
        current_ramp = ramp_snapshots[i]
        current_delta_weights = np.subtract(delta_weights_snapshots[i], delta_weights_snapshots[i - 1])
        axes2[0].plot(context.binned_x, current_ramp)
        axes2[1].plot(context.peak_locs, current_delta_weights, label='Induction lap: %i' % i)
    axes2[1].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
    clean_axes(axes2)
    fig2.tight_layout()
    fig2.show()

    # New figures

    fig, axes = plt.subplots(2, 3, figsize=(12., 6.5))
    xmax = max(5000., local_signal_filter_t[-1], global_filter_t[-1]) / 1000.
    xmax = math.ceil(xmax)
    axes[0][0].plot(local_signal_filter_t / 1000., local_signal_filter / np.max(local_signal_filter), color='gray',
                    label='Synaptic\neligibility signal')
    axes[0][0].plot(global_filter_t / 1000., global_filter / np.max(global_filter), color='k',
                    label='Dendritic\ngating signal')
    axes[0][0].set_xlabel('Time (s)')
    axes[0][0].set_ylabel('Normalized amplitude')
    axes[0][0].set_xlim(-0.5, xmax)
    axes[0][0].set_title('Plasticity signal kinetics', fontsize=mpl.rcParams['font.size'], pad=10.)
    axes[0][0].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, fontsize=mpl.rcParams['font.size'])

    axes[0][1].set_xlabel('Normalized eligibility signal')
    axes[0][1].set_ylabel('Normalized rate')
    axes[0][1].set_title('Sigmoidal q$_{+}$, sigmoidal q$_{-}$', fontsize=mpl.rcParams['font.size'], pad=10.)
    axes[0][1].plot(signal_xrange, pot_rate(signal_xrange), c='r', label='q$_{+}$ (Potentiation)')
    axes[0][1].plot(signal_xrange, dep_rate(signal_xrange) * dep_scale, c='c', label='q$_{-}$ (Depression)')
    axes[0][1].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1, fontsize=mpl.rcParams['font.size'])

    cmap = cm.jet
    for w in np.linspace(0., 1., 10):
        net_delta_weight = pot_rate(signal_xrange) * (1. - w) - dep_rate(signal_xrange) * dep_scale * w
        axes[0][2].plot(signal_xrange, net_delta_weight, c=cmap(w))
    axes[0][2].axhline(y=0., linestyle='--', c='grey')
    axes[0][2].set_xlabel('Normalized eligibility signal')
    axes[0][2].set_ylabel('Normalized rate')
    axes[0][2].set_title('Net rate of change in synaptic weight', fontsize=mpl.rcParams['font.size'], pad=10.)
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0][2])
    cbar.set_label('Initial synaptic weight\n(normalized)', rotation=270., labelpad=25.)

    bar_loc = max(10., np.max(model_ramp) + 1., np.max(target_ramp) + 1.) * 0.95
    delta_weights = np.subtract(final_weights, initial_weights)
    peak_weight = np.max(np.abs(delta_weights))
    axes[1][2].plot(context.peak_locs, delta_weights, c='k')
    axes[1][2].axhline(y=0., linestyle='--', c='grey')
    axes[1][2].hlines(peak_weight * 1.05, xmin=context.mean_induction_start_loc, xmax=context.mean_induction_stop_loc)
    axes[1][1].plot(context.binned_x, initial_ramp, label='Before', c='k')
    axes[1][1].plot(context.binned_x, model_ramp, label='After', c='c')
    axes[1][1].hlines(bar_loc, xmin=context.mean_induction_start_loc, xmax=context.mean_induction_stop_loc)
    axes[1][2].set_ylabel('Change in\nsynaptic weight')
    axes[1][2].set_xlabel('Location (cm)')
    axes[1][1].set_ylabel('Ramp amplitude (mV)')
    axes[1][1].set_xlabel('Location (cm)')
    axes[1][1].set_xticks(np.arange(0., context.track_length, 45.))
    axes[1][2].set_xticks(np.arange(0., context.track_length, 45.))
    axes[1][1].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
    axes[1][1].set_ylim([min(-1., np.min(model_ramp) - 1., np.min(target_ramp) - 1.),
                         max(10., np.max(model_ramp) + 1., np.max(target_ramp) + 1.)])
    axes[1][2].set_ylim([-peak_weight, peak_weight * 1.1])
    axes[1][1].set_title('Model fit', fontsize=mpl.rcParams['font.size'], pad=10.)

    axes[1][0].plot(context.binned_x, initial_target_ramp, label='Before', c='k')
    axes[1][0].plot(context.binned_x, target_ramp, label='After', c='c')
    axes[1][0].hlines(bar_loc, xmin=context.mean_induction_start_loc, xmax=context.mean_induction_stop_loc)
    axes[1][0].set_ylabel('Ramp amplitude (mV)')
    axes[1][0].set_xlabel('Location (cm)')
    axes[1][0].set_xticks(np.arange(0., context.track_length, 45.))
    axes[1][0].legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
    axes[1][0].set_ylim([min(-1., np.min(model_ramp) - 1., np.min(target_ramp) - 1.),
                         max(10., np.max(model_ramp) + 1., np.max(target_ramp) + 1.)])
    axes[1][0].set_title('Target (synthetic data)', fontsize=mpl.rcParams['font.size'], pad=10.)

    clean_axes(axes)
    fig.suptitle('Hybrid weight-dependent model; synthetic data, induction: %i' % 2,
                 fontsize=mpl.rcParams['font.size'], x=0.02, ha='left')
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.65, top=0.9)
    fig.show()
    """
    context.update(locals())


def get_args_static_model_ramp():
    """
    A nested map operation is required to compute model_ramp features. The arguments to be mapped are the same
    (static) for each set of parameters.
    :param x: array
    :return: list of list
    """
    # return [[1, 1, 1, 2, 2], ['control', 'depo', 'hyper', 'control', 'hyper']]
    return [[1, 1, 1, 2], ['control', 'depo', 'hyper', 'control']]


def get_args_static_model_ramp_include_hyper2():
    """
    A nested map operation is required to compute model_ramp features. The arguments to be mapped are the same
    (static) for each set of parameters.
    :param x: array
    :return: list of list
    """
    return [[1, 1, 1, 2, 2], ['control', 'depo', 'hyper', 'control', 'hyper']]


def compute_features_model_ramp(x, induction=None, condition=None, model_id=None, export=False, plot=False):
    """

    :param x: array
    :param induction: str
    :param condition: str
    :param model_id: int
    :param export: bool
    :param plot: bool
    :return: dict
    """
    load_data(induction, condition)
    update_source_contexts(x, context)
    start_time = time.time()
    if context.disp:
        print('Process: %i: computing model_ramp features for induction: %s, condition: %s with x: %s' % \
              (os.getpid(), induction, condition, ', '.join('%.3E' % i for i in x)))
        sys.stdout.flush()
    result = calculate_model_ramp(export=export, plot=plot, model_id=model_id)
    if context.disp:
        print('Process: %i: computing model_ramp features for induction: %s, condition: %s took %.1f s' % \
              (os.getpid(), induction, condition, time.time() - start_time))
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
    groups = ['induction1', 'induction1_depo', 'induction1_hyper', 'induction2', 'induction2_hyper']
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
            for condition in this_result_dict[induction]:
                group = 'induction' + str(induction)
                if condition in ['depo', 'hyper']:
                    group += '_%s' % condition
                for feature_name in grouped_feature_names:
                    key = group + '_' + feature_name
                    if key not in features:
                        features[key] = []
                    features[key].append(this_result_dict[induction][condition][feature_name])
                for feature_name in feature_names:
                    if feature_name in this_result_dict[induction][condition]:
                        if feature_name not in features:
                            features[feature_name] = []
                        features[feature_name].append(this_result_dict[induction][condition][feature_name])

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
    groups = ['induction1', 'induction1_depo', 'induction1_hyper', 'induction2', 'induction2_hyper']
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
    args = interface.execute(globals()[context.stages[0]['get_args_static']])
    group_size = len(args[0])
    sequences = [[x] * group_size] + args + [[model_id] * group_size] + [[context.export] * group_size] + \
                [[plot] * group_size]
    primitives = interface.map(compute_features_model_ramp, *sequences)
    new_features = interface.execute(filter_features_model_ramp, primitives, features, model_id, context.export)
    features.update(new_features)

    return features


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_biBTSP_synthetic_hybrid_norm_config.yaml')
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
    python -i optimize_biBTSP_synthetic_hybrid_norm.py --plot --framework=serial --interactive

    To execute using MPI parallelism with 1 controller process and N - 1 worker processes:
    mpirun -n N python -i -m mpi4py.futures optimize_biBTSP_synthetic_hybrid_norm.py --plot --framework=mpi --interactive

    To optimize the models by running many instances in parallel:
    mpirun -n N python -m mpi4py.futures -m nested.optimize --config-file-path=$PATH_TO_CONFIG_FILE --disp --export \
        --framework=mpi --pop-size=200 --path-length=3 --max-iter=50

    To plot results previously exported to a file on a single process:
    python -i optimize_biBTSP_synthetic_hybrid_norm.py --plot-summary-figure --model-file-path=$PATH_TO_MODEL_FILE \
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
    elif not debug:
        model_id = 0
        if 'model_key' in context() and context.model_key is not None:
            model_label = context.model_key
        else:
            model_label = 'test'

        features = get_features_interactive(context.interface, context.x0_array, model_id=model_id, plot=plot)
        features, objectives = context.interface.execute(get_objectives, features, model_id, context.export)
        if export:
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

        if plot:
            context.interface.apply(plt.show)

    if context.interactive:
        context.update(locals())
    else:
        context.interface.stop()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
