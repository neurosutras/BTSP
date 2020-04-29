"""

"""
from nested.utils import *
from scipy.optimize import minimize


class StateMachine(object):
    """

    """

    def __init__(self, ti=0., dt=1., states=None, rates=None):
        """

        :param ti: float
        :param dt: float (milliseconds)
        :param states: dict
        :param rates: dict (proportion of total/second)
        """
        self.dt = dt
        self.init_states = dict()
        self.states = dict()
        self.states_history = dict()
        self.rates = dict()
        if states is not None:
            self.update_states(states)
        if rates is not None:
            self.update_rates(rates)  # {'A': {'B': constant or iterable}}
        self.ti = ti
        self.t = self.ti
        self.t_history = np.array([self.t])
        self.i = 0

    def reset(self):
        """

        :param ti:
        :return:
        """
        self.t = self.ti
        self.i = 0
        self.t_history = np.array([self.t])
        self.states = dict(self.init_states)
        for s0 in self.states:
            self.states_history[s0] = np.array([self.states[s0]])

    def get_current_rates(self):
        """

        :return: dict
        """
        current = {}
        for s0 in self.rates:
            if s0 not in current:
                current[s0] = {}
            for s1 in self.rates[s0]:
                r = self.rates[s0][s1]
                if isinstance(r, Iterable):
                    if len(r) - 1 < self.i:
                        raise Exception('StateMachine: Insufficient array length for non-stationary rate: %s to %s ' %
                                        (s0, s1))
                    this_r = r[self.i]
                else:
                    this_r = r
                current[s0][s1] = this_r
        return current

    def update_transition(self, s0, s1, r):
        """

        :param s0: str
        :param s1: str
        :param r: float or array
        """
        if s0 not in self.states:
            raise Exception('StateMachine: Cannot update transition from invalid state: %s' % s0)
        if s1 not in self.states:
            raise Exception('StateMachine: Cannot update transition to invalid state: %s' % s1)
        if s0 not in self.rates:
            self.rates[s0] = {}
        self.rates[s0][s1] = r

    def update_rates(self, rates):
        """

        :param rates: dict
        """
        for s0 in rates:
            for s1, r in viewitems(rates[s0]):
                self.update_transition(s0, s1, r)

    def update_states(self, states):
        """

        :param states: dict
        """
        for s, v in viewitems(states):
            self.init_states[s] = v
            self.states[s] = v
            self.states_history[s] = np.array([v])

    def get_out_rate(self, state):
        """

        :param state: str
        :return: float
        """
        if state not in self.states:
            raise Exception('StateMachine: Invalid state: %s' % state)
        if state not in self.rates:
            return 0.
        out_rate = 0.
        for s1 in self.rates[state]:
            r = self.rates[state][s1]
            if isinstance(r, Iterable):
                if len(r) - 1 < self.i:
                    raise Exception('StateMachine: Insufficient array length for non-stationary rate: %s to %s ' %
                                    (state, s1))
                this_r = r[self.i]
            else:
                this_r = r
            out_rate += this_r
        return out_rate

    def step(self, n=1):
        """
        Need to think about how to deal with weights that vary from 0.8 to unknown amount....
        :param n: int
        """
        for i in range(n):
            next_states = dict(self.states)
            for s0 in self.rates:
                factor = 1.
                if self.states[s0] > 0.:
                    total_out = self.get_out_rate(s0) * self.dt / 1000. * self.states[s0]
                    if total_out > self.states[s0]:
                        factor = self.states[s0] / total_out
                for s1 in self.rates[s0]:
                    r = self.rates[s0][s1]
                    if isinstance(r, Iterable):
                        if len(r) - 1 < self.i:
                            raise Exception('StateMachine: Insufficient array length for non-stationary rate: %s to '
                                            '%s ' % (s0, s1))
                        this_r = r[self.i]
                    else:
                        this_r = r
                    this_delta = this_r * self.dt / 1000. * factor * self.states[s0]
                    next_states[s0] -= this_delta
                    next_states[s1] += this_delta
            self.states = dict(next_states)
            for s0 in self.states:
                self.states_history[s0] = np.append(self.states_history[s0], self.states[s0])
            self.i += 1
            self.t += self.dt
            self.t_history = np.append(self.t_history, self.t)

    def run(self):
        """

        """
        self.reset()
        min_steps = None
        for s0 in self.rates:
            for s1 in self.rates[s0]:
                r = self.rates[s0][s1]
                if isinstance(r, Iterable):
                    if min_steps is None:
                        min_steps = len(r)
                    else:
                        min_steps = min(min_steps, len(r))
        if min_steps is None:
            raise Exception('StateMachine: Use step method to specify number of steps for stationary process.')
        self.step(min_steps)

    def plot(self, states=None):
        """

        :param states:
        """
        if states is None:
            states = self.states.keys()
        elif not isinstance(states, Iterable):
            states = [states]
        fig, axes = plt.subplots(1)
        for state in states:
            if state in self.states:
                axes.plot(self.t_history, self.states_history[state], label=state)
            else:
                print('StateMachine: Not including invalid state: %s' % state)
        axes.set_xlabel('Time (ms)')
        axes.set_ylabel('Occupancy')
        axes.legend(loc='best', frameon=False, framealpha=0.5)
        clean_axes(axes)
        plt.show()
        plt.close()


def generate_spatial_rate_maps(x, n=200, peak_rate=1., field_width=90., track_length=187.):
    """
    Return a list of spatial rate maps with peak locations that span the track. Return firing rate vs. location
    computed at the resolution of the provided x array.
    :param x: array
    :param n: int
    :param peak_rate: float
    :param field_width: float
    :param track_length: float
    :return: list of array, array
    """
    gauss_sigma = field_width / 3. / np.sqrt(2.)  # contains 99.7% gaussian area
    d_peak_locs = track_length / float(n)
    peak_locs = np.arange(d_peak_locs / 2., track_length, d_peak_locs)
    spatial_rate_maps = []
    extended_x = np.concatenate([x - track_length, x, x + track_length])
    for peak_loc in peak_locs:
        gauss_force = peak_rate * np.exp(-((extended_x - peak_loc) / gauss_sigma) ** 2.)
        gauss_force = wrap_around_and_compress(gauss_force, x)
        spatial_rate_maps.append(gauss_force)
    return spatial_rate_maps, peak_locs


def get_complete_rate_maps(input_rate_maps, input_x, position, complete_run_vel_gate):
    """
    :param input_rate_maps: array
    :param input_x: array (x resolution of input)
    :param position: nested dict of array
    :param complete_run_vel_gate: array
    :return: list of array
    """
    complete_rate_maps = []
    for j in range(len(input_rate_maps)):
        this_complete_rate_map = np.array([])
        for group in ['pre', 'induction', 'post']:
            for i, this_position in enumerate(position[group]):
                this_rate_map = np.interp(this_position, input_x, input_rate_maps[j])
                this_complete_rate_map = np.append(this_complete_rate_map, this_rate_map)
        if len(this_complete_rate_map) != len(complete_run_vel_gate):
            print('get_complete_rate_maps: mismatched array length')
        this_complete_rate_map = np.multiply(this_complete_rate_map, complete_run_vel_gate)
        complete_rate_maps.append(this_complete_rate_map)
    return complete_rate_maps


def get_complete_ramp(current_ramp, input_x, position, complete_run_vel_gate, induction_gate, peak_ramp_amp):
    """

    :param current_ramp: array
    :param input_x: array (x resolution of input)
    :param position: nested dict of array
    :param complete_run_vel_gate: array
    :param induction_gate: array
    :param peak_ramp_amp: float
    :return: array
    """
    complete_ramp = np.array([])
    for group in ['pre', 'induction', 'post']:
        for i, this_position in enumerate(position[group]):
            this_lap_vm = np.interp(this_position, input_x, current_ramp)
            complete_ramp = np.append(complete_ramp, this_lap_vm)
    complete_ramp = np.multiply(complete_ramp, complete_run_vel_gate)
    complete_ramp[np.where(induction_gate == 1.)[0]] = peak_ramp_amp
    if len(complete_ramp) != len(complete_run_vel_gate):
        print('get_complete_ramp: mismatched array length')

    return complete_ramp


def get_exp_rise_decay_filter(rise, decay, max_time_scale, dt):
    """
    :param rise: float
    :param decay: float
    :param max_time_scale: float
    :param dt: float
    :return: array, array
    """
    filter_t = np.arange(0., 6. * max_time_scale, dt)
    filter = np.exp(-filter_t / decay) - np.exp(-filter_t / rise)
    peak_index = np.where(filter == np.max(filter))[0][0]
    decay_indexes = np.where(filter[peak_index:] < 0.001 * np.max(filter))[0]
    if np.any(decay_indexes):
        filter = filter[:peak_index + decay_indexes[0]]
    filter /= np.sum(filter)
    filter_t = filter_t[:len(filter)]
    return filter_t, filter


def get_exp_decay_filter(decay, max_time_scale, dt):
    """
    :param decay: float
    :param max_time_scale: float
    :param dt: float
    :return: array, array
    """
    filter_t = np.arange(0., 6. * max_time_scale, dt)
    filter = np.exp(-filter_t / decay)
    decay_indexes = np.where(filter < 0.001 * np.max(filter))[0]
    if np.any(decay_indexes):
        filter = filter[:decay_indexes[0]]
    filter /= np.sum(filter)
    filter_t = filter_t[:len(filter)]
    return filter_t, filter


def get_dual_signal_filters(local_signal_rise, local_signal_decay, global_signal_rise, global_signal_decay, dt,
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
    local_signal_filter_t, local_signal_filter = \
        get_exp_rise_decay_filter(local_signal_rise, local_signal_decay, max_time_scale, dt)
    global_filter_t, global_filter = \
        get_exp_rise_decay_filter(global_signal_rise, global_signal_decay, max_time_scale, dt)
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


def get_dual_exp_decay_signal_filters(local_signal_decay, global_signal_decay, dt, plot=False):
    """
    :param local_signal_decay: float
    :param global_signal_decay: float
    :param dt: float
    :param plot: bool
    :return: array, array
    """
    max_time_scale = max(local_signal_decay, global_signal_decay)
    local_signal_filter_t, local_signal_filter = \
        get_exp_decay_filter(local_signal_decay, max_time_scale, dt)
    global_filter_t, global_filter = \
        get_exp_decay_filter(global_signal_decay, max_time_scale, dt)
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


def get_triple_signal_filters(pot_signal_rise, pot_signal_decay, depot_signal_rise, depot_signal_decay,
                              global_signal_rise, global_signal_decay, dt, plot=False):
    """
    :param pot_signal_rise: float
    :param pot_signal_decay: float
    :param depot_signal_rise: float
    :param depot_signal_decay: float
    :param global_signal_rise: float
    :param global_signal_decay: float
    :param dt: float
    :param plot: bool
    :return: array, array
    """
    max_time_scale = max(pot_signal_rise + pot_signal_decay, depot_signal_rise + depot_signal_decay,
                         global_signal_rise + global_signal_decay)
    pot_signal_filter_t, pot_signal_filter = \
        get_exp_rise_decay_filter(pot_signal_rise, pot_signal_decay, max_time_scale, dt)
    depot_signal_filter_t, depot_signal_filter = \
        get_exp_rise_decay_filter(depot_signal_rise, depot_signal_decay, max_time_scale, dt)
    global_filter_t, global_filter = \
        get_exp_rise_decay_filter(global_signal_rise, global_signal_decay, max_time_scale, dt)
    if plot:
        fig, axes = plt.subplots(1)
        axes.plot(pot_signal_filter_t / 1000., pot_signal_filter / np.max(pot_signal_filter), color='c',
                  label='Potentiation eligibility signal filter')
        axes.plot(depot_signal_filter_t / 1000., depot_signal_filter / np.max(depot_signal_filter), color='r',
                  label='De-potentiation eligibility signal filter')
        axes.plot(global_filter_t / 1000., global_filter / np.max(global_filter), color='k',
                  label='Global plasticity signal filter')
        axes.set_xlabel('Time (s)')
        axes.set_ylabel('Normalized filter amplitude')
        axes.set_title('Plasticity signal filters')
        axes.legend(loc='best', frameon=False, framealpha=0.5, handlelength=1)
        axes.set_xlim(-0.5, max(5000., max_time_scale * 6.) / 1000.)
        clean_axes(axes)
        fig.tight_layout()
        fig.show()
    return pot_signal_filter_t, pot_signal_filter, depot_signal_filter_t, depot_signal_filter, \
           global_filter_t, global_filter


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


def get_local_signal_population(local_filter, rate_maps, dt):
    """

    :param local_filter: array
    :param rate_maps: list of array
    :param dt: float
    :return:
    """
    local_signals = []
    for rate_map in rate_maps:
        local_signals.append(get_local_signal(rate_map, local_filter, dt))
    return local_signals


def get_voltage_dependent_eligibility_signal_population(local_filter, normalized_ramp, phi, rate_maps, dt):
    """

    :param local_filter: array
    :param normalized_ramp: array
    :param phi: lambda
    :param rate_maps: list of array
    :param dt: float
    :return: list of array
    """
    local_signals = []
    this_phi = phi(normalized_ramp)

    for rate_map in rate_maps:
        local_signals.append(get_local_signal(np.multiply(rate_map, this_phi), local_filter, dt))

    return local_signals


def weights_path_distance_exceeds_threshold(weights_snapshots, threshold=2.):
    """
    If changes in weights across laps are monotonic, the path distance is equal to the euclidean distance. However, if
    weight changes change sign across laps, the path distance can increase. This method checks if the weight changes
    across the population of inputs exceed a threshold fold increase of path distance relative to euclidean distance.
    :param weights_snapshots: list of array
    :return: bool
    """
    weights_snapshots = np.array(weights_snapshots)
    weights_diff = np.diff(weights_snapshots, axis=0)
    path_distance = np.sum(np.abs(weights_diff), axis=0)
    path_distance_pop_sum = np.sum(path_distance)
    euc_distance = np.abs(np.sum(weights_diff, axis=0))
    euc_distance_pop_sum = np.sum(euc_distance)

    return path_distance_pop_sum > threshold * euc_distance_pop_sum


def sigmoid_segment(slope, th, xlim=None, ylim=None):
    """
    Transform a sigmoid to intersect x and y range limits.
    :param slope: float
    :param th: float
    :param xlim: pair of float
    :param ylim: pair of float
    :return: callable
    """
    if xlim is None:
        xlim = (0., 1.)
    if ylim is None:
        ylim = (0., 1.)
    if th < xlim[0] or th > xlim[1]:
        raise ValueError('sigmoid_segment: th: %.2E is out of range for xlim: %s' % (th, str(xlim)))
    y = lambda x: 1. / (1. + np.exp(-slope * (x - th)))
    target_amp = ylim[1] - ylim[0]
    y0 = y(xlim[0])
    y1 = y(xlim[1])
    current_amp = y1 - y0
    return lambda x: (target_amp / current_amp) * (1. / (1. + np.exp(-slope * (x - th))) - y0) + ylim[0]


def scaled_single_sigmoid(th, peak, x, ylim=None):
    """
    Transform a sigmoid to intersect x and y range limits.
    :param th: float
    :param peak: float
    :param x: array
    :param ylim: pair of float
    :return: callable
    """
    if ylim is None:
        ylim = (0., 1.)
    if th < x[0] or th > x[-1]:
        raise ValueError('scaled_single_sigmoid: th: %.2E is out of range for xlim: [%.2E, %.2E]' % (th, x[0], x[-1]))
    if peak == th:
        raise ValueError('scaled_single_sigmoid: peak and th: %.2E cannot be equal' % th)
    slope = 2. / (peak - th)
    y = lambda x: 1. / (1. + np.exp(-slope * (x - th)))
    start_val = y(x[0])
    end_val = y(x[-1])
    amp = end_val - start_val
    target_amp = ylim[1] - ylim[0]
    return lambda xi: (target_amp / amp) * (1. / (1. + np.exp(-slope * (xi - th))) - start_val) + ylim[0]


def scaled_double_sigmoid_orig(th1, peak1, th2, peak2, x, y_start=0., y_peak=1., y_end=0.):
    """
    Transform a double sigmoid to intersect x and y range limits. A double sigmoid is a product of an ascending and a
    descending sigmoid.
    :param th1: float
    :param peak1: float
    :param th2: float
    :param peak2: float
    :param x: array
    :param y_start: float
    :param y_peak: float
    :param y_end: float
    :return: callable
    """
    if th1 < x[0] or th1 > x[-1]:
        raise ValueError('scaled_double_sigmoid: th1: %.2E is out of range for xlim: [%.2E, %.2E]' % (th1, x[0], x[-1]))
    if th2 < x[0] or th2 > x[-1]:
        raise ValueError('scaled_double_sigmoid: th2: %.2E is out of range for xlim: [%.2E, %.2E]' % (th2, x[0], x[-1]))
    if peak1 == th1:
        raise ValueError('scaled_double_sigmoid: peak1 and th1: %.2E cannot be equal' % th1)
    if peak2 == th2:
        raise ValueError('scaled_double_sigmoid: peak2 and th2: %.2E cannot be equal' % th2)
    if peak2 / (x[-1] - x[0]) > 0.95 and th2 / (x[-1] - x[0]) > 0.95:
        return scaled_single_sigmoid(peak1, th1, x, [y_start, y_peak])
    slope1 = 2. / (peak1 - th1)
    slope2 = 2. / (peak2 - th2)
    y = lambda x: (1. / (1. + np.exp(-slope1 * (x - th1)))) * (1. / (1. + np.exp(-slope2 * (x - th2))))
    y_array = np.vectorize(y)
    peak_index = np.argmax(y_array(x))
    x_peak = x[peak_index]
    max_val = y(x_peak)
    start_val = y(x[0])
    end_val = y(x[-1])
    amp1 = max_val - start_val
    amp2 = max_val - end_val
    target_amp1 = y_peak - y_start
    target_amp2 = y_peak - y_end

    return lambda xi: (target_amp1 / amp1) * (y(xi) - start_val) + y_start if xi <= x_peak else \
        (target_amp2 / amp2) * (y(xi) - end_val) + y_end


def scaled_double_sigmoid(th1, peak1, th2, peak2, x, y_start=0., y_peak=1., y_end=0.):
    """
    Transform a double sigmoid to intersect x and y range limits. A double sigmoid is a product of an ascending and a
    descending sigmoid.
    :param th1: float
    :param peak1: float
    :param th2: float
    :param peak2: float
    :param x: array
    :param y_start: float
    :param y_peak: float
    :param y_end: float
    :return: callable
    """
    if th1 < x[0] or th1 > x[-1]:
        raise ValueError('scaled_double_sigmoid: th1: %.2E is out of range for xlim: [%.2E, %.2E]' % (th1, x[0], x[-1]))
    if th2 < x[0] or th2 > x[-1]:
        raise ValueError('scaled_double_sigmoid: th2: %.2E is out of range for xlim: [%.2E, %.2E]' % (th2, x[0], x[-1]))
    if peak1 == th1:
        raise ValueError('scaled_double_sigmoid: peak1 and th1: %.2E cannot be equal' % th1)
    if peak2 == th2:
        raise ValueError('scaled_double_sigmoid: peak2 and th2: %.2E cannot be equal' % th2)
    slope1 = 2. / (peak1 - th1)
    slope2 = 2. / (peak2 - th2)
    y = lambda xi: (1. / (1. + np.exp(-slope1 * (xi - th1)))) * (1. / (1. + np.exp(-slope2 * (xi - th2))))
    y_array = np.vectorize(y)
    peak_index = np.argmax(y_array(x))
    x_peak = x[peak_index]
    max_val = y(x_peak)
    start_val = y(x[0])
    end_val = y(x[-1])
    amp1 = max_val - start_val
    amp2 = max_val - end_val
    if amp1 == 0. or amp2 == 0.:
        raise ValueError('scaled_double_sigmoid: parameters out of range for scaled double sigmoid')
    target_amp1 = y_peak - y_start
    target_amp2 = y_peak - y_end

    return lambda xi: (target_amp1 / amp1) * (y(xi) - start_val) + y_start if xi <= x_peak else \
        (target_amp2 / amp2) * (y(xi) - end_val) + y_end


def visualize_scaled_double_sigmoid(th1, peak1, th2, peak2, x, y_start=0., y_peak=1., y_end=0.):
    """
    Transform a double sigmoid to intersect x and y range limits. A double sigmoid is a product of an ascending and a
    descending sigmoid.
    :param th1: float
    :param peak1: float
    :param th2: float
    :param peak2: float
    :param x: array
    :param y_start: float
    :param y_peak: float
    :param y_end: float
    :return: callable
    """
    if th1 < x[0] or th1 > x[-1]:
        raise ValueError('scaled_double_sigmoid: th1: %.2E is out of range for xlim: [%.2E, %.2E]' % (th1, x[0], x[-1]))
    if th2 < x[0] or th2 > x[-1]:
        raise ValueError('scaled_double_sigmoid: th2: %.2E is out of range for xlim: [%.2E, %.2E]' % (th2, x[0], x[-1]))
    if peak1 == th1:
        raise ValueError('scaled_double_sigmoid: peak1 and th1: %.2E cannot be equal' % th1)
    if peak2 == th2:
        raise ValueError('scaled_double_sigmoid: peak2 and th2: %.2E cannot be equal' % th2)
    slope1 = 2. / (peak1 - th1)
    slope2 = 2. / (peak2 - th2)
    y1 = (1. / (1. + np.exp(-slope1 * (x - th1))))
    y2 = (1. / (1. + np.exp(-slope2 * (x - th2))))
    y = lambda x: (1. / (1. + np.exp(-slope1 * (x - th1)))) * (1. / (1. + np.exp(-slope2 * (x - th2))))
    y_array = np.vectorize(y)
    fig, axes = plt.subplots(2)
    axes[0].plot(x, y1)
    axes[0].plot(x, y2)
    axes[1].plot(x, y_array(x))
    peak_index = np.argmax(y_array(x))
    x_peak = x[peak_index]
    max_val = y(x_peak)
    start_val = y(x[0])
    end_val = y(x[-1])
    amp1 = max_val - start_val
    amp2 = max_val - end_val
    if amp1 == 0. or amp2 == 0.:
        raise ValueError('scaled_double_sigmoid: parameters out of range for scaled double sigmoid')
    target_amp1 = y_peak - y_start
    target_amp2 = y_peak - y_end

    target = np.vectorize(lambda xi: (target_amp1 / amp1) * (y(xi) - start_val) + y_start if xi <= x_peak else
    (target_amp2 / amp2) * (y(xi) - end_val) + y_end)
    axes[1].plot(x, target(x))
    fig.show()


def subtract_baseline(waveform, baseline=None):
    """

    :param waveform: array
    :param baseline: float
    :return: array
    """
    new_waveform = np.array(waveform)
    if baseline is None:
        baseline = np.mean(new_waveform[np.where(new_waveform <= np.percentile(new_waveform, 10.))[0]])
    new_waveform -= baseline
    return new_waveform, baseline


def wrap_around_and_compress(waveform, interp_x):
    """

    :param waveform: array of len(3 * interp_x)
    :param interp_x: array
    :return: array of len(interp_x)
    """
    before = np.array(waveform[:len(interp_x)])
    after = np.array(waveform[2 * len(interp_x):])
    within = np.array(waveform[len(interp_x):2 * len(interp_x)])
    waveform = within[:len(interp_x)] + before[:len(interp_x)] + after[:len(interp_x)]
    return waveform


def get_indexes_from_ramp_bounds_with_wrap(x, start, peak, end, min):
    """

    :param x: array
    :param start: float
    :param peak: float
    :param end: float
    :param min: float
    :return: tuple of float: (start_index, peak_index, end_index, min_index)
    """
    peak_index = np.where(x >= peak)[0]
    if np.any(peak_index):
        peak_index = peak_index[0]
    else:
        peak_index = len(x) - 1
    min_index = np.where(x >= min)[0]
    if np.any(min_index):
        min_index = min_index[0]
    else:
        min_index = len(x) - 1
    if start < peak:
        if start < x[0]:
            start_index = 0
        else:
            start_index = np.where(x[:peak_index] <= start)[0][-1]
    else:
        start_index = peak_index + np.where(x[peak_index:] <= start)[0]
        if np.any(start_index):
            start_index = start_index[-1]
        else:
            start_index = len(x) - 1
    if end < peak:
        end_index = np.where(x > end)[0][0]
    else:
        end_index = peak_index + np.where(x[peak_index:] > end)[0]
        if np.any(end_index):
            end_index = end_index[0]
        else:
            end_index = len(x) - 1
    return start_index, peak_index, end_index, min_index


def get_model_ramp(delta_weights, ramp_x, input_x, input_rate_maps, ramp_scaling_factor, allow_offset=False,
                   impose_offset=None):
    """

    :param delta_weights: array
    :param ramp_x: array (x resolution of output ramp)
    :param input_x: array (x resolution of input_rate_maps)
    :param input_rate_maps: list of array
    :param ramp_scaling_factor: float
    :param allow_offset: bool (allow special case where baseline Vm before 1st induction is unknown)
    :param impose_offset: float (impose Vm offset from 1st induction on 2nd induction)
    :return: tuple: (array, float)
    """
    model_ramp = np.multiply(delta_weights.dot(np.array(input_rate_maps)), ramp_scaling_factor)
    if len(model_ramp) != len(ramp_x):
        model_ramp = np.interp(ramp_x, input_x, model_ramp)

    if impose_offset is not None:
        ramp_offset = impose_offset
        model_ramp -= impose_offset
    elif allow_offset:
        model_ramp, ramp_offset = subtract_baseline(model_ramp)
    else:
        ramp_offset = 0.

    return model_ramp, ramp_offset


def calculate_ramp_features(ramp, induction_loc, binned_x, interp_x, track_length, offset=False, smooth=False):
    """

    :param ramp: array
    :param induction_loc: float
    :param binned_x: array
    :param interp_x: array
    :param track_length: float
    :param offset: bool
    :param smooth: bool
    :return tuple of float
    """
    default_interp_x = interp_x
    extended_binned_x = np.concatenate([binned_x - track_length, binned_x, binned_x + track_length])
    if smooth:
        local_ramp = signal.savgol_filter(ramp, 21, 3, mode='wrap')
    else:
        local_ramp = np.array(ramp)
    extended_binned_ramp = np.concatenate([local_ramp] * 3)
    extended_interp_x = np.concatenate([default_interp_x - track_length, default_interp_x,
                                        default_interp_x + track_length])
    extended_ramp = np.interp(extended_interp_x, extended_binned_x, extended_binned_ramp)
    interp_ramp = extended_ramp[len(default_interp_x):2 * len(default_interp_x)]
    baseline_indexes = np.where(interp_ramp <= np.percentile(interp_ramp, 10.))[0]
    if len(baseline_indexes) > 0:
        baseline = np.mean(interp_ramp[baseline_indexes])
    else:
        baseline = np.min(interp_ramp)
    if offset:
        interp_ramp -= baseline
        extended_ramp -= baseline
    peak_index = np.where(interp_ramp == np.max(interp_ramp))[0][0] + len(interp_ramp)
    peak_val = extended_ramp[peak_index]
    peak_x = extended_interp_x[peak_index]
    start_index = np.where(extended_ramp[:peak_index] <=
                           0.15 * (peak_val - baseline) + baseline)[0]
    if len(start_index) > 0:
        start_index = start_index[-1]
    else:
        start_index = np.where(interp_ramp == np.min(interp_ramp))[0][0] + len(interp_ramp)
    end_index = peak_index + np.where(extended_ramp[peak_index:] <= 0.15 *
                                      (peak_val - baseline) + baseline)[0]
    if len(end_index) > 0:
        end_index = end_index[0]
    else:
        end_index = np.where(interp_ramp == np.min(interp_ramp))[0][0] + len(interp_ramp)
    start_loc = float(start_index % len(default_interp_x)) / float(len(default_interp_x)) * track_length
    end_loc = float(end_index % len(default_interp_x)) / float(len(default_interp_x)) * track_length
    peak_loc = float(peak_index % len(default_interp_x)) / float(len(default_interp_x)) * track_length
    min_index = np.where(interp_ramp == np.min(interp_ramp))[0][0] + len(interp_ramp)
    min_val = extended_ramp[min_index]
    min_loc = float(min_index % len(default_interp_x)) / float(len(default_interp_x)) * track_length
    peak_shift = peak_x - induction_loc
    if peak_shift > track_length / 2.:
        peak_shift = -(track_length - peak_shift)
    elif peak_shift < -track_length / 2.:
        peak_shift += track_length
    if peak_val == 0.:
        scaled_ramp = np.array(local_ramp)
    else:
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


def get_local_peak_shift(ramp, induction_loc, binned_x, interp_x, track_length, tolerance=30.):
    """
    If there are multiple local peaks of similar amplitude, return the minimum distance to the target_loc.
    :param ramp: array
    :param induction_loc: float
    :param binned_x: array
    :param interp_x: array
    :param track_length: float
    :param tolerance: float (cm)
    :return: float
    """
    default_interp_x = interp_x
    interp_ramp = np.interp(default_interp_x, binned_x, ramp)
    order = int((tolerance / track_length) * len(interp_ramp))

    peak_indexes = signal.argrelmax(interp_ramp, order=order, mode='wrap')[0]
    if not np.any(peak_indexes):
        peak_indexes = np.array([np.argmax(interp_ramp)])
    peak_locs = default_interp_x[peak_indexes]
    peak_shifts = []

    for peak_loc in peak_locs:
        peak_shift = peak_loc - induction_loc
        if peak_shift > track_length / 2.:
            peak_shift = -(track_length - peak_shift)
        elif peak_shift < -track_length / 2.:
            peak_shift += track_length
        peak_shifts.append(peak_shift)

    peak_shifts = np.array(peak_shifts)
    indexes = list(range(len(peak_shifts)))
    indexes.sort(key=lambda x: abs(peak_shifts[x]))

    return peak_locs[indexes][0], peak_shifts[indexes][0]


def get_residual_score(delta_weights, target_ramp, ramp_x, input_x, interp_x, input_rate_maps, ramp_scaling_factor,
                       induction_loc, track_length, target_range, bounds=None, allow_offset=False, impose_offset=None,
                       disp=False, full_output=False):
    """

    :param delta_weights: array
    :param target_ramp: array
    :param ramp_x: array (spatial resolution of ramp)
    :param input_x: array (spatial resolution of input_rate_maps)
    :param interp_x: array (spatial resolution for computing fine features)
    :param input_rate_maps: list of array
    :param ramp_scaling_factor: float
    :param induction_loc: float
    :param track_length: float
    :param target_range: dict
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
    if len(target_ramp) != len(input_x):
        exp_ramp = np.interp(input_x, ramp_x, target_ramp)
    else:
        exp_ramp = np.array(target_ramp)

    model_ramp, ramp_offset = get_model_ramp(delta_weights, ramp_x=ramp_x, input_x=input_x,
                                             input_rate_maps=input_rate_maps, ramp_scaling_factor=ramp_scaling_factor,
                                             allow_offset=allow_offset, impose_offset=impose_offset)

    Err = 0.
    if allow_offset:
        Err += (ramp_offset / target_range['ramp_offset']) ** 2.

    ramp_amp, ramp_width, peak_shift, ratio, start_loc, peak_loc, end_loc, min_val, min_loc = {}, {}, {}, {}, {}, {}, \
                                                                                              {}, {}, {}
    ramp_amp['target'], ramp_width['target'], peak_shift['target'], ratio['target'], start_loc['target'], \
    peak_loc['target'], end_loc['target'], min_val['target'], min_loc['target'] = \
        calculate_ramp_features(ramp=exp_ramp, induction_loc=induction_loc, binned_x=ramp_x, interp_x=interp_x,
                                track_length=track_length)

    ramp_amp['model'], ramp_width['model'], peak_shift['model'], ratio['model'], start_loc['model'], \
    peak_loc['model'], end_loc['model'], min_val['model'], min_loc['model'] = \
        calculate_ramp_features(ramp=model_ramp, induction_loc=induction_loc, binned_x=ramp_x, interp_x=interp_x,
                                track_length=track_length)

    if disp:
        print('exp: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, peak_loc: %.1f, ' \
              'end_loc: %.1f, min_val: %.1f, min_loc: %.1f' % \
              (ramp_amp['target'], ramp_width['target'], peak_shift['target'], ratio['target'], start_loc['target'],
               peak_loc['target'], end_loc['target'], min_val['target'], min_loc['target']))
        print('model: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, peak_loc: %.1f' \
              ', end_loc: %.1f, min_val: %.1f, min_loc: %.1f' % \
              (ramp_amp['model'], ramp_width['model'], peak_shift['model'], ratio['model'], start_loc['model'],
               peak_loc['model'], end_loc['model'], min_val['model'], min_loc['model']))
        sys.stdout.flush()

    start_index, peak_index, end_index, min_index = \
        get_indexes_from_ramp_bounds_with_wrap(ramp_x, start_loc['target'], peak_loc['target'], end_loc['target'],
                                               min_loc['target'])

    model_val_at_target_min_loc = model_ramp[min_index]
    Err += ((model_val_at_target_min_loc - min_val['target']) / target_range['delta_min_val']) ** 2.
    Err += ((min_val['model'] - min_val['target']) / target_range['delta_min_val']) ** 2.
    model_val_at_target_peak_loc = model_ramp[peak_index]
    Err += ((model_val_at_target_peak_loc - ramp_amp['target']) / target_range['delta_peak_val']) ** 2.

    for i in range(len(exp_ramp)):
        Err += ((exp_ramp[i] - model_ramp[i]) / target_range['residuals']) ** 2.
    # regularization
    for delta in np.diff(np.insert(delta_weights, 0, delta_weights[-1])):
        Err += (delta / target_range['weights_smoothness']) ** 2.

    if full_output:
        return model_ramp, delta_weights, ramp_offset, Err
    else:
        return Err


def get_adjusted_delta_weights_and_ramp_scaling_factor(delta_weights, input_rate_maps, target_peak_weight,
                                                       target_ramp_amp):
    """

    :param delta_weights: array
    :param input_rate_maps: list of array
    :param target_peak_weight: float
    :param target_ramp_amp: float
    :return: tuple: (array, float)
    """
    adjusted_delta_weights = np.multiply(delta_weights, target_peak_weight / np.max(delta_weights))
    input_matrix = np.array(input_rate_maps)
    model_ramp = adjusted_delta_weights.dot(input_matrix)
    ramp_scaling_factor = target_ramp_amp / np.max(model_ramp)

    return adjusted_delta_weights, ramp_scaling_factor


def get_initial_target_ramp_and_scaling_factor(ramp_x, input_x, interp_x, num_inputs, input_field_peak_rate,
                                               input_field_width, track_length, target_weights_width=108.,
                                               target_peak_delta_weight=1.5, target_peak_ramp_amp=6., plot=False,
                                               verbose=1):
    """

    :param ramp_x: array (spatial resolution of ramp)
    :param input_x: array (spatial resolution of input_rate_maps)
    :param interp_x: array (spatial resolution for computing fine features)
    :param num_inputs: int
    :param input_field_peak_rate: float
    :param input_field_width: float
    :param track_length: float
    :param target_weights_width: float  (legacy target ramp width is 108 cm (90 * 1.2),
                                        to produce a firing field of width 90)
    :param target_peak_delta_weight: float
    :param target_peak_ramp_amp: float (mV)
    :param plot: bool
    :param verbose: bool
    :return: tuple (array, float)
    """
    input_rate_maps, peak_locs = \
        generate_spatial_rate_maps(input_x, num_inputs, input_field_peak_rate, input_field_width, track_length)
    modulated_field_center = track_length * 0.5
    induction_start_loc = modulated_field_center + 10.
    induction_stop_loc = induction_start_loc + 5.
    tuning_amp = target_peak_delta_weight / 2.
    tuning_offset = tuning_amp
    force_delta_weights = tuning_amp * np.cos(
        2. * np.pi / target_weights_width * (peak_locs - modulated_field_center)) + \
                          tuning_offset
    left = np.where(peak_locs >= modulated_field_center - target_weights_width / 2.)[0][0]
    right = np.where(peak_locs > modulated_field_center + target_weights_width / 2.)[0][0]
    force_delta_weights[:left] = 0.
    force_delta_weights[right:] = 0.
    target_ramp = force_delta_weights.dot(input_rate_maps)
    ramp_scaling_factor = target_peak_ramp_amp / np.max(target_ramp)
    input_matrix = np.multiply(input_rate_maps, ramp_scaling_factor)
    target_ramp = force_delta_weights.dot(input_matrix)
    if len(target_ramp) != len(ramp_x):
        target_ramp = np.interp(ramp_x, input_x, target_ramp)

    ramp_amp, ramp_width, peak_shift, ratio, start_loc, peak_loc, end_loc, min_val, min_loc = \
        {}, {}, {}, {}, {}, {}, {}, {}, {}
    ramp_amp['target'], ramp_width['target'], peak_shift['target'], ratio['target'], start_loc['target'], \
    peak_loc['target'], end_loc['target'], min_val['target'], min_loc['target'] = \
        calculate_ramp_features(ramp=target_ramp, induction_loc=induction_start_loc, binned_x=ramp_x,
                                interp_x=interp_x, track_length=track_length)
    if verbose > 1:
        print('target: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, ' \
              'peak_loc: %.1f, end_loc: %.1f' % (ramp_amp['target'], ramp_width['target'], peak_shift['target'],
                                                 ratio['target'], start_loc['target'], peak_loc['target'],
                                                 end_loc['target']))
    if plot:
        x_start = induction_start_loc
        x_end = induction_stop_loc
        ylim = np.max(target_ramp)
        ymin = np.min(target_ramp)
        fig, axes = plt.subplots(1, 2)
        axes[0].plot(ramp_x, target_ramp, color='k')
        axes[0].hlines(ylim + 0.2, xmin=x_start, xmax=x_end, linewidth=2, colors='k')
        axes[0].set_xlabel('Location (cm)')
        axes[0].set_ylabel('Ramp amplitude (mV)')
        axes[0].set_xlim([0., track_length])
        axes[0].set_ylim([math.floor(ymin), max(math.ceil(ylim), ylim + 0.4)])
        axes[0].legend(loc='best', frameon=False, framealpha=0.5)

        ylim = np.max(force_delta_weights) + 1.
        ymin = np.min(force_delta_weights) + 1.
        axes[1].plot(peak_locs, force_delta_weights + 1., c='r')
        axes[1].hlines(ylim + 0.2, xmin=x_start, xmax=x_end, linewidth=2, colors='k')
        axes[1].set_xlabel('Location (cm)')
        axes[1].set_ylabel('Candidate synaptic weights (a.u.)')
        axes[1].set_xlim([0., track_length])
        axes[1].set_ylim([math.floor(ymin), max(math.ceil(ylim), ylim + 0.4)])
        clean_axes(axes)
        fig.tight_layout()
        fig.show()

    return target_ramp, ramp_scaling_factor


def calibrate_ramp_scaling_factor(ramp_x, input_x, interp_x, input_rate_maps, peak_locs, track_length, target_range,
                                  bounds, beta=2., initial_ramp_scaling_factor=0.002956, calibration_ramp_width=108.,
                                  calibration_ramp_amp=6., calibration_peak_delta_weight=1.5, plot=False, verbose=1):
    """
    Used to calibrate ramp_scaling_factor. Given a set of input_rate_maps and a target calibration_ramp_width,
    computes and a set of weights with the target calibration_peak_delta_weight and calibration_ramp_amp. Returns the
    new ramp_scaling_factor.
    :param ramp_x: array (spatial resolution of ramp)
    :param input_x: array (spatial resolution of input_rate_maps)
    :param interp_x: array (spatial resolution for computing fine features)
    :param input_rate_maps: array
    :param peak_locs: array
    :param calibration_ramp_width: float
    :param track_length: float
    :param target_range: dict
    :param bounds: tuple of float
    :param beta: float; regularization parameter
    :param initial_ramp_scaling_factor: float
    :param calibration_ramp_amp: float (mV)
    :param calibration_peak_delta_weight: float
    :param plot: bool
    :param verbose: int
    :return: float
    """
    calibration_sigma = calibration_ramp_width / 3. / np.sqrt(2.)
    calibration_peak_loc = track_length / 2.
    target_ramp = calibration_ramp_amp * np.exp(-((ramp_x - calibration_peak_loc) / calibration_sigma) ** 2.)

    induction_start_loc = calibration_peak_loc + 10.
    induction_stop_loc = calibration_peak_loc + 5.

    ramp_amp, ramp_width, peak_shift, ratio, start_loc, peak_loc, end_loc, min_val, min_loc = \
        {}, {}, {}, {}, {}, {}, {}, {}, {}
    ramp_amp['target'], ramp_width['target'], peak_shift['target'], ratio['target'], start_loc['target'], \
    peak_loc['target'], end_loc['target'], min_val['target'], min_loc['target'] = \
        calculate_ramp_features(ramp=target_ramp, induction_loc=induction_start_loc, binned_x=ramp_x,
                                interp_x=interp_x, track_length=track_length)

    if len(target_ramp) != len(input_x):
        interp_target_ramp = np.interp(input_x, ramp_x, target_ramp)
    else:
        interp_target_ramp = np.array(target_ramp)
    input_matrix = np.multiply(input_rate_maps, initial_ramp_scaling_factor)
    [U, s, Vh] = np.linalg.svd(input_matrix)
    V = Vh.T
    D = np.zeros_like(input_matrix)
    D[np.where(np.eye(*D.shape))] = s / (s ** 2. + beta ** 2.)
    input_matrix_inv = V.dot(D.conj().T).dot(U.conj().T)
    delta_weights = interp_target_ramp.dot(input_matrix_inv)

    SVD_delta_weights, SVD_ramp_scaling_factor = \
        get_adjusted_delta_weights_and_ramp_scaling_factor(delta_weights, input_rate_maps,
                                                           calibration_peak_delta_weight,
                                                           calibration_ramp_amp)
    if bounds is not None:
        SVD_delta_weights = np.maximum(np.minimum(SVD_delta_weights, bounds[1]), bounds[0])

    input_matrix = np.multiply(input_rate_maps, SVD_ramp_scaling_factor)
    SVD_model_ramp = SVD_delta_weights.dot(input_matrix)

    result = minimize(get_residual_score, SVD_delta_weights,
                      args=(target_ramp, ramp_x, input_x, interp_x, input_rate_maps, SVD_ramp_scaling_factor,
                            induction_start_loc, track_length, target_range, bounds), method='L-BFGS-B',
                      bounds=[bounds] * len(SVD_delta_weights), options={'disp': verbose > 1, 'maxiter': 100})

    LSA_delta_weights = result.x
    delta_weights, ramp_scaling_factor = \
        get_adjusted_delta_weights_and_ramp_scaling_factor(LSA_delta_weights, input_rate_maps,
                                                           calibration_peak_delta_weight,
                                                           calibration_ramp_amp)

    input_matrix = np.multiply(input_rate_maps, ramp_scaling_factor)
    model_ramp = delta_weights.dot(input_matrix)

    if len(model_ramp) != len(ramp_x):
        model_ramp = np.interp(ramp_x, input_x, model_ramp)
    ramp_amp['model'], ramp_width['model'], peak_shift['model'], ratio['model'], start_loc['model'], \
    peak_loc['model'], end_loc['model'], min_val['model'], min_loc['model'] = \
        calculate_ramp_features(ramp=model_ramp, induction_loc=induction_start_loc, binned_x=ramp_x, interp_x=interp_x,
                                track_length=track_length)

    if verbose > 1:
        print('target: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, ' \
              'peak_loc: %.1f, end_loc: %.1f' % (ramp_amp['target'], ramp_width['target'], peak_shift['target'],
                                                 ratio['target'], start_loc['target'], peak_loc['target'],
                                                 end_loc['target']))
        print('model: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, peak_loc: %.1f' \
              ', end_loc: %.1f' % (ramp_amp['model'], ramp_width['model'], peak_shift['model'], ratio['model'],
                                   start_loc['model'], peak_loc['model'], end_loc['model']))
        print('ramp_scaling_factor: %.6f' % ramp_scaling_factor)

    sys.stdout.flush()

    if plot:
        x_start = induction_start_loc
        x_end = induction_stop_loc
        ylim = max(np.max(target_ramp), np.max(model_ramp))
        ymin = min(np.min(target_ramp), np.min(model_ramp))
        fig, axes = plt.subplots(1, 2)
        axes[0].plot(ramp_x, target_ramp, label='Target', color='k')
        axes[0].plot(ramp_x, SVD_model_ramp, label='Model (SVD)', color='r')
        axes[0].plot(ramp_x, model_ramp, label='Model (LSA)', color='c')
        axes[0].hlines(ylim + 0.2, xmin=x_start, xmax=x_end, linewidth=2, colors='k')
        axes[0].set_xlabel('Location (cm)')
        axes[0].set_ylabel('Ramp amplitude (mV)')
        axes[0].set_xlim([0., track_length])
        axes[0].set_ylim([math.floor(ymin), max(math.ceil(ylim), ylim + 0.4)])
        axes[0].legend(loc='best', frameon=False, framealpha=0.5)

        ylim = max(np.max(SVD_delta_weights), np.max(delta_weights)) + 1.
        ymin = min(np.min(SVD_delta_weights), np.min(delta_weights)) + 1.
        axes[1].plot(peak_locs, SVD_delta_weights + 1., c='r', label='Model (SVD)')
        axes[1].plot(peak_locs, delta_weights + 1., c='c', label='Model (LSA)')
        axes[1].hlines(ylim + 0.2, xmin=x_start, xmax=x_end, linewidth=2, colors='k')
        axes[1].set_xlabel('Location (cm)')
        axes[1].set_ylabel('Candidate synaptic weights (a.u.)')
        axes[1].set_xlim([0., track_length])
        axes[1].set_ylim([math.floor(ymin), max(math.ceil(ylim), ylim + 0.4)])
        clean_axes(axes)
        fig.tight_layout()
        fig.show()

    return ramp_scaling_factor


def get_delta_weights_LSA(target_ramp, ramp_x, input_x, interp_x, input_rate_maps, peak_locs, ramp_scaling_factor,
                          induction_start_loc, induction_stop_loc, track_length, target_range, bounds,
                          initial_delta_weights=None, beta=2., allow_offset=False, impose_offset=None, plot=False,
                          label='', verbose=1):
    """
    Uses least square approximation to estimate a set of weights to match any arbitrary place field ramp, agnostic
    about underlying kernel, induction velocity, etc.
    :param target_ramp: dict of array
    :param ramp_x: array (spatial resolution of ramp)
    :param input_x: array (spatial resolution of input_rate_maps)
    :param interp_x: array (spatial resolution for computing fine features)
    :param input_rate_maps: array
    :param peak_locs: array
    :param ramp_scaling_factor: float
    :param induction_start_loc: float
    :param induction_stop_loc: float
    :param track_length: float
    :param target_range: dict
    :param bounds: tuple of float
    :param initial_delta_weights: array
    :param beta: float; regularization parameter
    :param allow_offset: bool (allow special case where baseline Vm before 1st induction is unknown)
    :param impose_offset: float (impose Vm offset from 1st induction on 2nd induction)
    :param plot: bool
    :param label: str
    :param verbose: int
    :return: tuple of array
    """
    if len(target_ramp) != len(input_x):
        exp_ramp = np.interp(input_x, ramp_x, target_ramp)
    else:
        exp_ramp = np.array(target_ramp)

    input_matrix = np.multiply(input_rate_maps, ramp_scaling_factor)
    if initial_delta_weights is None:
        [U, s, Vh] = np.linalg.svd(input_matrix)
        V = Vh.T
        D = np.zeros_like(input_matrix)
        D[np.where(np.eye(*D.shape))] = s / (s ** 2. + beta ** 2.)
        input_matrix_inv = V.dot(D.conj().T).dot(U.conj().T)
        initial_delta_weights = exp_ramp.dot(input_matrix_inv)
        initial_ramp = initial_delta_weights.dot(input_matrix)
        SVD_scaling_factor = np.max(exp_ramp) / np.max(initial_ramp)
        initial_delta_weights *= SVD_scaling_factor

    if bounds is not None:
        initial_delta_weights = np.maximum(np.minimum(initial_delta_weights, bounds[1]), bounds[0])

    initial_ramp = initial_delta_weights.dot(input_matrix)

    result = minimize(get_residual_score, initial_delta_weights,
                      args=(target_ramp, ramp_x, input_x, interp_x, input_rate_maps, ramp_scaling_factor,
                            induction_start_loc, track_length, target_range, bounds, allow_offset, impose_offset),
                      method='L-BFGS-B', bounds=[bounds] * len(initial_delta_weights),
                      options={'disp': verbose > 1, 'maxiter': 100})

    if verbose > 1:
        print('get_delta_weights_LSA: process: %i; %s:' % (os.getpid(), label))
        sys.stdout.flush()
    model_ramp, delta_weights, ramp_offset, residual_score = \
        get_residual_score(result.x, target_ramp, ramp_x, input_x, interp_x, input_rate_maps, ramp_scaling_factor,
                           induction_start_loc, track_length, target_range, bounds, allow_offset, impose_offset,
                           disp=verbose > 1, full_output=True)

    if plot:
        x_start = induction_start_loc
        x_end = induction_stop_loc
        ylim = max(np.max(target_ramp), np.max(model_ramp))
        ymin = min(np.min(target_ramp), np.min(model_ramp))
        fig, axes = plt.subplots(1, 2)
        axes[0].plot(ramp_x, target_ramp, label='Target', color='k')
        axes[0].plot(ramp_x, initial_ramp, label='Model (Initial)', color='r')
        axes[0].plot(ramp_x, model_ramp, label='Model (LSA)', color='c')
        axes[0].hlines(ylim + 0.2, xmin=x_start, xmax=x_end, linewidth=2, colors='k')
        axes[0].set_xlabel('Location (cm)')
        axes[0].set_ylabel('Ramp amplitude (mV)')
        axes[0].set_xlim([0., track_length])
        axes[0].set_ylim([math.floor(ymin), max(math.ceil(ylim), ylim + 0.4)])
        axes[0].legend(loc='best', frameon=False, framealpha=0.5)
        if len(label) > 0:
            axes[0].set_title(label)

        ylim = max(np.max(initial_delta_weights), np.max(delta_weights)) + 1.
        ymin = min(np.min(initial_delta_weights), np.min(delta_weights)) + 1.
        axes[1].plot(peak_locs, initial_delta_weights + 1., c='r', label='Model (Initial)')
        axes[1].plot(peak_locs, delta_weights + 1., c='c', label='Model (LSA)')
        axes[1].hlines(ylim + 0.2, xmin=x_start, xmax=x_end, linewidth=2, colors='k')
        axes[1].set_xlabel('Location (cm)')
        axes[1].set_ylabel('Candidate synaptic weights (a.u.)')
        axes[1].set_xlim([0., track_length])
        axes[1].set_ylim([math.floor(ymin), max(math.ceil(ylim), ylim + 0.4)])
        clean_axes(axes)
        fig.tight_layout()
        fig.show()

    return model_ramp, delta_weights, ramp_offset, residual_score


def get_target_synthetic_ramp(induction_loc, ramp_x, track_length, target_peak_val=8., target_min_val=0.,
                              target_asymmetry=1.8, target_peak_shift=-10., target_ramp_width=187., plot=False):
    """
    :param induction_loc: float
    :param ramp_x: array (spatial resolution of ramp)
    :param track_length: float
    :param target_peak_val: float
    :param target_min_val: float
    :param target_asymmetry: float
    :param target_peak_shift: float
    :param target_ramp_width: float
    :param plot: bool
    :return: array
    """
    peak_loc = induction_loc + target_peak_shift
    amp = target_peak_val - target_min_val
    extended_x = np.concatenate([ramp_x - track_length, ramp_x, ramp_x + track_length])
    left_width = target_asymmetry / (1. + target_asymmetry) * target_ramp_width + target_peak_shift
    right_width = 1. / (1. + target_asymmetry) * target_ramp_width - target_peak_shift
    left_sigma = 2. * left_width / 3. / np.sqrt(2.)
    right_sigma = 2. * right_width / 3. / np.sqrt(2.)
    left_waveform = amp * np.exp(-((extended_x - peak_loc) / left_sigma) ** 2.)
    right_waveform = amp * np.exp(-((extended_x - peak_loc) / right_sigma) ** 2.)
    peak_index = np.argmax(left_waveform)
    waveform = np.array(left_waveform)
    waveform[peak_index + 1:] = right_waveform[peak_index + 1:]
    waveform = wrap_around_and_compress(waveform, ramp_x)

    waveform -= np.min(waveform)
    waveform /= np.max(waveform)
    waveform *= amp
    waveform += target_min_val

    if plot:
        fig, axes = plt.subplots()
        axes.plot(ramp_x, waveform)
        axes.set_ylabel('Ramp amplitude (mV)')
        axes.set_xlabel('Location (cm)')
        clean_axes(axes)
        fig.show()

    return waveform


def merge_exported_biBTSP_model_output_files_from_yaml(yaml_file_path, label='all_cells_merged_exported_model_output',
                                                       input_dir=None, output_dir=None, exclude=None, include=None,
                                                       verbose=True):
    """

    :param yaml_file_path: str (path to file)
    :param label: str
    :param input_dir: str (path to dir)
    :param output_dir: str (path to dir)
    :param exclude: list of str
    :param include: list of str
    :param verbose: bool
    """
    from nested.optimize_utils import merge_exported_data
    if not os.path.isfile(yaml_file_path):
        raise Exception('merge_exported_biBTSP_model_output_files_from_yaml: missing yaml_file at specified path: '
                        '%s' % yaml_file_path)
    filename_dict = read_from_yaml(yaml_file_path)
    if not len(filename_dict) > 0:
        raise RuntimeError('merge_exported_biBTSP_model_output_files_from_yaml: no data exported; empty filename_dict '
                           'loaded from path: %s' % yaml_file_path)
    if label is None:
        label = ''
    else:
        label = '_%s' % label
    if input_dir is None:
        input_prefix = ''
    elif not os.path.isdir(input_dir):
        raise RuntimeError('merge_exported_biBTSP_model_output_files_from_yaml: cannot find input_dir: %s' % input_dir)
    else:
        input_prefix = '%s/' % input_dir
    if output_dir is None:
        output_prefix = ''
    elif not os.path.isdir(output_dir):
        raise RuntimeError('merge_exported_biBTSP_model_output_files_from_yaml: cannot find output_dir: %s' %
                           output_dir)
    else:
        output_prefix = '%s/' % output_dir
    if include is None:
        include = list(filename_dict.keys())
    if exclude is not None:
        for model_key in exclude:
            if model_key in include:
                include.remove(model_key)
    for model_key in filename_dict:
        if model_key not in include:
            continue
        for input_field_width in filename_dict[model_key]:
            new_file_path = '%s%s_biBTSP_%s_%scm%s.hdf5' % \
                            (output_prefix, datetime.datetime.today().strftime('%Y%m%d'), model_key,
                             input_field_width, label)
            file_path_list = ['%s%s' % (input_prefix, filename) for filename in
                              filename_dict[model_key][input_field_width]]
            merge_exported_data(file_path_list, new_file_path, verbose=verbose)


def update_min_t_arrays(binned_extra_x, t, x, backward_t, forward_t):
    """

    :param binned_extra_x: array
    :param t: array
    :param x: array
    :param backward_t: array
    :param forward_t: array
    :return array, array
    """
    binned_t = np.interp(binned_extra_x, x, t)
    temp_backward_t = np.empty_like(binned_extra_x)
    temp_backward_t[:] = np.nan
    temp_forward_t = np.array(temp_backward_t)
    backward_indexes = np.where(binned_t <= 0.)[0]
    forward_indexes = np.where(binned_t >= 0.)[0]
    temp_backward_t[backward_indexes] = binned_t[backward_indexes]
    temp_forward_t[forward_indexes] = binned_t[forward_indexes]
    backward_t = np.nanmax([backward_t, temp_backward_t], axis=0)
    forward_t = np.nanmin([forward_t, temp_forward_t], axis=0)
    return backward_t, forward_t


def merge_min_t_arrays(binned_x, binned_extra_x, extended_binned_x, induction_loc, backward_t, forward_t, debug=False):
    """

    :param binned_x: array
    :param binned_extra_x: array
    :param extended_binned_x: array
    :param induction_loc:
    :param backward_t:
    :param forward_t:
    :param debug: bool
    :return: array
    """
    merged_min_t = np.empty_like(binned_extra_x)
    merged_min_t[:] = np.nan
    extended_min_t = np.empty_like(extended_binned_x)
    extended_min_t[:] = np.nan
    before = np.where(binned_extra_x < induction_loc)[0]
    if np.any(before):
        merged_min_t[before] = backward_t[before]
        extended_min_t[np.add(before[:-1], 2 * len(binned_x))] = forward_t[before[:-1]]
    else:
        merged_min_t[0] = backward_t[0]
        if debug:
            print('merge_min_t_arrays: no before indexes')
    after = np.where(binned_extra_x >= induction_loc)[0]
    if np.any(after):
        merged_min_t[after] = forward_t[after]
        extended_min_t[after[1:-1]] = backward_t[after[1:-1]]
    else:
        if debug:
            print('merge_min_t_arrays: no after indexes')
    if debug:
        for i in range(len(merged_min_t)):
            val = merged_min_t[i]
            if np.isnan(val):
                print('merge_min_t_arrays: nan in merged_min_t at index: %i' % i)
                break
        fig4, axes4 = plt.subplots(1)
        axes4.plot(binned_extra_x, backward_t, binned_extra_x, forward_t)
        axes4.plot(binned_extra_x, merged_min_t, label='Merged')
        axes4.legend(loc='best', frameon=False, framealpha=0.5)
        fig4.show()

        print('merge_min_t_arrays: val at backward_t[0]: %.2f; val at forward_t[-1]: %.2f' % \
              (backward_t[0], forward_t[-1]))
    extended_min_t[len(binned_x):2 * len(binned_x)] = merged_min_t[:-1]
    return extended_min_t


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def colorline(x, y, z, cmap='RdBu_r', vmin=None, vmax=None, linewidth=1., alpha=1.):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    from matplotlib.collections import LineCollection
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    if vmin is None:
        vmin = np.min(z)
    if vmax is None:
        vmax = np.max(z)

    norm = plt.Normalize(vmin, vmax)

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm,
                        linewidth=linewidth, alpha=alpha)

    return lc


def get_circular_distance(start_loc, end_loc, track_length):
    delta_loc = end_loc - start_loc
    if delta_loc > track_length / 2.:
        delta_loc -= track_length
    elif delta_loc < -track_length / 2.:
        delta_loc += track_length
    return delta_loc


def get_min_induction_t(complete_t, complete_position, binned_x, track_length, induction_loc, num_induction_laps,
                        plot=False, title=None):
    """

    :param complete_t: array
    :param complete_position: array
    :param binned_x: array
    :param track_length: float
    :param induction_loc: float
    :param num_induction_laps: int
    :param plot: bool
    :param title: str
    :return:
    """
    induction_loc_start_times = []
    target_loc = induction_loc
    for i in range(num_induction_laps):
        this_start_index = np.where(complete_position >= target_loc)[0][0]
        induction_loc_start_times.append(complete_t[this_start_index])
        target_loc += track_length
        loc_offset = 0

    backward_intervals = np.empty((len(binned_x), len(induction_loc_start_times)))
    forward_intervals = np.empty((len(binned_x), len(induction_loc_start_times)))

    for j, start_time in enumerate(induction_loc_start_times):
        offset_t = np.subtract(complete_t, start_time)
        for i, loc in enumerate(binned_x):
            next_indexes = np.where(complete_position >= loc + loc_offset)[0]
            if len(next_indexes) > 0:
                next_t = offset_t[next_indexes[0]]
                if next_t > 0.:
                    prev_indexes = np.where(complete_position < loc + loc_offset - track_length)[0]
                    if len(prev_indexes) > 0:
                        prev_t = offset_t[prev_indexes[-1]]
                    else:
                        prev_t = np.nan
                else:
                    prev_indexes = np.where(complete_position < loc + loc_offset)[0]
                    if len(prev_indexes) > 0:
                        prev_t = offset_t[prev_indexes[-1]]
                    else:
                        prev_t = np.nan
                    next_indexes = np.where(complete_position >= loc + loc_offset + track_length)[0]
                    if len(next_indexes) > 0:
                        next_t = offset_t[next_indexes[0]]
                    else:
                        next_t = np.nan
            else:
                next_t = np.nan
                prev_indexes = np.where(complete_position < loc + loc_offset - track_length)[0]
                if len(prev_indexes) > 0:
                    prev_t = offset_t[prev_indexes[-1]]
                else:
                    prev_t = np.nan
            # if prev_t > 0. or next_t < 0.:
            #     print('j: %i, loc: %.1f, prev_t: %.1f, next_t: %.1f' % (j, loc, prev_t, next_t))
            if prev_t > 0.:
                prev_t = np.nan
            if next_t < 0.:
                next_t = np.nan
            forward_intervals[i, j] = next_t
            backward_intervals[i, j] = prev_t
        loc_offset += track_length

    backward = np.nanmax(backward_intervals, axis=1)
    forward = np.nanmin(forward_intervals, axis=1)

    nearest = []
    for i in range(len(backward)):
        if np.isnan(backward[i]):
            if np.isnan(forward[i]):
                nearest.append(np.nan)
            else:
                nearest.append(forward[i])
        elif np.isnan(forward[i]):
            nearest.append(backward[i])
        elif -backward[i] < forward[i]:
            nearest.append(backward[i])
        else:
            nearest.append(forward[i])

    nearest = np.array(nearest)
    if plot:
        fig, axes = plt.subplots()
        for j in range(len(induction_loc_start_times)):
            axes.plot(binned_x, backward_intervals[:, j] / 1000., c='grey', alpha=0.5)
            axes.plot(binned_x, forward_intervals[:, j] / 1000., c='grey', alpha=0.5)
        axes.plot(binned_x, backward / 1000., c='r')
        axes.plot(binned_x, forward / 1000., c='c')
        axes.plot(binned_x, nearest / 1000., c='k')
        axes.set_xlabel('Location (cm)')
        axes.set_ylabel('Time (s)')
        clean_axes(axes)
        fig.tight_layout()
        if title is not None:
            fig.suptitle(title, y=0.98)
            fig.subplots_adjust(top=0.9)
        fig.show()

    return nearest


def get_clean_induction_t_indexes(t, max_delta_t=1000.):
    t = np.copy(t)
    sorted_indexes = np.argsort(t)
    for i, val in enumerate(np.diff(t[sorted_indexes])):
        if val >= max_delta_t:
            if t[sorted_indexes[i]] < 0.:
                t[sorted_indexes[:i + 1]] = np.nan
            else:
                t[sorted_indexes[i + 1:]] = np.nan
    sorted_indexes = np.argsort(t)
    valid_indexes = ~np.isnan(t[sorted_indexes])

    return sorted_indexes[valid_indexes]


def get_biBTSP_analysis_results(data_file_path, binned_x, binned_extra_x, extended_binned_x, reference_delta_t,
                                track_length, dt, debug=False, truncate=False):
    """

    :param data_file_path: str (path)
    :param binned_x: array
    :param binned_extra_x: array
    :param extended_binned_x: array
    :param reference_delta_t: array
    :param track_length: float
    :param dt: float
    :param debug: bool
    :param truncate: bool
    :return: tuple
    """
    peak_ramp_amp = []
    total_induction_dur = []
    initial_induction_delta_vm = defaultdict(list)

    group_indexes = defaultdict(list)

    exp_ramp = defaultdict(lambda: defaultdict(dict))
    extended_exp_ramp = defaultdict(lambda: defaultdict(dict))
    delta_exp_ramp = defaultdict(dict)
    mean_induction_loc = defaultdict(dict)
    extended_min_delta_t = defaultdict(dict)
    extended_delta_exp_ramp = defaultdict(dict)
    interp_initial_exp_ramp = defaultdict(dict)
    interp_delta_exp_ramp = defaultdict(dict)
    interp_final_exp_ramp = defaultdict(dict)

    with h5py.File(data_file_path, 'r') as f:
        for cell_key in f['data']:
            for induction_key in f['data'][cell_key]:
                induction_locs = f['data'][cell_key][induction_key].attrs['induction_locs']
                induction_durs = f['data'][cell_key][induction_key].attrs['induction_durs']
                if induction_key == '1':
                    if f['data'][cell_key].attrs['spont']:
                        group = 'spont'
                    else:
                        group = 'exp1'
                else:
                    group = 'exp2'
                group_indexes[group].append(len(total_induction_dur))
                total_induction_dur.append(np.sum(induction_durs))

                if induction_key == '1' and '2' in f['data'][cell_key]:
                    exp_ramp[cell_key][induction_key]['after'] = \
                        f['data'][cell_key]['2']['processed']['exp_ramp']['before'][:]
                else:
                    exp_ramp[cell_key][induction_key]['after'] = \
                        f['data'][cell_key][induction_key]['processed']['exp_ramp']['after'][:]

                if 'before' in f['data'][cell_key][induction_key]['processed']['exp_ramp']:
                    exp_ramp[cell_key][induction_key]['before'] = \
                        f['data'][cell_key][induction_key]['processed']['exp_ramp']['before'][:]
                    delta_exp_ramp[cell_key][induction_key] = np.subtract(exp_ramp[cell_key][induction_key]['after'],
                                                                          exp_ramp[cell_key][induction_key]['before'])
                else:
                    exp_ramp[cell_key][induction_key]['before'] = \
                        np.zeros_like(exp_ramp[cell_key][induction_key]['after'])
                    delta_exp_ramp[cell_key][induction_key], discard = \
                        subtract_baseline(exp_ramp[cell_key][induction_key]['after'])

                peak_ramp_amp.append(np.max(exp_ramp[cell_key][induction_key]['after']))
                mean_induction_loc[cell_key][induction_key] = np.mean(induction_locs)
                extended_delta_exp_ramp[cell_key][induction_key] = \
                    np.concatenate([delta_exp_ramp[cell_key][induction_key]] * 3)
                for category in exp_ramp[cell_key][induction_key]:
                    extended_exp_ramp[cell_key][induction_key][category] = \
                        np.concatenate([exp_ramp[cell_key][induction_key][category]] * 3)
                if debug:
                    fig1, axes1 = plt.subplots(1)
                backward_t = np.empty_like(binned_extra_x)
                backward_t[:] = np.nan
                forward_t = np.array(backward_t)
                for i in range(len(induction_locs)):
                    this_induction_loc = mean_induction_loc[cell_key][induction_key]
                    key = str(i)
                    this_position = f['data'][cell_key][induction_key]['processed']['position']['induction'][key][:]
                    this_t = f['data'][cell_key][induction_key]['processed']['t']['induction'][key][:]
                    this_induction_index = np.where(this_position >= this_induction_loc)[0][0]
                    this_induction_t = this_t[this_induction_index]
                    if i == 0 and 'pre' in f['data'][cell_key][induction_key]['raw']['position']:
                        pre_position = f['data'][cell_key][induction_key]['processed']['position']['pre']['0'][:]
                        pre_t = f['data'][cell_key][induction_key]['processed']['t']['pre']['0'][:]
                        pre_t -= len(pre_t) * dt
                        pre_t -= this_induction_t
                        backward_t, forward_t = update_min_t_arrays(binned_extra_x, pre_t, pre_position, backward_t,
                                                                    forward_t)
                        if debug:
                            axes1.plot(np.subtract(pre_position,
                                                   track_length + mean_induction_loc[cell_key][induction_key]), pre_t,
                                       label='Lap: Pre')
                    elif i > 0:
                        prev_t -= len(prev_t) * dt
                        prev_induction_t = prev_t[prev_induction_index]
                        backward_t, forward_t = update_min_t_arrays(binned_extra_x,
                                                                    np.subtract(prev_t, this_induction_t),
                                                                    prev_position, backward_t, forward_t)
                        if debug:
                            axes1.plot(np.subtract(prev_position,
                                                   track_length + mean_induction_loc[cell_key][induction_key]),
                                       np.subtract(prev_t, this_induction_t), label='Lap: %s (Prev)' % prev_key)

                        backward_t, forward_t = update_min_t_arrays(binned_extra_x,
                                                                    np.subtract(this_t, prev_induction_t),
                                                                    this_position, backward_t, forward_t)
                        if debug:
                            axes1.plot(np.subtract(this_position,
                                                   mean_induction_loc[cell_key][induction_key] - track_length),
                                       np.subtract(this_t, prev_induction_t), label='Lap: %s (Next)' % key)
                    backward_t, forward_t = update_min_t_arrays(binned_extra_x, np.subtract(this_t, this_induction_t),
                                                                this_position, backward_t, forward_t)
                    if debug:
                        axes1.plot(np.subtract(this_position, mean_induction_loc[cell_key][induction_key]),
                                   np.subtract(this_t, this_induction_t), label='Lap: %s (Current)' % key)
                    if i == len(induction_locs) - 1 and 'post' in f['data'][cell_key][induction_key]['raw']['position']:
                        post_position = f['data'][cell_key][induction_key]['processed']['position']['post']['0'][:]
                        post_t = f['data'][cell_key][induction_key]['processed']['t']['post']['0'][:]
                        post_t += len(this_t) * dt
                        post_t -= this_induction_t
                        backward_t, forward_t = update_min_t_arrays(binned_extra_x, post_t, post_position, backward_t,
                                                                    forward_t)
                        if debug:
                            axes1.plot(np.subtract(post_position,
                                                   mean_induction_loc[cell_key][induction_key] - track_length), post_t,
                                       label='Lap: Post')
                    prev_key = key
                    prev_induction_index = this_induction_index
                    prev_t = this_t
                    prev_position = this_position
                extended_min_delta_t[cell_key][induction_key] = \
                    merge_min_t_arrays(binned_x, binned_extra_x, extended_binned_x,
                                       mean_induction_loc[cell_key][induction_key], backward_t, forward_t)
                this_extended_delta_position = np.subtract(extended_binned_x,
                                                           mean_induction_loc[cell_key][induction_key])
                if debug:
                    axes1.plot(this_extended_delta_position, extended_min_delta_t[cell_key][induction_key], c='k',
                               label='Min Interval')
                    fig1.suptitle('Cell: %s; Induction: %s' % (cell_key, induction_key),
                                  fontsize=mpl.rcParams['font.size'])
                    box = axes1.get_position()
                    axes1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    axes1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, framealpha=0.5)
                    axes1.set_xlabel('Position relative to plateau onset (cm)')
                    axes1.set_ylabel('Time relative to plateau onset (ms)')
                    clean_axes(axes1)
                    fig1.show()

                mask = ~np.isnan(extended_min_delta_t[cell_key][induction_key])
                indexes = np.where((extended_min_delta_t[cell_key][induction_key][mask] >= reference_delta_t[0]) &
                                   (extended_min_delta_t[cell_key][induction_key][mask] <= reference_delta_t[-1]))[0]
                bad_indexes = np.where((reference_delta_t <
                                        extended_min_delta_t[cell_key][induction_key][mask][indexes[0]]) |
                                       (reference_delta_t >
                                        extended_min_delta_t[cell_key][induction_key][mask][indexes[-1]]))[0]

                interp_delta_exp_ramp[cell_key][induction_key] = \
                    np.interp(reference_delta_t, extended_min_delta_t[cell_key][induction_key][mask],
                              extended_delta_exp_ramp[cell_key][induction_key][mask])
                interp_initial_exp_ramp[cell_key][induction_key] = \
                    np.interp(reference_delta_t, extended_min_delta_t[cell_key][induction_key][mask],
                              extended_exp_ramp[cell_key][induction_key]['before'][mask])
                interp_final_exp_ramp[cell_key][induction_key] = \
                    np.interp(reference_delta_t, extended_min_delta_t[cell_key][induction_key][mask],
                              extended_exp_ramp[cell_key][induction_key]['after'][mask])

                if truncate and len(bad_indexes) > 0:
                    interp_delta_exp_ramp[cell_key][induction_key][bad_indexes] = np.nan
                    interp_initial_exp_ramp[cell_key][induction_key][bad_indexes] = np.nan
                    interp_final_exp_ramp[cell_key][induction_key][bad_indexes] = np.nan

                if debug:
                    fig2, axes2 = plt.subplots()
                    axes2.plot(extended_min_delta_t[cell_key][induction_key],
                               extended_delta_exp_ramp[cell_key][induction_key], c='k')
                    axes2.plot(reference_delta_t, interp_delta_exp_ramp[cell_key][induction_key], c='r')
                    axes2.set_xlim(reference_delta_t[0], reference_delta_t[-1])
                    fig2.suptitle('Cell: %s; Induction: %s' % (cell_key, induction_key),
                                  fontsize=mpl.rcParams['font.size'])
                    axes2.set_xlabel('Time relative to plateau onset (ms)')
                    axes2.set_ylabel('Change in ramp\namplitude (mV)')
                    clean_axes(axes2)
                    fig2.show()

                if 'DC_soma' in f['data'][cell_key][induction_key].attrs and \
                        f['data'][cell_key][induction_key].attrs['DC_soma']:
                    if 'DC_soma_val' in f['data'][cell_key][induction_key].attrs and \
                            f['data'][cell_key][induction_key].attrs['DC_soma_val'] is not None:
                        this_induction_vm = np.ones_like(reference_delta_t) * \
                                            f['data'][cell_key][induction_key].attrs['DC_soma_val']
                    elif 'initial_induction_delta_vm' in f['data'][cell_key][induction_key]:
                        if 'mean_induction_t' not in f['data'][cell_key][induction_key]:
                            raise RuntimeError('get_biBTSP_analysis_results: cell: %s, induction: %s; when providing an'
                                               'initial_induction_delta_vm array for a DC_soma experiment, a '
                                               'mean_induction_t array must also be provided.' %
                                               (cell_key, induction_key))
                        mean_induction_t = f['data'][cell_key][induction_key]['processed']['mean_induction_t'][:]
                        this_induction_vm = \
                            f['data'][cell_key][induction_key]['processed']['initial_induction_delta_vm'][:]
                        this_induction_vm = np.interp(reference_delta_t, mean_induction_t, this_induction_vm)
                        indexes = np.where((reference_delta_t < np.min(mean_induction_t) * 1000.) |
                                           (reference_delta_t > np.max(mean_induction_t) * 1000.))[0]
                        this_induction_vm[indexes] = np.nan
                    else:
                        raise RuntimeError('get_biBTSP_analysis_results: cell: %s, induction: %s; for a DC_soma '
                                           'experiments, either a DC_soma_val or an initial_induction_delta_vm array '
                                           'must be provided.' % (cell_key, induction_key))
                else:
                    this_induction_vm = np.copy(interp_initial_exp_ramp[cell_key][induction_key])
                initial_induction_delta_vm[group].append(this_induction_vm)

    total_induction_dur = np.array(total_induction_dur)
    peak_ramp_amp = np.array(peak_ramp_amp)

    return peak_ramp_amp, total_induction_dur, initial_induction_delta_vm, group_indexes, exp_ramp, extended_exp_ramp, \
           delta_exp_ramp, mean_induction_loc, extended_min_delta_t, extended_delta_exp_ramp, interp_initial_exp_ramp, \
           interp_delta_exp_ramp, interp_final_exp_ramp


def get_biBTSP_analysis_results_alt(data_file_path, reference_delta_t, debug=False, truncate=2.5):
    """

    :param data_file_path: str (path)
    :param reference_delta_t: array
    :param debug: bool
    :param truncate: float
    :return: tuple
    """
    peak_ramp_amp = []
    total_induction_dur = []

    group_indexes = defaultdict(list)

    exp_ramp = defaultdict(lambda: defaultdict(dict))
    delta_exp_ramp = defaultdict(dict)
    mean_induction_loc = defaultdict(dict)
    min_induction_t = defaultdict(dict)
    clean_induction_t_indexes = defaultdict(dict)
    interp_delta_exp_ramp = defaultdict(dict)
    interp_exp_ramp = defaultdict(lambda: defaultdict(dict))

    with h5py.File(data_file_path, 'r') as f:
        for cell_key in f['data']:
            for induction_key in f['data'][cell_key]:
                induction_locs = f['data'][cell_key][induction_key].attrs['induction_locs']
                induction_durs = f['data'][cell_key][induction_key].attrs['induction_durs']
                if induction_key == '1' and f['data'][cell_key].attrs['spont']:
                    group = 'spont'
                else:
                    group = 'exp%s' % induction_key
                group_indexes[group].append(len(total_induction_dur))
                total_induction_dur.append(np.sum(induction_durs))

                exp_ramp[cell_key][induction_key]['after'] = \
                    f['data'][cell_key][induction_key]['processed']['exp_ramp']['after'][:]

                if 'before' in f['data'][cell_key][induction_key]['processed']['exp_ramp']:
                    exp_ramp[cell_key][induction_key]['before'] = \
                        f['data'][cell_key][induction_key]['processed']['exp_ramp']['before'][:]
                    delta_exp_ramp[cell_key][induction_key] = np.subtract(exp_ramp[cell_key][induction_key]['after'],
                                                                          exp_ramp[cell_key][induction_key]['before'])
                else:
                    exp_ramp[cell_key][induction_key]['before'] = \
                        np.zeros_like(exp_ramp[cell_key][induction_key]['after'])
                    delta_exp_ramp[cell_key][induction_key], discard = \
                        subtract_baseline(exp_ramp[cell_key][induction_key]['after'])

                peak_ramp_amp.append(np.max(exp_ramp[cell_key][induction_key]['after']))
                mean_induction_loc[cell_key][induction_key] = np.mean(induction_locs)
                if 'min_induction_t' in f['data'][cell_key][induction_key]['processed']:
                    this_induction_t = f['data'][cell_key][induction_key]['processed']['min_induction_t'][:]
                else:
                    raise RuntimeError(
                        'get_biBTSP_analysis_results: problem finding min_induction_t trace for cell: %s, induction: %s'
                        ' in data file: %s' % (cell_key, induction_key, data_file_path))
                this_clean_indexes = get_clean_induction_t_indexes(this_induction_t, truncate * 1000.)
                this_induction_t = this_induction_t[this_clean_indexes]
                min_induction_t[cell_key][induction_key] = this_induction_t
                clean_induction_t_indexes[cell_key][induction_key] = this_clean_indexes
                bad_indexes = np.where((reference_delta_t < np.min(this_induction_t)) |
                                       (reference_delta_t > np.max(this_induction_t)))[0]
                interp_exp_ramp[cell_key][induction_key]['before'] = np.interp(
                    reference_delta_t, this_induction_t,
                    exp_ramp[cell_key][induction_key]['before'][this_clean_indexes])
                interp_exp_ramp[cell_key][induction_key]['before'][bad_indexes] = np.nan
                interp_exp_ramp[cell_key][induction_key]['after'] = np.interp(
                    reference_delta_t, this_induction_t, exp_ramp[cell_key][induction_key]['after'][this_clean_indexes])
                interp_exp_ramp[cell_key][induction_key]['after'][bad_indexes] = np.nan
                interp_delta_exp_ramp[cell_key][induction_key] = np.interp(
                    reference_delta_t, this_induction_t, delta_exp_ramp[cell_key][induction_key][this_clean_indexes])
                interp_delta_exp_ramp[cell_key][induction_key][bad_indexes] = np.nan

                if debug:
                    fig2, axes2 = plt.subplots()
                    axes2.plot(reference_delta_t, interp_delta_exp_ramp[cell_key][induction_key], c='r')
                    axes2.plot(this_induction_t,
                               delta_exp_ramp[cell_key][induction_key][this_clean_indexes], c='k')
                    axes2.set_xlim(reference_delta_t[0], reference_delta_t[-1])
                    fig2.suptitle('Cell: %s; Induction: %s' % (cell_key, induction_key),
                                  fontsize=mpl.rcParams['font.size'])
                    axes2.set_xlabel('Time relative to plateau onset (ms)')
                    axes2.set_ylabel('Change in ramp\namplitude (mV)')
                    clean_axes(axes2)
                    fig2.show()

    total_induction_dur = np.array(total_induction_dur)
    peak_ramp_amp = np.array(peak_ramp_amp)

    return peak_ramp_amp, total_induction_dur, group_indexes, exp_ramp, delta_exp_ramp, mean_induction_loc, \
           interp_exp_ramp, interp_delta_exp_ramp, min_induction_t, clean_induction_t_indexes


def get_biBTSP_DC_soma_analysis_results(data_file_path, binned_x, binned_extra_x, extended_binned_x, reference_delta_t,
                                        track_length, dt, group='exp1', induction=1, debug=False, truncate=False):
    """

    :param data_file_path: str (path)
    :param binned_x: array
    :param binned_extra_x: array
    :param extended_binned_x: array
    :param reference_delta_t: array
    :param track_length: float
    :param dt: float
    :param group: str
    :param induction: int
    :param debug: bool
    :param truncate: bool
    :return: tuple
    """
    initial_induction_delta_vm = []
    interp_delta_exp_ramp = []
    interp_initial_exp_ramp = []

    induction_key = str(induction)
    target_group = group

    exp_ramp = defaultdict(lambda: defaultdict(dict))
    extended_exp_ramp = defaultdict(lambda: defaultdict(dict))
    delta_exp_ramp = defaultdict(dict)
    mean_induction_loc = defaultdict(dict)
    extended_min_delta_t = defaultdict(dict)
    extended_delta_exp_ramp = defaultdict(dict)

    with h5py.File(data_file_path, 'r') as f:
        for cell_key in f['data']:
            if induction_key not in f['data'][cell_key]:
                continue
            if induction_key == '1':
                if f['data'][cell_key].attrs['spont']:
                    this_group = 'spont'
                else:
                    this_group = 'exp1'
            elif induction_key == '2':
                this_group = 'exp2'
            else:
                raise RuntimeError('get_biBTSP_DC_soma_analysis_results: group: %s not recognized' % this_group)
            if this_group != target_group:
                continue
            if induction_key == '1' and '2' in f['data'][cell_key]:
                exp_ramp[cell_key][induction_key]['after'] = \
                    f['data'][cell_key]['2']['processed']['exp_ramp']['before'][:]
            else:
                exp_ramp[cell_key][induction_key]['after'] = \
                    f['data'][cell_key][induction_key]['processed']['exp_ramp']['after'][:]

            if 'before' in f['data'][cell_key][induction_key]['processed']['exp_ramp']:
                exp_ramp[cell_key][induction_key]['before'] = \
                    f['data'][cell_key][induction_key]['processed']['exp_ramp']['before'][:]
                delta_exp_ramp[cell_key][induction_key] = np.subtract(exp_ramp[cell_key][induction_key]['after'],
                                                                      exp_ramp[cell_key][induction_key]['before'])
            else:
                exp_ramp[cell_key][induction_key]['before'] = \
                    np.zeros_like(exp_ramp[cell_key][induction_key]['after'])
                delta_exp_ramp[cell_key][induction_key], discard = \
                    subtract_baseline(exp_ramp[cell_key][induction_key]['after'])

            if 'mean_induction_t' not in f['data'][cell_key][induction_key]['processed']:
                induction_locs = f['data'][cell_key][induction_key].attrs['induction_locs']
                mean_induction_loc[cell_key][induction_key] = np.mean(induction_locs)
                extended_delta_exp_ramp[cell_key][induction_key] = \
                    np.concatenate([delta_exp_ramp[cell_key][induction_key]] * 3)
                for category in exp_ramp[cell_key][induction_key]:
                    extended_exp_ramp[cell_key][induction_key][category] = \
                        np.concatenate([exp_ramp[cell_key][induction_key][category]] * 3)
                if debug:
                    fig1, axes1 = plt.subplots(1)
                backward_t = np.empty_like(binned_extra_x)
                backward_t[:] = np.nan
                forward_t = np.array(backward_t)
                for i in range(len(induction_locs)):
                    this_induction_loc = mean_induction_loc[cell_key][induction_key]
                    key = str(i)
                    this_position = f['data'][cell_key][induction_key]['processed']['position']['induction'][key][:]
                    this_t = f['data'][cell_key][induction_key]['processed']['t']['induction'][key][:]
                    this_induction_index = np.where(this_position >= this_induction_loc)[0][0]
                    this_induction_t = this_t[this_induction_index]
                    if i == 0 and 'pre' in f['data'][cell_key][induction_key]['raw']['position']:
                        pre_position = f['data'][cell_key][induction_key]['processed']['position']['pre']['0'][:]
                        pre_t = f['data'][cell_key][induction_key]['processed']['t']['pre']['0'][:]
                        pre_t -= len(pre_t) * dt
                        pre_t -= this_induction_t
                        backward_t, forward_t = update_min_t_arrays(binned_extra_x, pre_t, pre_position, backward_t,
                                                                    forward_t)
                        if debug:
                            axes1.plot(np.subtract(pre_position,
                                                   track_length + mean_induction_loc[cell_key][induction_key]), pre_t,
                                       label='Lap: Pre')
                    elif i > 0:
                        prev_t -= len(prev_t) * dt
                        prev_induction_t = prev_t[prev_induction_index]
                        backward_t, forward_t = update_min_t_arrays(binned_extra_x,
                                                                    np.subtract(prev_t, this_induction_t),
                                                                    prev_position, backward_t, forward_t)
                        if debug:
                            axes1.plot(np.subtract(prev_position,
                                                   track_length + mean_induction_loc[cell_key][induction_key]),
                                       np.subtract(prev_t, this_induction_t), label='Lap: %s (Prev)' % prev_key)

                        backward_t, forward_t = update_min_t_arrays(binned_extra_x,
                                                                    np.subtract(this_t, prev_induction_t),
                                                                    this_position, backward_t, forward_t)
                        if debug:
                            axes1.plot(np.subtract(this_position,
                                                   mean_induction_loc[cell_key][induction_key] - track_length),
                                       np.subtract(this_t, prev_induction_t), label='Lap: %s (Next)' % key)
                    backward_t, forward_t = update_min_t_arrays(binned_extra_x, np.subtract(this_t, this_induction_t),
                                                                this_position, backward_t, forward_t)
                    if debug:
                        axes1.plot(np.subtract(this_position, mean_induction_loc[cell_key][induction_key]),
                                   np.subtract(this_t, this_induction_t), label='Lap: %s (Current)' % key)
                    if i == len(induction_locs) - 1 and 'post' in f['data'][cell_key][induction_key]['raw']['position']:
                        post_position = f['data'][cell_key][induction_key]['processed']['position']['post']['0'][:]
                        post_t = f['data'][cell_key][induction_key]['processed']['t']['post']['0'][:]
                        post_t += len(this_t) * dt
                        post_t -= this_induction_t
                        backward_t, forward_t = update_min_t_arrays(binned_extra_x, post_t, post_position, backward_t,
                                                                    forward_t)
                        if debug:
                            axes1.plot(np.subtract(post_position,
                                                   mean_induction_loc[cell_key][induction_key] - track_length), post_t,
                                       label='Lap: Post')
                    prev_key = key
                    prev_induction_index = this_induction_index
                    prev_t = this_t
                    prev_position = this_position
                extended_min_delta_t[cell_key][induction_key] = \
                    merge_min_t_arrays(binned_x, binned_extra_x, extended_binned_x,
                                       mean_induction_loc[cell_key][induction_key], backward_t, forward_t)
                this_extended_delta_position = np.subtract(extended_binned_x,
                                                           mean_induction_loc[cell_key][induction_key])
                if debug:
                    axes1.plot(this_extended_delta_position, extended_min_delta_t[cell_key][induction_key], c='k',
                               label='Min Interval')
                    fig1.suptitle('Cell: %s; Induction: %s' % (cell_key, induction_key),
                                  fontsize=mpl.rcParams['font.size'])
                    box = axes1.get_position()
                    axes1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    axes1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, framealpha=0.5)
                    axes1.set_xlabel('Position relative to plateau onset (cm)')
                    axes1.set_ylabel('Time relative to plateau onset (ms)')
                    clean_axes(axes1)
                    fig1.show()

                mask = ~np.isnan(extended_min_delta_t[cell_key][induction_key])
                indexes = np.where((extended_min_delta_t[cell_key][induction_key][mask] >= reference_delta_t[0]) &
                                   (extended_min_delta_t[cell_key][induction_key][mask] <= reference_delta_t[-1]))[0]
                bad_indexes = np.where((reference_delta_t <
                                        extended_min_delta_t[cell_key][induction_key][mask][indexes[0]]) |
                                       (reference_delta_t >
                                        extended_min_delta_t[cell_key][induction_key][mask][indexes[-1]]))[0]

                this_interp_delta_exp_ramp = \
                    np.interp(reference_delta_t, extended_min_delta_t[cell_key][induction_key][mask],
                              extended_delta_exp_ramp[cell_key][induction_key][mask])
                this_interp_initial_exp_ramp = \
                    np.interp(reference_delta_t, extended_min_delta_t[cell_key][induction_key][mask],
                              extended_exp_ramp[cell_key][induction_key]['before'][mask])

                if truncate and len(bad_indexes) > 0:
                    this_interp_delta_exp_ramp[bad_indexes] = np.nan
                    this_interp_initial_exp_ramp[bad_indexes] = np.nan

                if debug:
                    fig2, axes2 = plt.subplots()
                    axes2.plot(extended_min_delta_t[cell_key][induction_key],
                               extended_delta_exp_ramp[cell_key][induction_key], c='k')
                    axes2.plot(reference_delta_t, this_interp_delta_exp_ramp, c='r')
                    axes2.set_xlim(reference_delta_t[0], reference_delta_t[-1])
                    fig2.suptitle('Cell: %s; Induction: %s' % (cell_key, induction_key),
                                  fontsize=mpl.rcParams['font.size'])
                    axes2.set_xlabel('Time relative to plateau onset (ms)')
                    axes2.set_ylabel('Change in ramp\namplitude (mV)')
                    clean_axes(axes2)
                    fig2.show()

                if 'DC_soma' in f['data'][cell_key][induction_key].attrs and \
                        f['data'][cell_key][induction_key].attrs['DC_soma']:
                    if 'DC_soma_val' in f['data'][cell_key][induction_key].attrs and \
                            f['data'][cell_key][induction_key].attrs['DC_soma_val'] is not None:
                        this_induction_vm = np.ones_like(reference_delta_t) * \
                                            f['data'][cell_key][induction_key].attrs['DC_soma_val']
                    else:
                        raise RuntimeError('get_biBTSP_analysis_results: cell: %s, induction: %s; for a DC_soma '
                                           'experiments, either a DC_soma_val or an initial_induction_delta_vm array '
                                           'must be provided.' % (cell_key, induction_key))
                else:
                    this_induction_vm = np.copy(this_interp_initial_exp_ramp)
                initial_induction_delta_vm.append(this_induction_vm)
                interp_delta_exp_ramp.append(this_interp_delta_exp_ramp)
                interp_initial_exp_ramp.append(this_interp_initial_exp_ramp)
            else:
                mean_induction_t = f['data'][cell_key][induction_key]['processed']['mean_induction_t'][:]
                bad_indexes = np.where((reference_delta_t < np.min(mean_induction_t)) |
                                       (reference_delta_t > np.max(mean_induction_t)))[0]
                this_interp_delta_exp_ramp = \
                    np.interp(reference_delta_t, mean_induction_t, delta_exp_ramp[cell_key][induction_key])
                this_interp_initial_exp_ramp = \
                    np.interp(reference_delta_t, mean_induction_t, exp_ramp[cell_key][induction_key]['before'])
                if 'initial_induction_delta_vm' in f['data'][cell_key][induction_key]['processed']:
                    this_induction_vm = f['data'][cell_key][induction_key]['processed']['initial_induction_delta_vm'][:]
                    this_induction_vm = np.interp(reference_delta_t, mean_induction_t, this_induction_vm)
                elif 'DC_soma' in f['data'][cell_key][induction_key].attrs and \
                        f['data'][cell_key][induction_key].attrs['DC_soma'] and \
                        'DC_soma_val' in f['data'][cell_key][induction_key].attrs and \
                        f['data'][cell_key][induction_key].attrs['DC_soma_val'] is not None:
                    this_induction_vm = np.ones_like(reference_delta_t) * \
                                        float(f['data'][cell_key][induction_key].attrs['DC_soma_val'])
                else:
                    this_induction_vm = np.copy(this_interp_initial_exp_ramp)
                if len(bad_indexes) > 0:
                    this_interp_delta_exp_ramp[bad_indexes] = np.nan
                    this_induction_vm[bad_indexes] = np.nan
                    this_interp_initial_exp_ramp[bad_indexes] = np.nan
                interp_delta_exp_ramp.append(this_interp_delta_exp_ramp)
                interp_initial_exp_ramp.append(this_interp_initial_exp_ramp)
                initial_induction_delta_vm.append(this_induction_vm)

    return initial_induction_delta_vm, interp_initial_exp_ramp, interp_delta_exp_ramp


def get_low_pass_filtered_trace(trace, t, down_dt=0.5):
    import scipy.signal as signal
    down_t = np.arange(np.min(t), np.max(t), down_dt)
    # 2000 ms Hamming window, ~3 Hz low-pass filter
    window_len = int(2000./down_dt)
    pad_len = int(window_len / 2.)
    ramp_filter = signal.firwin(window_len, 2., nyq=1000. / 2. / down_dt)
    down_sampled = np.interp(down_t, t, trace)
    padded_trace = np.zeros(len(down_sampled) + window_len)
    padded_trace[pad_len:-pad_len] = down_sampled
    padded_trace[:pad_len] = down_sampled[::-1][-pad_len:]
    padded_trace[-pad_len:] = down_sampled[::-1][:pad_len]
    down_filtered = signal.filtfilt(ramp_filter, [1.], padded_trace, padlen=pad_len)
    down_filtered = down_filtered[pad_len:-pad_len]
    filtered = np.interp(t, down_t, down_filtered)
    return filtered