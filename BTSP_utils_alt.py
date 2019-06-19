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
                if hasattr(r, '__iter__'):
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
            for s1, r in rates[s0].iteritems():
                self.update_transition(s0, s1, r)

    def update_states(self, states):
        """

        :param states: dict
        """
        for s, v in states.iteritems():
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
            if hasattr(r, '__iter__'):
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
                    if hasattr(r, '__iter__'):
                        if len(r) - 1 < self.i:
                            raise Exception('StateMachine: Insufficient array length for non-stationary rate: %s to '
                                            '%s ' % (s0, s1))
                        this_r = r[self.i]
                    else:
                        this_r = r
                    # print 'this_r: %.4E, factor: %.4E, %s: %.4E' % (this_r, factor, s0, self.states[s0])
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
                if hasattr(r, '__iter__'):
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
        elif not hasattr(states, '__iter__'):
            states = [states]
        fig, axes = plt.subplots(1)
        for state in states:
            if state in self.states:
                axes.plot(self.t_history, self.states_history[state], label=state)
            else:
                print 'StateMachine: Not including invalid state: %s' % state
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
    peak_locs = default_interp_x[peak_indexes]
    if not np.any(peak_locs):
        return None, None
    peak_shifts = []

    for peak_loc in peak_locs:
        peak_shift = peak_loc - induction_loc
        if peak_shift > track_length / 2.:
            peak_shift = -(track_length - peak_shift)
        elif peak_shift < -track_length / 2.:
            peak_shift += track_length
        peak_shifts.append(peak_shift)

    peak_shifts = np.array(peak_shifts)
    indexes = range(len(peak_shifts))
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
        print 'exp: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, peak_loc: %.1f, ' \
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
    Err += ((model_val_at_target_min_loc - min_val['target']) / target_range['delta_min_val']) ** 2.
    Err += ((min_val['model'] - min_val['target']) / target_range['delta_min_val']) ** 2.
    model_val_at_target_peak_loc = model_ramp[peak_index]
    Err += ((model_val_at_target_peak_loc - ramp_amp['target']) / target_range['delta_peak_val']) ** 2.

    for i in xrange(len(exp_ramp)):
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
    :param target_weights_width: float  (legacy target ramp width is 108 cm (90 * 1.2), to produce a firing field of width 90)
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
    force_delta_weights = tuning_amp * np.cos(2. * np.pi / target_weights_width * (peak_locs - modulated_field_center)) + \
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
        print 'target: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, ' \
              'peak_loc: %.1f, end_loc: %.1f' % (ramp_amp['target'], ramp_width['target'], peak_shift['target'],
                                                 ratio['target'], start_loc['target'], peak_loc['target'],
                                                 end_loc['target'])
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


def get_weights_LSA_scaling_factor(ramp_x, input_x, interp_x, input_rate_maps, peak_locs, target_ramp, track_length,
                                   target_range, bounds, beta=2., initial_ramp_scaling_factor=0.002956,
                                   target_peak_delta_weight=1.5, plot=False, verbose=1):
    """
    Reality check for the least square approximation method to estimate weights. Used to calibrate ramp_scaling_factor.
    Given a set of input_rate_maps and a target_ramp, computes and a set of weights with the target_peak_delta_weight.
    Returns the weights and a new ramp_scaling_factor.
    :param ramp_x: array (spatial resolution of ramp)
    :param input_x: array (spatial resolution of input_rate_maps)
    :param interp_x: array (spatial resolution for computing fine features)
    :param input_rate_maps: array
    :param peak_locs: array
    :param target_ramp: array
    :param track_length: float
    :param target_range: dict
    :param bounds: tuple of float
    :param beta: float; regularization parameter
    :param initial_ramp_scaling_factor: float
    :param target_peak_delta_weight: float
    :param target_peak_ramp_amp: float (mV)
    :param plot: bool
    :param verbose: int
    :return: tuple of array
    """
    target_peak_ramp_amp = np.max(target_ramp)
    if len(target_ramp) != len(ramp_x):
        target_ramp = np.interp(ramp_x, input_x, target_ramp)
    modulated_field_center = ramp_x[np.argmax(target_ramp)]
    induction_start_loc = modulated_field_center + 10.
    induction_stop_loc = induction_start_loc + 5.

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
        get_adjusted_delta_weights_and_ramp_scaling_factor(delta_weights, input_rate_maps, target_peak_delta_weight,
                                                           target_peak_ramp_amp)
    input_matrix = np.multiply(input_rate_maps, SVD_ramp_scaling_factor)
    SVD_model_ramp = SVD_delta_weights.dot(input_matrix)

    result = minimize(get_residual_score, SVD_delta_weights,
                      args=(target_ramp, ramp_x, input_x, interp_x, input_rate_maps, SVD_ramp_scaling_factor,
                            induction_start_loc, track_length, target_range, bounds), method='L-BFGS-B',
                      bounds=[bounds] * len(SVD_delta_weights), options={'disp': verbose > 1, 'maxiter': 100})

    LSA_delta_weights = result.x
    delta_weights, ramp_scaling_factor = \
        get_adjusted_delta_weights_and_ramp_scaling_factor(LSA_delta_weights, input_rate_maps, target_peak_delta_weight,
                                                           target_peak_ramp_amp)

    input_matrix = np.multiply(input_rate_maps, ramp_scaling_factor)
    model_ramp = delta_weights.dot(input_matrix)

    if len(model_ramp) != len(ramp_x):
        model_ramp = np.interp(ramp_x, input_x, model_ramp)
    ramp_amp['model'], ramp_width['model'], peak_shift['model'], ratio['model'], start_loc['model'], \
    peak_loc['model'], end_loc['model'], min_val['model'], min_loc['model'] = \
        calculate_ramp_features(ramp=model_ramp, induction_loc=induction_start_loc, binned_x=ramp_x, interp_x=interp_x,
                                track_length=track_length)

    if verbose > 1:
        print 'target: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, ' \
              'peak_loc: %.1f, end_loc: %.1f' % (ramp_amp['target'], ramp_width['target'], peak_shift['target'],
                                                 ratio['target'], start_loc['target'], peak_loc['target'],
                                                 end_loc['target'])
        print 'model: amp: %.1f, ramp_width: %.1f, peak_shift: %.1f, asymmetry: %.1f, start_loc: %.1f, peak_loc: %.1f' \
              ', end_loc: %.1f' % (ramp_amp['model'], ramp_width['model'], peak_shift['model'], ratio['model'],
                                   start_loc['model'], peak_loc['model'], end_loc['model'])
        print 'ramp_scaling_factor: %.6f' % ramp_scaling_factor

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

    return target_ramp, model_ramp, delta_weights, ramp_scaling_factor


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

    initial_ramp = initial_delta_weights.dot(input_matrix)

    result = minimize(get_residual_score, initial_delta_weights,
                      args=(target_ramp, ramp_x, input_x, interp_x, input_rate_maps, ramp_scaling_factor,
                            induction_start_loc, track_length, target_range, bounds, allow_offset, impose_offset),
                      method='L-BFGS-B', bounds=[bounds] * len(initial_delta_weights),
                      options={'disp': verbose > 1, 'maxiter': 100})

    if verbose > 1:
        print 'get_delta_weights_LSA: process: %i; %s:' % (os.getpid(), label)
    model_ramp, delta_weights, ramp_offset, residual_score = \
        get_residual_score(result.x, target_ramp, ramp_x, input_x, interp_x, input_rate_maps, ramp_scaling_factor,
                           induction_start_loc, track_length, target_range, bounds, allow_offset, impose_offset,
                           disp=verbose > 1, full_output=True)

    if plot:
        x_start = induction_start_loc
        x_end = induction_stop_loc
        ylim = max(np.max(target_ramp), np.max(model_ramp))
        ymin = min(np.min(target_ramp), np.min(model_ramp))
        fig, axes = plt.subplots(1,2)
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
