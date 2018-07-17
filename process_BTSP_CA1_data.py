__author__ = 'Aaron D. Milstein'
from BTSP_utils import *
import click
import pandas as pd
# from plot_results import *


"""
Magee lab CA1 BTSP2 place field data is in a series of text files. This organizes them into a single .hdf5 file with a 
standard format:

defaults:
    attrs: default scalars
    default position and time arrays
    input_rate_maps (200)
    peak_locs (200)
data:
    cell_id:
        induction:
            attrs:
                induction_locs
                induction_durs
            raw:
                attrs:
                    sampling_rate
                position:
                    pre: (if available)
                    induction:
                    post: (if available)
                t:
                    pre: (if available)
                    induction:
                    post: (if available)
                current:
                exp_ramp:
                    before (if available)
                    after
            processed:
                position:
                    pre:
                    induction:
                    post:
                t:
                    pre:
                    induction:
                    post:
                current:
                exp_ramp:
                    before (if available)
                    after
                exp_ramp_vs_t: (interpolated against mean_t)
                    before (if available)
                    after
                mean_position
                mean_t
            complete:
                run_vel
                run_vel_gate
                position
                t
                induction_gate
"""

context = Context()


@click.command()
@click.option("--cell-id", type=int, default=1)
@click.option("--induction", type=int, default=1)
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/process_BTSP_CA1_data_config.yaml')
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='data')
@click.option("--plot", type=int, default=1)
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=click.Path(exists=False, file_okay=True, dir_okay=False))
def main(cell_id, induction, config_file_path, data_dir, plot, export, export_file_path):
    """

    :param cell_id: int
    :param induction: int
    :param config_file_path: str (path)
    :param data_dir: str
    :param plot: bool
    :param export: bool
    :param export_file_path: str (path)
    """
    context.update(locals())
    initialize()
    print 'Processing cell_id: %i, induction: %i' % (cell_id, induction)
    if export_file_path is None:
        export_file_path = context.data_dir + '/%s_BTSP_CA1_data.hdf5' % \
                                              datetime.datetime.today().strftime('%Y%m%d_%H%M')
    raw_position = {}
    position = {}
    raw_current = []
    current = []
    induction_locs = []
    induction_durs = []
    raw_t = {}
    t = {}

    if plot>1:
        fig, axes = plt.subplots(2)
    for group in context.position_laps:
        for i, p in enumerate(context.position_laps[group]):
            if group not in position:
                raw_position[group] = []
                position[group] = []
                raw_t[group] = []
                t[group] = []
            this_position = context.data[p] * context.position_scale
            this_raw_t = np.arange(0., len(this_position) * context.position_dt,
                                   context.position_dt)[:len(this_position)]
            raw_t[group].append(this_raw_t)
            raw_position[group].append(this_position)
            if plot>1:
                axes[0].plot(this_raw_t / 1000., this_position)
            this_position = this_position / np.max(this_position) * context.track_length
            this_dur = len(this_position) * context.position_dt
            this_t = np.arange(0., this_dur + context.dt / 2., context.dt)[:len(this_position)]
            t[group].append(this_t)
            this_position = np.interp(this_t, this_raw_t, this_position)
            position[group].append(this_position)
            if plot>1:
                axes[1].plot(this_t / 1000., this_position, label=group+str(i))
    if plot>1:
        axes[0].set_ylabel('Position (cm)')
        axes[0].set_title('Raw position')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Position (cm)')
        axes[1].set_title('Interpolated position')
        axes[1].legend(loc='best', frameon=False, framealpha=0.5)
        fig.tight_layout()
        clean_axes(axes)
        plt.show()
        plt.close()

    vel_window_bins = int(context.vel_window / context.dt) / 2
    for count in xrange(2):
        complete_run_vel = np.array([])
        complete_position = np.array([])
        complete_t = np.array([])
        running_dur = 0.
        running_length = 0.
        for group in (group for group in ['pre', 'induction', 'post'] if group in position):
            for this_position, this_t in zip(position[group], t[group]):
                complete_position = np.append(complete_position, np.add(this_position, running_length))
                complete_t = np.append(complete_t, np.add(this_t, running_dur))
                running_length += context.track_length
                running_dur += len(this_t) * context.dt

        for i in range(len(complete_position)):
            indexes = range(i - vel_window_bins, i + vel_window_bins + 1)
            this_position_window = np.sum(np.diff(complete_position.take(indexes, mode='wrap')))
            this_t_window = np.sum(np.diff(complete_t.take(indexes, mode='wrap')))
            complete_run_vel = np.append(complete_run_vel, np.divide(this_position_window, this_t_window) * 1000.)
        if count == 0 and plot>1:
            fig, axes = plt.subplots(1)
            axes.plot(complete_t / 1000., complete_run_vel)
            axes.set_xlabel('Time (s)')
            axes.set_ylabel('Running speed (cm/s)')
            clean_axes(axes)
            plt.show()
            plt.close()

        if count == 0:
            delta_t_list = []
            for group in (group for group in ['pre', 'induction', 'post'] if group in position):
                for i, this_position in enumerate(position[group]):
                    this_delta_t = np.interp(context.default_interp_x, this_position, t[group][i])
                    this_delta_t = np.diff(this_delta_t)
                    delta_t_list.append(this_delta_t)
            mean_delta_t = np.mean(delta_t_list, axis=0)

        if count == 0:
            temp_t = np.append(0., np.cumsum(mean_delta_t))
            mean_t = np.arange(0., temp_t[-1] + context.dt / 2., context.dt)
            mean_position = np.interp(mean_t, temp_t, context.default_interp_x)
            if 'pre' not in position:
                position['pre'] = [mean_position]
                t['pre'] = [mean_t]
            if 'post' not in position:
                position['post'] = [mean_position]
                t['post'] = [mean_t]
        else:
            for i in xrange(len(t['pre'])):
                complete_t -= len(t['pre'][i]) * context.dt
                complete_position -= context.track_length
            if plot>1:
                fig, axes = plt.subplots(1)
                axes.plot(complete_t / 1000., complete_position)
                axes.set_xlabel('Time (s)')
                axes.set_ylabel('Position (cm)')
                clean_axes(axes)
                fig.tight_layout()
                plt.show()
                plt.close()
            break

    complete_run_vel_gate = np.ones_like(complete_run_vel)
    complete_run_vel_gate[np.where(complete_run_vel <= 5.)[0]] = 0.
    if plot>1:
        fig, axes = plt.subplots(1)
        axes2 = axes.twinx()
        axes.plot(complete_t / 1000., complete_run_vel)
        axes2.plot(complete_t / 1000., complete_run_vel_gate, c='k')
        axes.set_xlabel('Time (s)')
        axes.set_ylabel('Running speed (cm/s)')
        clean_axes(axes)
        axes2.tick_params(direction='out')
        axes2.spines['top'].set_visible(False)
        axes2.spines['left'].set_visible(False)
        axes2.get_xaxis().tick_bottom()
        axes2.get_yaxis().tick_right()
        fig.tight_layout()
        plt.show()
        plt.close()

    exp_ramp_raw = {'after': context.data[context.ramp_laps['after']][:100] * context.ramp_scale}
    exp_ramp = {'after': signal.savgol_filter(exp_ramp_raw['after'], 21, 3, mode='wrap')}
    exp_ramp_vs_t = {}
    if 'before' in context.ramp_laps:
        exp_ramp_raw['before'] = context.data[context.ramp_laps['before']][:100] * context.ramp_scale
        exp_ramp['before'] = signal.savgol_filter(exp_ramp_raw['before'], 21, 3, mode='wrap')
        ramp_baseline = np.min(exp_ramp['before'])
        exp_ramp_raw['before'] -= ramp_baseline
        exp_ramp['before'] -= ramp_baseline
        exp_ramp_vs_t['before'] = np.interp(mean_position, context.binned_x, exp_ramp['before'])
    else:
        ramp_baseline = np.min(exp_ramp['after'])
    exp_ramp_raw['after'] -= ramp_baseline
    exp_ramp['after'] -= ramp_baseline
    exp_ramp_vs_t['after'] = np.interp(mean_position, context.binned_x, exp_ramp['after'])

    induction_gate = np.array([])
    if plot>0:
        fig, axes = plt.subplots(2, 2)
        axes[1][0].plot(context.binned_x, exp_ramp['after'])
        if 'before' in context.ramp_laps:
            axes[1][0].plot(context.binned_x, exp_ramp['before'])
            axes[1][0].plot(context.binned_x, exp_ramp_raw['before'])
        axes[1][0].plot(context.binned_x, exp_ramp_raw['after'])
        axes[1][0].set_xlabel('Position (cm)')
        axes[0][0].set_xlabel('Position (cm)')
        axes[1][0].set_ylabel('Ramp amplitude (mV)')
        axes[1][1].set_ylabel('Ramp amplitude (mV)')
        axes[1][1].set_xlabel('Time (s)')
        axes[0][1].set_xlabel('Time (s)')
        axes[0][0].set_ylabel('Induction current (nA)')
        axes[0][1].set_ylabel('Induction gate (a.u.)')
    for i in xrange(len(position['pre'])):
        induction_gate = np.append(induction_gate, np.zeros_like(position['pre'][i]))
    for i, this_position in enumerate(position['induction']):
        c = context.current_laps[i]
        this_current = context.data[c]
        raw_current.append(this_current)
        this_raw_t = raw_t['induction'][i]
        this_t = t['induction'][i]
        this_current = np.interp(this_t, this_raw_t, this_current)
        this_current, current_baseline = subtract_baseline(this_current)
        current.append(this_current)
        this_induction_gate = np.zeros_like(this_current)
        indexes = np.where(this_current >= 0.5 * np.max(this_current))[0]
        this_induction_gate[indexes] = 1.
        start_index = indexes[0]
        this_induction_loc = this_position[start_index]
        induction_locs.append(this_induction_loc)
        this_induction_dur = len(indexes) * context.dt
        induction_durs.append(this_induction_dur)
        induction_gate = np.append(induction_gate, this_induction_gate)
        if plot:
            axes[0][0].plot(this_position, this_current, label='Lap %i: Loc: %i cm, Dur: %i ms' %
                                                               (i, this_induction_loc, this_induction_dur))
            axes[0][1].plot(np.subtract(this_t, this_t[start_index]) / 1000., this_induction_gate)
    for i in xrange(len(position['post'])):
        induction_gate = np.append(induction_gate, np.zeros_like(position['post'][i]))
    mean_induction_loc = np.mean(induction_locs)
    mean_induction_index = np.where(mean_position >= mean_induction_loc)[0][0]
    mean_induction_onset = mean_t[mean_induction_index]
    context.update(locals())

    if plot>0:
        peak_val, ramp_width, peak_shift, ratio, start_loc, peak_loc, end_loc, min_val, min_loc = \
            calculate_ramp_features(context,exp_ramp['after'], mean_induction_loc)
        start_index, peak_index, end_index, min_index = \
            get_indexes_from_ramp_bounds_with_wrap(context.binned_x, start_loc, peak_loc, end_loc, min_loc)
        axes[1][0].scatter(context.binned_x[[start_index, peak_index, end_index]],
                           exp_ramp['after'][[start_index, peak_index, end_index]])
        start_index, peak_index, end_index, min_index = \
            get_indexes_from_ramp_bounds_with_wrap(mean_position, start_loc, peak_loc, end_loc, min_loc)
        this_shifted_t = np.subtract(mean_t, mean_induction_onset) / 1000.
        axes[1][1].plot(this_shifted_t, exp_ramp_vs_t['after'])
        axes[1][1].scatter(this_shifted_t[[start_index, peak_index, end_index]],
                           exp_ramp_vs_t['after'][[start_index, peak_index, end_index]])
        if 'before' in context.ramp_laps:
            axes[1][1].plot(this_shifted_t, exp_ramp_vs_t['before'])
        axes[0][0].legend(loc='best', frameon=False, framealpha=0.5)
        clean_axes(axes)
        fig.tight_layout()
        plt.show()
        plt.close()

    if export:
        export_data()


def initialize():
    """

    """
    dt = 1.  # ms
    input_field_width = 90.  # cm
    input_field_peak_rate = 40.  # Hz
    num_inputs = 200  # 200
    track_length = 187.  # cm

    binned_dx = track_length / 100.  # cm
    binned_x = np.arange(0., track_length+binned_dx/2., binned_dx)[:100] + binned_dx/2.
    generic_dx = binned_dx / 100.  # cm
    generic_x = np.arange(0., track_length, generic_dx)

    default_run_vel = 30.  # cm/s
    generic_position_dt = generic_dx / default_run_vel * 1000.  # ms
    generic_t = np.arange(0., len(generic_x)*generic_position_dt, generic_position_dt)[:len(generic_x)]

    default_interp_t = np.arange(0., generic_t[-1], dt)
    default_interp_x = np.interp(default_interp_t, generic_t, generic_x)
    default_interp_dx = dt * default_run_vel / 1000.  # cm

    extended_x = np.concatenate([generic_x - track_length, generic_x, generic_x + track_length])

    meta_data = read_from_yaml(context.config_file_path)

    if context.cell_id not in meta_data or context.induction not in meta_data[context.cell_id]:
        raise Exception('Cannot process cell_id: %i, induction: %i' % (context.cell_id, context.induction))

    file_path = context.data_dir + '/BTSP/' + meta_data[context.cell_id][context.induction]['file_name']
    df = pd.read_table(file_path, sep='\t', header=0)

    data = {}
    for c in range(len(df.values[0, :])):
        data[c] = df.values[:, c][~np.isnan(df.values[:, c])]

    sampling_rate = meta_data[context.cell_id][context.induction]['sampling_rate']  # Hz
    position_dt = 1000./sampling_rate
    ramp_scale = 1000.
    position_scale = 10.
    vel_window = 10.  # ms
    # generates a predicted 6 mV depolarization given peak_delta_weights = 1.5
    ramp_scaling_factor = 2.956E-03

    # True for spontaneously-occurring field, False for or induced field
    spont = meta_data[context.cell_id][context.induction]['spont']
    position_laps = meta_data[context.cell_id][context.induction]['position_laps']
    current_laps = meta_data[context.cell_id][context.induction]['current_laps']
    ramp_laps = meta_data[context.cell_id][context.induction]['ramp_laps']

    context.update(locals())
    context.spatial_rate_maps, context.peak_locs = generate_spatial_rate_maps(num_inputs, input_field_peak_rate,
                                                                              input_field_width, track_length, binned_x)


def export_data(export_file_path=None):
    """

    :param export_file_path: str (path)
    """
    if export_file_path is None:
        export_file_path = context.export_file_path
    with h5py.File(export_file_path, 'a') as f:
        if 'defaults' not in f:
            f.create_group('defaults')
            f['defaults'].attrs['dt'] = context.dt  # ms
            f['defaults'].attrs['input_field_width'] = context.input_field_width  # cm
            f['defaults'].attrs['input_field_peak_rate'] = context.input_field_peak_rate  # Hz
            f['defaults'].attrs['num_inputs'] = context.num_inputs
            f['defaults'].attrs['track_length'] = context.track_length  # cm
            f['defaults'].attrs['binned_dx'] = context.binned_dx  # cm
            f['defaults'].attrs['generic_dx'] = context.generic_dx  # cm
            f['defaults'].attrs['default_run_vel'] = context.default_run_vel  # cm/s
            f['defaults'].attrs['generic_position_dt'] = context.generic_position_dt  # ms
            f['defaults'].attrs['default_interp_dx'] = context.default_interp_dx  # cm
            f['defaults'].attrs['ramp_scaling_factor'] = context.ramp_scaling_factor
            f['defaults'].create_dataset('binned_x', compression='gzip', compression_opts=9, data=context.binned_x)
            f['defaults'].create_dataset('generic_x', compression='gzip', compression_opts=9, data=context.generic_x)
            f['defaults'].create_dataset('generic_t', compression='gzip', compression_opts=9, data=context.generic_t)
            f['defaults'].create_dataset('default_interp_t', compression='gzip', compression_opts=9,
                                         data=context.default_interp_t)
            f['defaults'].create_dataset('default_interp_x', compression='gzip', compression_opts=9,
                                         data=context.default_interp_x)
            f['defaults'].create_dataset('extended_x', compression='gzip', compression_opts=9, data=context.extended_x)
            f['defaults'].create_dataset('input_rate_maps', compression='gzip', compression_opts=9,
                                         data=context.spatial_rate_maps)
            f['defaults'].create_dataset('peak_locs', compression='gzip', compression_opts=9, data=context.peak_locs)
        if 'data' not in f:
            f.create_group('data')
        cell_key = str(context.cell_id)
        induction_key = str(context.induction)
        if cell_key not in f['data']:
            f['data'].create_group(cell_key)
        if induction_key not in f['data'][cell_key]:
            f['data'][cell_key].create_group(induction_key)
        f['data'][cell_key].attrs['spont'] = context.spont
        this_group = f['data'][cell_key][induction_key]
        this_group.attrs['induction_locs'] = context.induction_locs
        this_group.attrs['induction_durs'] = context.induction_durs
        this_group.create_group('raw')
        this_group['raw'].attrs['sampling_rate'] = context.sampling_rate
        this_group['raw'].create_group('position')
        this_group['raw'].create_group('t')
        this_group['raw'].create_group('current')
        for category in context.raw_position:
            this_group['raw']['position'].create_group(category)
            this_group['raw']['t'].create_group(category)
            for i in xrange(len(context.raw_position[category])):
                lap_key = str(i)
                this_group['raw']['position'][category].create_dataset(lap_key, compression='gzip', compression_opts=9, 
                                                                       data=context.raw_position[category][i])
                this_group['raw']['t'][category].create_dataset(lap_key, compression='gzip', compression_opts=9, 
                                                                data=context.raw_t[category][i])
        for i in xrange(len(context.raw_current)):
            lap_key = str(i)
            this_group['raw']['current'].create_dataset(lap_key, compression='gzip', compression_opts=9, 
                                                        data=context.raw_current[i])
        this_group['raw'].create_group('exp_ramp')
        this_group['raw']['exp_ramp'].create_dataset('after', compression='gzip', compression_opts=9, 
                                                     data=context.exp_ramp_raw['after'])
        if 'before' in context.ramp_laps:
            this_group['raw']['exp_ramp'].create_dataset('before', compression='gzip', compression_opts=9,
                                                         data=context.exp_ramp_raw['before'])
        this_group.create_group('processed')
        this_group['processed'].create_group('position')
        this_group['processed'].create_group('t')
        this_group['processed'].create_group('current')
        for category in context.position:
            this_group['processed']['position'].create_group(category)
            this_group['processed']['t'].create_group(category)
            for i in xrange(len(context.position[category])):
                lap_key = str(i)
                this_group['processed']['position'][category].create_dataset(lap_key, compression='gzip', 
                                                                             compression_opts=9, 
                                                                             data=context.position[category][i])
                this_group['processed']['t'][category].create_dataset(lap_key, compression='gzip', compression_opts=9, 
                                                                      data=context.t[category][i])
        this_group['processed'].create_dataset('mean_position', compression='gzip', compression_opts=9,
                                               data=context.mean_position)
        this_group['processed'].create_dataset('mean_t', compression='gzip', compression_opts=9, data=context.mean_t)
        for i in xrange(len(context.current)):
            lap_key = str(i)
            this_group['processed']['current'].create_dataset(lap_key, compression='gzip', compression_opts=9, 
                                                              data=context.current[i])
        this_group['processed'].create_group('exp_ramp')
        this_group['processed']['exp_ramp'].create_dataset('after', compression='gzip', compression_opts=9,
                                                           data=context.exp_ramp['after'])
        this_group['processed'].create_group('exp_ramp_vs_t')
        this_group['processed']['exp_ramp_vs_t'].create_dataset('after', compression='gzip', compression_opts=9,
                                                           data=context.exp_ramp_vs_t['after'])
        if 'before' in context.ramp_laps:
            this_group['processed']['exp_ramp'].create_dataset('before', compression='gzip', compression_opts=9,
                                                               data=context.exp_ramp['before'])
            this_group['processed']['exp_ramp_vs_t'].create_dataset('before', compression='gzip', compression_opts=9,
                                                                    data=context.exp_ramp_vs_t['before'])
        this_group.create_group('complete')
        this_group['complete'].create_dataset('run_vel', compression='gzip', compression_opts=9,
                                              data=context.complete_run_vel)
        this_group['complete'].create_dataset('run_vel_gate', compression='gzip', compression_opts=9,
                                              data=context.complete_run_vel_gate)
        this_group['complete'].create_dataset('position', compression='gzip', compression_opts=9,
                                              data=context.complete_position)
        this_group['complete'].create_dataset('t', compression='gzip', compression_opts=9,
                                              data=context.complete_t)
        this_group['complete'].create_dataset('induction_gate', compression='gzip', compression_opts=9,
                                              data=context.induction_gate)
    print 'Exported data for cell: %i, induction: %i to %s' % (context.cell_id, context.induction,
                                                               context.export_file_path)


def generate_spatial_rate_maps(n=200, peak_rate=1., field_width=90., track_length=187., x=None):
    """
    Return a list of spatial rate maps with peak locations that span the track. Return firing rate vs. location
    computed at the resolution of the provided x array. Default is track_length/10000 bins.
    :param n: int
    :param peak_rate: float
    :param field_width: float
    :param track_length: float
    :param x: array
    :return: list of array, array
    """
    if x is None:
        x = context.generic_x
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


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1,sys.argv)+1):])