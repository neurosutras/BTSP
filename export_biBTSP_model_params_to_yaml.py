__author__ = 'milsteina'
from nested.optimize_utils import *
import click


@click.command()
@click.option("--model-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='data/20190812_biBTSP_SRL_B_90cm_all_cells_merged_exported_model_output.hdf5')
@click.option("--export-file-name", type=str)
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
def main(model_file_path, export_file_name, output_dir):
    """

    :param model_file_path: str (path)
    :param export_file_name: str
    :param output_dir: str (dir path)
    """
    if not os.path.isfile(model_file_path):
        raise IOError('Invalid model_file_path: %s' % model_file_path)
    if not os.path.isdir(output_dir):
        raise IOError('Invalid output_dir: %s' % output_dir)
    export_file_path = '%s/%s' % (output_dir, export_file_name)
    model_params = dict()
    with h5py.File(model_file_path, 'r') as f:
        shared_context_key = 'shared_context'
        group = f[shared_context_key]
        param_names = np.array(group['param_names'][:], dtype='str')
        exported_data_key = 'exported_data'
        for cell_key in f[exported_data_key]:
            description = 'model_ramp_features'
            group = next(iter(viewvalues(f[exported_data_key][cell_key])))[description]
            param_array = group['param_array'][:]
            param_dict = param_array_to_dict(param_array, param_names)
            model_params[int(cell_key)] = dict()
            model_params[int(cell_key)].update(param_dict)
    pprint.pprint(model_params)
    write_to_yaml(export_file_path, model_params, convert_scalars=True)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
