__author__ = 'milsteina'
from nested.optimize_utils import *
import click


@click.command()
@click.option("--legend-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              required=True)
@click.option("--export-file-path", type=str)
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--verbose", is_flag=True)
def main(legend_file_path, export_file_path, data_dir, verbose):
    """

    :param legend_file_path: str (path)
    :param export_file_path: str
    :param data_dir: str (dir path)
    :param verbose: bool
    """
    if not os.path.isdir(data_dir):
        raise IOError('Invalid data_dir: %s' % data_dir)
    model_params = dict()
    legend = read_from_yaml(legend_file_path)
    for key, file_name in legend.items():
        file_path = '%s/%s' % (data_dir, file_name)
        report = OptimizationReport(file_path=file_path)
        model_params[key] = param_array_to_dict(report.survivors[0].x, report.param_names)
    if verbose:
        pprint.pprint(model_params)
        sys.stdout.flush()
    write_to_yaml(export_file_path, model_params, convert_scalars=True)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
