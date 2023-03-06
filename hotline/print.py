from tabulate import tabulate
from IPython import embed

from hotline.hotline import *


def percent_to_str(percent):
  return f'{percent:.2f}'.rstrip('0').rstrip('.') + '%'


def print_op_in_tree(op, level=None, **kwargs):
  print(''.join(['    '] * level) + f'{op.get("idx", "")} {op.get("name", "")} {op.get("type", "")}')


def print_op_in_tree_with_runtime(op, level=None, top_level_only=False, **kwargs):
  if op['type'] == 'root':
    return
  if top_level_only and level != 0:
    return
  level_spaces = ''.join(['    '] * level)
  name = op_name(op)
  runtime_str = ''
  for res_name, res in op['resources'].items():
    if not res['time']['parent_is_longest']:
      continue  # only print resources that are the longest straggler
    percent = percent_to_str(res['time']['relative_dur']*100)
    if percent == '0%' and res['time']['relative_dur'] != 0:
      percent = '<0.01%'
    runtime_str += f'{res_name} {percent}, '
  runtime_str = runtime_str[0:-2]  # remove trailing space and comma added by loop
  print(level_spaces + ' (' + runtime_str + ') ' + name)


def op_name(op):
  op_type = op["type"] if op["type"] != op["name"] else ""
  return f'{op["name"]} {op_type}'


def op_idx_name(op):
  return f'{op.get("idx")} {op_name(op)}'


def print_interest_results_table(df):
    log.info("Table of most interesting results results:")
    columns_to_include = [
      'metadata.model',
      'metadata.dataset',
      'config.num_gpus',
      'trace_event_count',
      'trace_disk_size',
      'runtime_without_profiling',
      'runtime_with_profiling',
      'runtime_profiling_overhead_factor',
      'hotline_analysis_time',
      'hotline_annotation_count',
      'total_accuracy_str',
    ]
    headers = [col.replace('_', '_\n') for col in columns_to_include]  # Make headers with _ span multiple lines to make the table more compact.
    headers = [col.replace('.', '.\n') for col in headers]  # Make headers with . span multiple lines to make the table more compact.
    df_subset = df.reindex(columns=columns_to_include)
    print(tabulate(df_subset, headers=headers, tablefmt='orgtbl', maxcolwidths=13, stralign="right"))
