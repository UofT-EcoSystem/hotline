import sys
import orjson
import io

from IPython import embed
from tabulate import tabulate

from hotline.hotline import *
import hotline.perfetto as h_perfetto
import hotline.annotate as h_annotate
import hotline.print as h_print


# Uninteresting slices
IGNORE_LIST = [
  # Ignored because they are PyTorch internals
  'aten::zero_',
  'aten::zeros',
  'aten::empty',
  # Ideally we would not ignore these, but they are Python built-ins that are hard to detect
  'aten::view',
  'aten::transpose',
  'aten::as_strided',
]


def is_interesting_slice(slice):
  return slice['name'] not in IGNORE_LIST


def remove_ignore_insignficant_slices(slices):
  return [slice for slice in slices if is_interesting_slice(slice)]


def calc_accuracy(diff_count, manual_count, detected_count):
  try:
    accuracy = 100 - (abs(diff_count) / max(manual_count, detected_count) * 100)
    accuracy = max(accuracy, 0)  # Don't allow negative accuracy in the case where the two compared ops have different names and none of them match1
    return accuracy
  except ZeroDivisionError:
    return 0


def count_num_per_unique_name(slices):
  slices = remove_ignore_insignficant_slices(slices)
  counts = {}
  for slice in slices:
    name = slice['name']
    if name not in counts:
      counts[name] = 1
    else:
      counts[name] += 1
  return counts


def percent_diff_for_each_key(d1, d2):
  d = {}
  for key in d1:
    try:
      diff = abs(d1[key] - d2[key])
      d[key] = calc_accuracy(diff, d1[key], d2[key])
    except:
      d[key] = 0  # 0% accuracy if not found
  for key in d2:
    if key not in d1:
      d[key] = 0  # 0% accuracy if not found
  return d


def count_diff_for_each_key(d1, d2):
  d = {}
  for key in d1:
    d[key] = abs(d1[key] - d2.get(key, 0))
  for key in d2:
    if key not in d1:
      d[key] = abs(-d2[key])
  return d


def print_accuracy_table(name, detected_counts, manual_counts, count_diff, percent_diff, diff_count_total, manual_count_total, detected_count_total):
  # Print accuracy table for this op
  table = []
  for key in manual_counts:
    row_name = key[0:33]  # each is an event name, we only use the first 33 characters to fit table on screen
    table.append([row_name, manual_counts.get(key, 0), detected_counts.get(key, 0), count_diff[key], h_print.percent_to_str(percent_diff[key])])
  for key in detected_counts:
    if key not in manual_counts:
      row_name = key[0:33]  # each is an event name, we only use the first 33 characters to fit table on screen
      table.append([row_name, manual_counts.get(key, 0), detected_counts.get(key, 0), count_diff[key], h_print.percent_to_str(percent_diff[key])])

  diff_percent_total = calc_accuracy(diff_count_total, manual_count_total, detected_count_total)
  table.append(['Total', manual_count_total, detected_count_total, diff_count_total, h_print.percent_to_str(diff_percent_total)])
  # log.info(f'\n\nAccuracy: {name}')
  # print(tabulate(table, headers=['name', 'manual', 'detected', 'diff', 'accuracy']))


def check_accuracy_by_name_count(op, detected_slices, manual_slices):
  manual_counts = count_num_per_unique_name(manual_slices)
  detected_counts = count_num_per_unique_name(detected_slices)

  percent_diff = percent_diff_for_each_key(manual_counts, detected_counts)
  count_diff = count_diff_for_each_key(manual_counts, detected_counts)

  diff_count_total = sum([abs(x) for x in count_diff.values()])
  manual_count_total = sum([abs(x) for x in manual_counts.values()])
  detected_count_total = sum([abs(x) for x in detected_counts.values()])

  name = f'{op["name"]} {op.get("idx", "")}'
  print_accuracy_table(name, detected_counts, manual_counts, count_diff, percent_diff, diff_count_total, manual_count_total, detected_count_total)

  return diff_count_total, manual_count_total, detected_count_total


def print_manual_to_detected_mapping_table(tp_manual, idx_to_op_map):
  """Example output:
      idx     manual slice        detected op (name type idx)
      ------  ------------------  -----------------------------
      0
      1       hid=1 dataload      Load Data training loop 1
      2       hid=2 forward       Forward training loop 2
      3       hid=3 DataParallel  DataParallel DataParallel 3
      99.37%  158                 159
  """
  if not idx_to_op_map:
    return
  table = []
  detected_count = 0
  manual_count = 0
  for idx in range(max(idx_to_op_map.keys())):
    detected_op = idx_to_op_map.get(idx, {})
    detected_name = f'{detected_op.get("name", "")} {detected_op.get("type", "")} {detected_op.get("idx", "")}'
    manual_name = h_annotate.get_manual_annotation_by_idx(tp_manual, idx).get('name')
    detected_count += 1 if detected_name else 0
    manual_count += 1 if manual_name else 0
    table.append([idx, manual_name, detected_name])

  accuracy = 100 - calc_accuracy(min(detected_count, manual_count),detected_count, manual_count)
  accuracy = h_print.percent_to_str(accuracy)
  table.append([accuracy, manual_count, detected_count])
  # log.info('print_manual_to_detected_mapping_table:')
  # print(tabulate(table, headers=['idx', 'manual slice', 'detected op (name type idx)']))


def test_accuracy(tp_manual, idx_to_op_map, flow_index, metadata):
  """We are going to match IDs found in the manual annotation to the detected ops."""
  # if not was_step_called:  # TODO: Fix me for user friendliness
  #   raise Exception('When testing accuracy you must call hotline.annotate.step() after each training iteration.')
  diff_count = {}
  manual_count = {}
  detected_count = {}
  accuracy = {}
  diff_sum = 0
  manual_sum = 0
  detected_sum = 0

  start_idx = 0
  end_idx = len(idx_to_op_map)

  for idx, op in idx_to_op_map.items():
    if idx < start_idx or idx >= end_idx:
      continue

    # Find manual annotation for this op
    manual_annotation = h_annotate.get_manual_annotation_by_idx(tp_manual, idx)
    if not manual_annotation:
      log.error('[accuracy.py] Manual annotation not found for detected op: %s', op['name'])
      embed()

    # Get detected slices and slices in manual annotation
    detected_slices = h_slice.get_slices(op)
    manual_slices = h_perfetto.get_descendant_and_connected_slices(tp_manual, flow_index, manual_annotation["id"])
    if op.get('is_model_pass') == 'Backward':
      # Get slices on other CPU threads
      bw_slices = h_perfetto.get_slices_on_other_threads_on_same_process_between_time_range(tp_manual, manual_annotation)
      # Get GPU kernels
      bw_slices.extend(h_perfetto.get_connected_slices(tp_manual, flow_index, bw_slices))
      manual_slices.extend(bw_slices)
    manual_slices = h_annotate._remove_manual_annotations(manual_slices)  # For example, forward pass has manual annotations of the layers and ops, we don't want to count those events in the accuracy caclulation

    # Check accuracy between detected slices and slices in manual annotation
    diff_count[idx], manual_count[idx], detected_count[idx] = check_accuracy_by_name_count(op, detected_slices, manual_slices)
    accuracy[idx] = calc_accuracy(diff_count[idx], manual_count[idx], detected_count[idx])
    diff_sum += diff_count[idx]
    manual_sum += manual_count[idx]
    detected_sum += detected_count[idx]

  total_accuracy = calc_accuracy(diff_sum, manual_sum, detected_sum)
  total_accuracy_str = h_print.percent_to_str(total_accuracy)

  # Print accuracy summary table
  table = []
  for idx, op in idx_to_op_map.items():
    if idx < start_idx or idx >= end_idx:
      continue
    name = f'{op.get("idx", "")} {op["name"]}'
    table.append([name, manual_count[idx], detected_count[idx], diff_count[idx], h_print.percent_to_str(accuracy[idx])])
  table.append(['Total', manual_sum, detected_sum, diff_sum, total_accuracy_str])
  log.info('\nAccuracy Summary')
  print(tabulate(table, headers=['op name', 'manual', 'detected', 'diff', 'accuracy']))
  log.info('\n')

  return total_accuracy_str
