
import os
import random, string
import json
from IPython import embed

from hotline.hotline import *


builtin_slice = slice

high_level_ops = [
  {
    'name': 'Load Data',
    'raw_names': [
      'pop of dict',
      'get_data',
      'DataLoad',
      'aten::to',
      'bytes',
    ],
    # Real world examples:
      # dataloader
      # <built-in method pop of dict object at 0x7f65cd21e3c0>
      # enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__
      # torch/utils/data/dataloader.py(1173): _get_data
      # <built-in method getvalue of _io.BytesIO object at 0x7f86f21290d0>
  },
  {
    'name': 'Forward',
    'raw_names': [
      'DataParallel',
      'forward',
    ],
    # Real world examples:
      # DataParallel.forward
      # nn.Module: DataParallel
  },
  {
    'name': 'Calc Loss',
    'raw_names': [
      'loss'
    ],
    # Real world examples:
      # aten::cross_entropy_loss
      # nn.Module: CrossEntropyLoss
  },
  {
    'name': 'Zero Grad',
    'raw_names': [
      'zero_grad',
    ],
    # Real world examples:
      # Optimizer.zero_grad#SGD.zero_grad
      # Optimizer.zero_grad#Adam.zero_grad
      # torch/optim/optimizer.py(191): zero_grad
  },
  {
    'name': 'Backward',
    'raw_names': [
      'autograd::',
      'backward',
    ],
    # Real world examples:
      # torch/_tensor.py(340): backward
  },
  {
    'name': 'Optimizer',
    'raw_names': [
      'optim',  # be careful, this also matches to: Optimizer.zero_grad#SGD.zero_grad
    ],
    # Real world examples:
      # Optimizer.step#SGD.step
      # Optimizer.step#Adam.step
      # torch/optim/optimizer.py(83): wrapper
  },
]

def convert_detected_to_op(op):
  """ Basically just add resources. """
  new_op = {
    'name': op['name'],
    'type': 'training loop',
    'instances': len(op['slices']),
  }
  h_op.append_list_to_resource(new_op, 'default_res', 'slices', op['slices'])
  return new_op

def remove_profiler_step_trailing_op(detected_ops):
  # Remove the slice "lib/python3.8/site-packages/torch/profiler/profiler.py(481): step" because it is not apart of training, it is a profiling slice that adds the runtime of the last step of training, usually the optimizer.
  if '): step' in detected_ops[-1]["slices"][-1]['name']:
    detected_ops[-1]["slices"].pop()

def _detect_training_loop(high_level_ops, slices):
  """ The algorithm looks for the training loop in the trace by substring matching the names of possible high-level steps of training to the in the names of event trace.

  TODO: warn if more than one match
  """

  if not high_level_ops:
    return []
  if not slices:
    return [{'name': op['name'], 'slices':[]} for op in high_level_ops]

  # Detect stages expecting them to be in the right order in the slices
  idx_op = 0
  op_searched = high_level_ops[idx_op]
  done = False
  idx_slice = 0
  idx_start = 0
  saved_idx = 0
  saved2_idx = 0
  idx_previous_end = 0
  remove_zero_grad = False

  skipped = 0
  skipped_names = []
  ops_detected = []
  while not done:
    slice = slices[idx_slice]
    is_match = h_name.match(slice['name'], op_searched['raw_names'])
    is_last_slice = idx_slice == len(slices) - 1
    is_last_op_searched = idx_op == len(high_level_ops) - 1

    if is_match:
      idx_last_match = idx_slice
      saved_idx = idx_slice

    # Check if matches to another op
    is_match_to_next_searched = h_name.match(slice['name'], high_level_ops[idx_op+1]['raw_names']) if not is_last_op_searched else False

    if is_match_to_next_searched:
      saved_idx = idx_slice
      # If we couldn't find a high-level op, but we found the op after it, then we can fill in the middle missing op
      if skipped == 2:
        raise NotImplementedError(f'Cannot skip detecting two high level ops consecutively. Skipped: {skipped_names}')
      if skipped == 1:
        # Change the last matched high-level op to end on it's last match rather than including stuff in between
        previous_slices = slices[idx_previous_end:saved2_idx+1]
        previous_op = {'name': high_level_ops[idx_op-1]['name'], 'slices': previous_slices}
        ops_detected.append(previous_op)
        idx_start = saved2_idx + 1

      range = builtin_slice(idx_start, idx_slice)
      idx_previous_end = idx_slice
      idx_start = idx_slice

      # Create high level op
      slice_group = slices[range]
      op_detected = {'name': op_searched['name'], 'slices': slice_group}
      ops_detected.append(op_detected)
      skipped = 0

      # Increment to next high level op_searched
      if not is_last_op_searched:
        idx_op += 1
        op_searched = high_level_ops[idx_op]
        is_last_op_searched = idx_op == len(high_level_ops) - 1

    # If we are at the end of the trace, but we didn't find the last high level op, then we are not done
    if is_last_slice and not is_last_op_searched:
      log.debug([slice['name'] for slice in slices])
      # if idx_previous_end is not None:
      skipped += 1  # set the flag so that we can handle the missing high level op
      skipped_names.append(high_level_ops[idx_op]['name'])
      idx_slice = idx_previous_end  # skip backwards in slices list to after last match
      idx_start = idx_slice  # set start of range include our new position
      idx_op += 1  # increment to next high level op
      op_searched = high_level_ops[idx_op]  # increment to next high level op
      saved2_idx = saved_idx

    if idx_slice >= len(slices) - 1:
      done = True
    idx_slice += 1

  if len(ops_detected) < 3:  # probably not a training loop
    log.warn(f'[detect_training_loop] Not enough training loop ops detected. Returning a default op containing all slices.')
    return [{'name': 'default', 'slices': slices}]

  # Create high level op to the end of the trace
  if idx_last_match < idx_start:
    log.warn(f"Last op {op_searched['name']} not found in trace")
  slice_group = slices[idx_start:]  # to the end of the trace
  op_detected = {'name': op_searched['name'], 'slices': slice_group}
  ops_detected.append(op_detected)
  ops_detected = [op for op in ops_detected if 'slices' in op and len(op['slices'])]

  return ops_detected

def detect_training_loop(tp, annotations):
  # Get ProfileStep
  profile_step_slice, between_time_range = h_perfetto.get_profile_step_slice(tp)

  # Get primary tracks/timelines
  primary_track_ids, on_primary_tracks = h_perfetto.track_ids_for_process_of_slice(tp, profile_step_slice)

  # Get slices in ProfilerStep across all threads of the primary cpu process
  slices = tp.query_dict(f'SELECT * from slices WHERE {on_primary_tracks} AND {between_time_range}')

  # Exclude the profile step because it is at an uninteresting depth we want to ignore
  step_slice = slices.pop(0)
  assert 'ProfilerStep' in step_slice['name']
  h_slice._normalize_slice_depth(slices)

  # Select slices at minimum depth (on a per track basis because the track that had ProfilerStep has a different minimum depth)
  top_level_slices = []
  for track_id in primary_track_ids:
    track_slices = [slice for slice in slices if slice['track_id'] == track_id]
    min_depth_track_slices = h_slice.get_slices_at_depth(track_slices, 'minimum')
    assert(len(min_depth_track_slices))  # make sure the track is not empty
    top_level_slices.extend(min_depth_track_slices)
  slices = top_level_slices

  # Sort slices so we can iterate through time
  slices = sorted(slices, key=lambda d: d['ts'])

  # Detect high level sections
  detected_ops = _detect_training_loop(high_level_ops, slices)

  # Remove trailing slice generated by the profiler if it exists
  remove_profiler_step_trailing_op(detected_ops)

  # Convert to resource format
  ops = [convert_detected_to_op(op) for op in detected_ops]

  # Add filename and lineinfo when available if annotations were used
  if annotations:
    for op, annotation in zip(ops, annotations):
      if op["name"].lower() == annotation["name"].lower():
        op['source_stack_trace'] = annotation['source_stack_trace']
        op['source_file_name'] = annotation['source_file_name']
        op['source_file_num'] = annotation['source_file_num']

  # Remove leading uninteresting slices from the first op ("Load Data)") since have seen GNN have an "aten::empty" slice at the beginning of the trace with a super long gap until the first real slice.
  try:
    slices = h_slice.get_slices(ops[0], depth='minimum')
    for idx, slice in enumerate(slices):
      if h_accuracy.is_interesting_slice(slice):
        break  # found first interesting slice
    slices = [s for s in slices[idx:]]  # remove uninteresting slices
    ops[0]['resources']['default_res']['slices'] = slices
  except:
    log.warn(f'[detect_training_loop] Failed to remove leading uninteresting slices from the first op ("Load Data)")')

  # Calculate timing
  [h_time.add_time(op) for op in ops]

  # Get nested slices at lower depths for each high level section
  for op in ops:
    time =  op['resources']['default_res']['time']
    start_ts = time['ts']
    end_ts = time['ts'] + time['dur']
    within_high_level_section = f'ts BETWEEN {start_ts} AND {end_ts}'
    sub_slices = tp.query_dict(f'SELECT * FROM slices WHERE {on_primary_tracks} AND {within_high_level_section};')
    op['resources']['default_res']['slices'] = sub_slices

  return ops
