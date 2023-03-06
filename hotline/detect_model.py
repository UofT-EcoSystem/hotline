import copy

from hotline.hotline import *

import hotline.perfetto as h_perfetto
import hotline.tree as h_tree
import hotline.op as h_op
import hotline.name as h_name
import hotline.time as h_time
import hotline.print as h_print
import hotline.slice as h_slice
import hotline.accuracy as h_accuracy

def locate_forward_backward_annotations(op, num_gpus, **kwargs):
  def banned_name(op):
    banned_names = [
        'replicate', 'scatter', 'gather', 'unflatten',  # seen in forward pass
        'broadcast', 'accumulateGrad'      # seen in backward pass
      ]
    for name in banned_names:
      if name.lower() in op['name'].lower():
        return True

  if op['name'] in ['Forward','Backward'] and op['type'] == 'training loop':
    # Only look within the forward/backward stages of training

    if num_gpus == 1 or 'ops' not in op:
      # If single gpu OR has no sub ops THEN set True
      op['is_model_pass'] = op['name']  # store forward or backward
      log.debug(f'Detected the {op["name"]} pass.')
      return

    # Get top-N sub-op with most slices, where N is the number of GPUs, mark them as model passes
    slice_counts, ranks = h_slice.count_slices_in_each_op(op['ops'])
    num_located = 0
    target_rank = 0
    while num_located < num_gpus:
      target_rank += 1
      try:
        idx = ranks.index(target_rank)
      except ValueError as e:
        log.error(f'We failed to detect the {op["name"]} pass. Please investigate.')
        raise e
      candidate_op = op['ops'][idx]
      if not banned_name(candidate_op):
        candidate_op['is_model_pass'] = op['name']  # store forward or backward
        num_tracks = len(candidate_op["resources"].keys())
        num_located += num_tracks  # if this annotation has multiple tracks, those are CPU threads launching to different GPUs
        log.info(f'\nDetected the {op["name"]} pass in the {candidate_op["name"]} op.\n')


def add_remainder_op(self, res_name):
  slice = {
    'name': 'Remaining slices not assosciated with any operation in the model definition',
  }
  slice_group = [slice]
  sub_slices = self.slices[self.slice_idx :] # the remaining slices

  # Check if slices are already contained within an existing op the existing model definition. Check this by removing slices if they start before the last model def op finishes, ie. is contained.
  ts, dur = h_time.slices_time_stats(self.model_def[0]["resources"][res_name]["slices"])
  last_op_end_ts = ts + dur
  sub_slices = [sub_slice for sub_slice in sub_slices if sub_slice["ts"] > last_op_end_ts]
  if not sub_slices:
    return

  depth = 0
  op = h_op.create_op(slice_group, sub_slices, res_name, depth)
  op['pretty_name'] = 'Remaining Ops'
  self.model_def[0]["ops"].append(op.copy())
  self.model_def[0]["resources"][res_name]["slices"].extend(sub_slices)


def remove_repeating_ops(op, **kwargs):
  if 'ops' not in op:
    return

  new_sub_ops = []
  last_name = None
  last_type = None
  for sub_op in op['ops']:
    if 'ops' not in sub_op and (sub_op['name'] == last_name or sub_op['type'] == last_type):
      # Only remove leafs with the same name
      # print(f'removed {sub_op["name"]}')
      continue
    else:
      new_sub_ops.append(sub_op)
      last_name = sub_op['name']
      last_type = sub_op['type']

  op['ops'] = new_sub_ops


def remove_lonely_resources(op, **kwargs):
  """Remove resource if it only has a single slice"""
  if 'resources' not in op:
    return
  remove_names = [
    'Memcpy HtoD (Device -> Device)',
    'Memset (Device)',
  ]
  pop_list = []
  for res_name, res in op['resources'].items():
    slices = res.get('slices', [])
    if len(slices) == 1 and slices[0]['name'] in remove_names:
      pop_list.append(res_name)
  for pop_name in pop_list:
    op['resources'].pop(pop_name)

def mark_is_backward(op, **kwargs):
  op['is_backward_op'] = True
  if 'idx' in op:
    del op['idx']  # remove the idx as it is the same as the matching forward pass op. TODO: add a new idx for the backward pass op but need to know what the next idx is


def detect_model_dfs(ops, state, is_backward, tp):
  """
  Forward pass: Post-order tree traversal. A kind of depth-first search. Is a recursive function. Used to step through the model definition in forward order, leafs first, then parents. https://en.wikipedia.org/wiki/Tree_traversal

  Backward pass: Reverse pre-order tree traversal. A kind of depth-first search. Is a recursive function. Used to step through the model definition in reverse order, which is the  same order as the backward pass. https://en.wikipedia.org/wiki/Tree_traversal#Reverse_pre-order,_NRL
  """
  if is_backward:
    ops = list(reversed(ops))
  for idx in range(len(ops)):
    op = ops[idx]
    if 'ops' in op:
      detect_model_dfs(op['ops'], state, is_backward, tp)

    # Need next op to handle "Edge case: Don't match if next detected op has the same name"
    # Limitation: current next_op will only work for direct sibiling ops of the same parent, not when two leafs are technically next but have different parents. Possible solution: on each op set the parent so that the child can traverse up and anywhere.
    next_op = None
    if idx + 1 < len(ops):
      next_op = ops[idx + 1]

    infer_span(op, state, is_backward, tp, next_op)
  return ops, state


def next_slice_also_matches(op, state, is_backward, tp):
  """
  Match to last instance of the operation name before the next operation. This means keep iterating over slices until we find the next op, include all slices up to the last time fuzzy_op_match found a match.

  Example:
    Model definition ops: conv, relu
    Slices: mem, conv, conv, mem, relu
    Previous erroneous annotation: conv(mem, conv), relu(conv, mem, relu)
    Fixed annotation: conv(mem, conv, conv), relu(mem, relu)

  Edge case: Don't match if next detected op has the same name
    Example:
    Model definition ops: conv, conv, relu
    Slices: conv, conv, relu
    Previous erroneous annotation: conv(conv, conv)
    Fixed annotation: conv(conv), conv(conv), relu(relu)

"""
  try:
    next_slice = state['slices'][state['slice_idx']]
  except IndexError:
    return False

  op_found, is_fused = h_name.fuzzy_op_match(next_slice, op, is_backward, tp)
  if op_found:
    return True

  heuristic_names = ['aten::to', 'aten::copy_']
  if op['name'] in heuristic_names and next_slice['dur'] <= 6000:
    # Heuristic to match slices that are instantanous or 1 microsecond (up to 6us might be better) because they are likely to be noise as we have seen when loading models in Stable-Diffusion inference where the next 100 slices is a pattern of the same name slice then an instantanous slice and it repeats.
    log.warn(f'Using heuristic to include slice in op but is not the ideal candidate for this heuristic. The slice is: {next_slice}')
    # TODO: Add an extra check to make sure we aren't making an obvious mistake when the next slice matches the next op. This is a failure case of this heuristic currently but can be fixed using the next op when traversing the tree with detect_model_dfs().
    return True


def next_op_same_name(op, next_op):
  if not op or not next_op:
    return False
  return op['name'] == next_op['name'] or op['type'] == next_op['type']

def infer_span(op, state, is_backward, tp, next_op):
  """Iterating through the backward trace until we find the op aka. position in the pytorch model archecture."""
  # log.info(f'[infer_span] Processing: {op.get("idx")}, {op["name"]}, {op["type"]}')
  # Handle parents aka. modules (such as "layer1") that contain base ops such as "ReLU"
  if 'ops' in op:
    log.info(f'Branch "{op["name"]}, {op["type"]}"')
    op_found = h_slice.add_sub_op_slices_to_op(op, res_name=state['res_name'])
    return op_found

  if state['slice_idx'] == len(state['slices']):
    # We already reached the end of the bw trace so stop doing work
    state["op_not_found_count"] += 1
    return

  # Handle base ops such as "ReLU"
  start_idx = state['slice_idx']
  while state['slice_idx'] < len(state['slices']):
    slice = state['slices'][state['slice_idx']]
    op_found, is_fused = h_name.fuzzy_op_match(slice, op, is_backward, tp)
    # is_fused = False # Feature Toggle: Disable fused op matching
    if not op_found or not is_fused:
      state['slice_idx'] += 1 # increment by one in default case
    if not op_found and state['last_found_was_fused']:
      start_idx += 1 # make up for fused
    log.debug(f'[infer_span] {op["name"]} {state["slice_idx"]} {op_found} {is_fused}')
    if op_found:
      log.info(f'Found  hid={h_print.op_idx_name(op)}, slice_idx={state["slice_idx"]}, in slice: {h_name.rename_slice(slice["name"])} slice_id={slice["id"]} dur={slice["dur"]} ✅')  # NOTE: slice_id will be different if opened with Perfetto.trace_processor as a subset or superset (like with hotline manual annotations).
      while next_slice_also_matches(op, state, is_backward, tp) and not next_op_same_name(op, next_op):
        state['slice_idx'] += 1
      # is_fused = False # Feature Toggle: Disable fused op matching
      if is_fused:
        slices = state['slices'][start_idx : state['slice_idx'] + 1]
        slices = [slice.copy() for slice in slices]  # note this copy allows repeated slices in the case of fusion to be edited seperately
        state['last_found_was_fused'] = True
        state['op_found_count'] += 1
      else:
        slices = state['slices'][start_idx : state['slice_idx']]
        state['last_found_was_fused'] = False
      log.debug(f'[infer_span] {start_idx} - {state["slice_idx"]} [{len(slices)}]')
      # h_slice.add_slices_to_op(op, slices, 'cpu')
      h_perfetto.add_slices(op, slices, tp)
      state['last_found_slice_idx'] = state['slice_idx']
      state['op_found_count'] += 1
      return op_found

  if not op_found:
    # Reset slice_idx to last found position to continue from there looking for the next op, ignoring the one we couldn't find.
    state['op_not_found_count'] += 1
    state['slice_idx'] = state['last_found_slice_idx']



def detect_model(op, model_ops, tp, parent_op=None, **kwargs):
  if 'is_model_pass' not in op:
    return

  for res_name, res in op['resources'].items():
    if 'cpu' not in res_name:
      continue
    slices = res.get('slices', [])

    if len(slices) < 30:
      # TODO Actually check if actually a model pass on a secondary thread. The primary thread will likely have not have more than 30 slices.
      continue

    fw_slice = [slice for slice in slices if slice['name'] in ['DataParallel.forward', 'nn.Module: DataParallel']]
    if fw_slice:
      fw_slice = fw_slice[0]
      slices = tp.query_dict(f'SELECT * FROM descendant_slice({fw_slice["id"]})')
    slices =  h_slice.remove_upper_depths_if_only_one_slice(slices)
    slices =  h_slice.get_slices_at_depth(slices, 'minimum')

    log.info(f'Begin {op["is_model_pass"]} model detection for "{op["name"]}" on resource "{res_name}" with {len(slices)} slices at min depth.')
    state = {}
    state['slices'] = slices
    state['slice_idx'] = 0
    state['res_name'] = res_name
    state['last_found_slice_idx'] = 0
    state['op_found_count'] = 0
    state['op_not_found_count'] = 0
    state['last_found_was_fused'] = False
    is_backward = True if op['is_model_pass'] == 'Backward' else False

    model_ops = copy.deepcopy(model_ops)
    detect_model_dfs(model_ops, state, is_backward, tp)
    op['ops'] = model_ops

    if is_backward:
      # Reverse ops
      op['ops'].reverse() # reverse top level op
      h_tree.pre_order_depth_first(op, h_op.reverse_op) # reverse all child ops recursively
      h_tree.pre_order_depth_first(op, mark_is_backward)
