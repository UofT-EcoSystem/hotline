import copy
from collections import OrderedDict

from IPython import embed

from hotline.hotline import *

import hotline.name as h_name
import hotline.util as h_util
import hotline.slice as h_slice

def get_unique_resource_names(ops):
  uniq_res_names = set()
  for op in ops:
    uniq_res_names.update(list(op['resources'].keys()))
  return list(uniq_res_names)


def add_dict_to_resource(op, res_name, category, **kwargs):
  if 'resources' not in op:
    op['resources'] = {}

  if res_name not in op['resources']:
    op['resources'][res_name] = {}

  if category not in op['resources'][res_name]:
    op['resources'][res_name][category] = {}

  for key, value in kwargs.items():
    op['resources'][res_name][category][key] = value


def append_list_to_resource(op, res_name, list_name, list_):
  if not list_:
    return

  if 'resources' not in op:
    op['resources'] = {}

  if res_name not in op['resources']:
    op['resources'][res_name] = {}

  if list_name in op['resources'][res_name]:
    if isinstance(list_, list):
      op['resources'][res_name][list_name].extend(list_)
    else:
      op['resources'][res_name][list_name].append(list_)
  else:
    if isinstance(list_, list):
      op['resources'][res_name][list_name] = list_
    else:
      op['resources'][res_name][list_name] = [list_]

  # if 'ts' in op['resources'][res_name][list_name][0]:
  #   # Sort so weird things don't happen
  #   # Comment out for speedup
  #   log.debug(f'sorting {len(op["resources"][res_name][list_name])} elements')
  #   op['resources'][res_name][list_name] = sorted(op['resources'][res_name][list_name], key=lambda d: d['ts'])


def create_op(slice_group, sub_slices, res_name, depth):
  sub_op = {
    'name': h_name.calc_predominant_name(slice_group),
    'type': 'generated',
    'generated_depth': depth,
    'instances': len(slice_group),
  }
  append_list_to_resource(sub_op, res_name, 'slices', sub_slices)

  return sub_op


def join_ops(ops):
  [h_slice.normalize_slice_depth(op) for op in ops]  # We want the predominant name to reflect all ops being joined, even if different ops are at different depths.
  slices = h_slice.get_slices(ops, filter_by_resource='cpu')

  if not slices:
    log.warn('No slices when joining. Please investigate.')
    predominant_name = 'No slices'
  else:
    predominant_name = h_name.calc_predominant_name(slices)

  new_op = {
    'name': predominant_name,
    'type': 'generated',
    'instances': len(slices),
    'id': h_util.random_id()
  }

  for res_name in get_unique_resource_names(ops):
    res_slices = []
    for op in ops:
      res_slices.extend(h_slice.get_slices(op, filter_by_resource=res_name))
    append_list_to_resource(new_op, res_name, 'slices', res_slices)

  return new_op


def create_top_level_op(ops, name):
  top_op = {'name': name, 'type': 'root', 'ops': ops }
  slices = h_slice.get_slices(ops)
  append_list_to_resource(top_op, 'default_res', 'slices', slices)
  return top_op


def reverse_op(op, **kwargs):
  log.debug(f'[reverse_op] Processing: {op.get("idx")}, {op["name"]}, {op["type"]}')
  if not 'ops' in op:
    return
  op['ops'].reverse()


def check_for_missing_op(op, **kwargs):
  """Add first slice start time, last slice finish time, and duration."""
  log.debug(f'[check_for_missing_op] Processing: {op.get("idx")}, {op["name"]}, {op["type"]}')
  if not 'resources' in op:
    msg = f'❌❌ We did not find "{op.get("idx")}, {op["name"]}, {op["type"]}" on any resource of the trace! Error'
    log.error(msg)
    raise ValueError(msg)

def delete_missing_ops(op, **kwargs):
  if 'ops' in op:
    reduced_sub_ops = []
    for sub_op in op['ops']:
      if 'resources' in sub_op:
        reduced_sub_ops.append(sub_op)
    op['ops'] = reduced_sub_ops


def sort_resource(op, **kwargs):
  if 'resources' not in op:
    return
  log.debug(f'[sort_resource] Processing: {op.get("idx")}, {op["name"]}, {op["type"]}')
  op['resources'] = OrderedDict(sorted(op['resources'].items()))


def add_random_id(op, **kwargs):
  if not op.get('id'):
    op['id'] = h_util.random_id()


def split_default_resource(op, track_index, **kwargs):
  """Seperate the slices from the default_res to different timelines"""
  if 'resources' not in op:
    return

  slices = op["resources"]["default_res"]["slices"]
  op["resources"].pop("default_res")  # delete default resource

  # Sort slices by track
  slices_by_track = {}
  for slice in slices:
    track_id = slice["track_id"]
    if track_id not in slices_by_track:
      slices_by_track[track_id] = [slice]
    else:
      slices_by_track[track_id].append(slice)

  # Add slices into seperate resources
  for track_id in slices_by_track.keys():
    append_list_to_resource(op, 'cpu' + str(track_id), 'slices', slices_by_track[track_id])


def get_idx_by_name_from_idx_to_op_map(name, idx_to_op_map):
  for idx, op in idx_to_op_map.items():
    if op['name'] == name:
      return idx
