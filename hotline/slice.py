import copy

from hotline.hotline import *

import hotline.op as h_op


def get_slices_at_depth(slices, depth):
  if not slices:
    return []
  if len(get_unique_tracks(slices)) > 1:
    msg = 'When getting slices at depth, found more than one track. Ambigious.'
    log.warn(msg)
    # raise ValueError(msg)
  if isinstance(depth, str):
    if depth == 'minimum':
      depth = min([slice["depth"] for slice in slices])

  return [slice for slice in slices if slice['depth'] == depth]


def get_unique_depths(slices):
  """Example: Convert  [2, 2, 1, 2, 3] to [1, 2, 3] so we can loop over unique depths in order, lowest first"""
  depths = [slice["depth"] for slice in slices]
  uniq_depths = depths.copy()
  uniq_depths.sort()
  uniq_depths = list(set(uniq_depths))
  return uniq_depths


def get_unique_tracks(slices):
  return sorted(set([slice["track_id"] for slice in slices]))


def get_slices(ops, filter_by_resource='', depth=None):
  all_slices = []
  if not isinstance(ops, list):
    ops = [ops]

  for op in ops:
    slices = []
    if 'resources' in op:
      for res_name, res in op['resources'].items():
        if filter_by_resource in res_name and 'slices' in res:
          s = res['slices']
          if depth:
            s = get_slices_at_depth(s, depth)
          slices.extend(s)
    all_slices.extend(slices)

  return all_slices


def _normalize_slice_depth(slices):
  min_depth = min([slice["depth"] for slice in slices])
  for slice in slices:
    slice['depth'] -= min_depth


def normalize_slice_depth(op):
  """Normalize depth of slices to always start at 0."""
  if 'resources' in op:
    for res_name, res in op['resources'].items():
      slices = res['slices']
      _normalize_slice_depth(slices)


def get_sub_slices_for_range(slices, slice_group):
  if not slice_group:
    return []
  start_slice = slice_group[0]
  end_slice = slice_group[-1]
  end_ts = end_slice['ts'] + end_slice['dur']
  start_ts = start_slice['ts']
  total_dur = end_slice['ts'] + end_slice['dur'] - start_slice['ts']

  slices_between_range = [
    slice for slice in slices
      if slice['ts'] >= start_ts
      and slice['ts'] < end_ts
      and slice['dur'] <= total_dur
    ]
  return slices_between_range


def remove_slices(op, **kwargs):
  if 'resources' not in op:
    return
  # if 'ops' in op:  # only apply to non-leaf nodes
  for res_name, res in op['resources'].items():
    if 'slices' in res:
      res.pop('slices', None)


def add_sub_op_slices_to_op(given_op, res_name):
  """ Used when making annotations. Puts all slices from child ops into parent op."""
  if len(given_op['ops']) == 0:
    raise ValueError('Cannot have an "ops" key that is empty. Please investigate this situation.')

  def dft_append_slices(op):
    if 'ops' in op:
      for sub_op in op['ops']:
          dft_append_slices(sub_op)
    else:
      # log.info(f'[dft_append_slices] Processing: {op.get("idx")}, {op["name"]}, {op["type"]} for {given_op["name"]}')
      if 'resources' in op and res_name in op['resources']:
        slices = op['resources'][res_name].get('slices', [])
        slices = copy.copy(slices)  # fix duplicates being added because all lists are the same
        h_op.append_list_to_resource(given_op, res_name, 'slices', slices)

  [dft_append_slices(op) for op in given_op['ops']]


def add_slices_to_op(op, slices, res_prefix):
  for slice in slices:
    h_op.append_list_to_resource(op, res_prefix + str(slice['track_id']), 'slices', slice)


def get_op_with_most_slices(ops):
  max_slice_count = 0
  max_slice_count_index = None
  for index, op in enumerate(ops):
    slice_count = len(get_slices(op))
    if slice_count > max_slice_count:
      max_slice_count = slice_count
      max_slice_count_index = index
  return ops[max_slice_count_index]


def get_track_with_most_slices(slices):
  count_per_track = {}
  for slice in slices:
    track_id = slice['track_id']
    if track_id not in count_per_track:
      count_per_track[track_id] = 1
    else:
      count_per_track[track_id] += 1

  # Getting key with maximum value in dictionary
  max_slices_track_id = max(count_per_track, key=count_per_track.get)

  return count_per_track, max_slices_track_id


def remove_tracks_with_n_slices(slices, n):
  count_per_track = {}
  if not slices: return
  tid = 'tid' if slices[0].get('tid') is not None else 'track_id'  # use tid or track_id

  for slice in slices:
    track_id = slice[tid]
    if track_id not in count_per_track:
      count_per_track[track_id] = 1
    else:
      count_per_track[track_id] += 1

  # Filter slices if track only has n slices
  track_ids_to_keep = [track_id for track_id, count in count_per_track.items() if count != n]
  filtered_slices = [slice for slice in slices if slice[tid] in track_ids_to_keep]
  if filtered_slices:
    return filtered_slices
  return slices


def count_slices_in_each_op(ops):
  # Calc slice counts
  lengths = []
  for op in ops:
    lengths.append(len(get_slices(op)))

  # Ranks from most to least number of slices
  _sorted = list(reversed(sorted(lengths)))
  ranks = [ _sorted.index(value) + 1 for value in lengths ]

  # Remove duplicates so that '[2, 1, 7, 8, 3, 3, 3, 3]' becomes '[2, 1, 7, 8, 3, 4, 5, 6]'
  for rank in set(ranks):
    rank_count = 0
    for idx, r in enumerate(ranks):
      if r == rank:
        rank_count += 1
        if rank_count > 1:
          ranks[idx] += rank_count - 1

  return lengths, ranks


def remove_slices_at_depth(slices, depth):
  """Remove all slices at a given depth."""
  for slice in slices:
    if slice['depth'] == depth:
      slices.remove(slice)


def remove_upper_depths_if_only_one_slice(slices):
  """If there is only one slice at a depth, remove it. Only applies to upper depths until a depth with multiple slices is found."""
  depths = get_unique_depths(slices)
  for depth in depths:
    slices_at_depth = get_slices_at_depth(slices, depth)
    if len(slices_at_depth) == 1:
      remove_slices_at_depth(slices, depth)
    else:
      return slices
