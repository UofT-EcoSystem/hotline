
from hotline.hotline import *



def raise_sub_ops_of_dominant_op(op, raise_op_threshold, **kwargs):
  apply_to_exceptions = [
    'Forward',
    'Backward',
    'DataParallel',
    'Transformer',
    'decoder',
    'encoder',
    'layers',
  ]
  if 'ops' not in op:
    return
  if 'resources' not in op:
    return
  if any([sub_op for sub_op in op['ops'] if 'resources' not in sub_op]):
    return
  # Don't apply to non-generated ops with exceptions, because we don't want those to disappear or change structure
  if any([sub_op for sub_op in op['ops'] if sub_op['type'] != 'generated' and sub_op['name'] not in apply_to_exceptions]) and op['name'] not in apply_to_exceptions:
    return

  log.debug(f'[raise_sub_ops_of_dominant_op] Processing: {op.get("idx")}, {op["name"]}, {op["type"]}')
  old_ops = op['ops']
  new_ops = []
  max_dur = max([res['time']['dur'] for _, res in op['resources'].items()])
  for sub_op in old_ops:
    max_relative_dur = max([res['time']['relative_dur'] for _, res in sub_op['resources'].items()])
    # if this op takes up the majority of the time replace it with it's children
    if max_relative_dur > raise_op_threshold and 'ops' in sub_op:
      # adjust the time of the children ops to take instead of 100% take the % of the op being replaced.
      for idx, child_op in enumerate(sub_op['ops']):
        for res_name, _ in child_op.get('resources',{}).items():
          # make adjustment to dur
          child_relative_dur = child_op['resources'][res_name]['time']['relative_dur']
          child_op['resources'][res_name]['time']['relative_dur'] = child_relative_dur * max_relative_dur

          # make adjustment to gap
          child_relative_gap_to_previous = child_op['resources'][res_name]['time']['relative_gap_to_previous']
          child_op['resources'][res_name]['time']['relative_gap_to_previous'] = child_relative_gap_to_previous * max_relative_dur

          # make adjustment to gap if child op doesn't start where it's containing sub_op began
          if idx == 0 and new_ops and res_name in sub_op['resources'] and res_name in new_ops[-1]['resources']:
            time_to_previous_sub_op = sub_op['resources'][res_name]['time']['ts'] - (new_ops[-1]['resources'][res_name]['time']['ts'] + new_ops[-1]['resources'][res_name]['time']['dur'])
            relative_prev_end_to_start_of_sub_op = time_to_previous_sub_op / max_dur
            child_op['resources'][res_name]['time']['relative_gap_to_previous'] += relative_prev_end_to_start_of_sub_op


      new_ops.extend(sub_op['ops'])
    else:
      new_ops.append(sub_op)

  op['ops'] = new_ops






def detect_arbitrary_annotations(op, max_generated_depth, tiny_op_threshold, **kwargs):
  """Looks at slices of leaf ops and will create new ops either for groups of small slices, same names, or individual long slices."""
  except_list = [
    # Nothing here yet
  ]

  if 'ops' in op or op.get('generated_depth', 0) >= max_generated_depth:  # limit generated depth
    return
  if 'resources' not in op:
    return
  log.debug(f'[detect_arbitrary_annotations] Processing: {op.get("idx")}, {op["name"]}, {op["type"]}')

  # Calculate thresh so that many tiny ops can be grouped.
  max_dur = max([h_time.slices_time_stats(res['slices'])[1] for _, res in op['resources'].items()]) # note the [1] here means 'dur', [0] would mean 'ts'
  short_slice_thresh = max_dur * tiny_op_threshold

  sub_ops = []
  for res_name, res in op['resources'].items():
    all_slices = res.get('slices', [])
    # Get slices on track with most slices, ignore the other tracks. We will make annotations using time spans found on track with most slices
    _, max_slices_track_id = h_slice.get_track_with_most_slices(all_slices)
    slices = [slice for slice in all_slices if slice['track_id'] == max_slices_track_id]

    # Get unique slice depths so that we can select only what we are interested in. We are interested in the first depth that has more than 1 slice on it.
    depths = [slice["depth"] for slice in slices]
    uniq_depths = h_slice.get_unique_depths(slices) # Convert  [2, 2, 1, 2, 3] to [1, 2, 3] so we can loop over unique depths in order, lowest first

    # NOTE(deferred): Slices on depths that have only 1 slice get dropped, I'm not sure if this is good (less noise) or bad (lost runtime information). An extreme edge case is a long slice that has a few tiny slices clustered together. In this case the idle time on either side of the cluster is large and knowledge of the long slice that caused it is lost. The knowledge is not completely lost because any parent op which contains the long slice is still aware.

    # Select slices of interest at first interesting depth.
    slices_of_interest = h_slice.get_slices_at_depth(slices, 'minimum') # default to selecting the lowest depth (ex. 0), mainly for the case that only 1 slice is present at each depth
    for depth in uniq_depths:
      if depths.count(depth) > 1:
        # This depth is interesting because it has more than 1 slice on it.
        slices_of_interest = h_slice.get_slices_at_depth(slices, depth)
        break

    # Loop over slices of interets looking for repetitions
    instances = 1
    for idx in range(len(slices_of_interest)+1):
      if idx == 0:
        continue  # skip first

      if idx == len(slices_of_interest):
        # Handle last
        slice = slices_of_interest[idx - 1]
        prev_slice = slices_of_interest[idx - 1]
        is_last_slice = True
      else:
        slice = slices_of_interest[idx]
        prev_slice = slices_of_interest[idx - 1]
        is_last_slice = False

      # Group these two ops if they have the same name or are both very short
      is_same_name = slice['name'] == prev_slice['name']
      is_short_slice = prev_slice['dur'] <= short_slice_thresh and slice['dur'] <= short_slice_thresh
      if (
          (is_same_name or is_short_slice) and
          not slice['name'] in except_list and  # never merge these exceptions with other slices
          not is_last_slice
        ):
        instances += 1

      # Found a new section. Get slices in this section, all of their sub slices, and create a new op.
      else:
        slice_group = slices_of_interest[idx - instances : idx]
        sub_slices = h_slice.get_sub_slices_for_range(all_slices, slice_group)

        # Fix weirdness involving ops with dur=0 by adding to sub_slices any item in the slice_group that is missing from sub_slices.
        sub_slice_ids = [slice['id'] for slice in sub_slices]
        for slice in slice_group:
          if slice['id'] not in sub_slice_ids:
            sub_slices.append(slice)

        # log.info(f'{idx} {prev_slice["name"]} slice_group={[sub_slice["name"] for sub_slice in slice_group]} sub_slices={[sub_slice["name"] for sub_slice in sub_slices]} slices_of_interest={[sub_slice["name"] for sub_slice in slices_of_interest]} #slices={len(slices)}\n')

        depth = op.get('generated_depth', 0) + 1
        sub_op = h_op.create_op(slice_group, sub_slices, res_name, depth)
        if (
            # op['name'] == sub_op['name'] or    # same name
            # # len(sub_slices) == len(slices) or  # same slices
            (op['name'] == sub_op['name'] and    # same name
            len(sub_slices) == len(slices)) or  # same slices
            not sub_slices                     # no slices
          ):
          # Prevent infinite recursion! We require that fewer slices will go into the new sub op than are present in the parent.
          continue
        else:
          # Create a new op reperesenting this section
          sub_ops.append(sub_op)
          instances = 1

  if sub_ops:
    op['ops'] = sub_ops



def join_short_or_same_name_op(op, tiny_op_threshold, **kwargs):
  """Looks at ops and will combine ops either for groups of small slices or same names. Similar to detect_arbitrary_annotations() but operates on ops instead of slices."""
  if 'ops' not in op:
    return
  # Don't apply to non-generated ops, because we don't want those to disappear or change structure
  if any([sub_op for sub_op in op['ops'] if sub_op['type'] != 'generated']):
    log.debug(f'[join_short_or_same_name_op] Skipping: {op.get("idx")}, {op["name"]}, {op["type"]}')
    return
  log.debug(f'[join_short_or_same_name_op] Processing: {op.get("idx")}, {op["name"]}, {op["type"]}')

  # Calculate thresh so that many tiny ops can be grouped.
  max_dur = max([h_time.slices_time_stats(res['slices'])[1] for _, res in op['resources'].items()]) # note the [1] here means 'dur', [0] would mean 'ts'
  short_op_thresh = max_dur * tiny_op_threshold # %

  new_sub_ops = []
  instances = 1
  for idx in range(len(op['ops'])+1):
    if idx == 0:
      continue  # skip first

    if idx == len(op['ops']):
      # Handle last
      sub_op = op['ops'][idx - 1]
      prev_sub_op = op['ops'][idx - 1]
      is_last_sub_op = True
    else:
      sub_op = op['ops'][idx]
      prev_sub_op = op['ops'][idx - 1]
      is_last_sub_op = False

    # Group these two ops if they have the same name or are both very short
    is_same_name = sub_op['name'] == prev_sub_op['name']
    sub_op_max_dur = max([h_time.slices_time_stats(res['slices'])[1] for _, res in sub_op['resources'].items()])
    prev_sub_op_max_dur = max([h_time.slices_time_stats(res['slices'])[1] for _, res in prev_sub_op['resources'].items()])
    is_short_sub_op = sub_op_max_dur <= short_op_thresh and prev_sub_op_max_dur <= short_op_thresh  # if this op and the previous op are short
    if (is_same_name or is_short_sub_op) and not is_last_sub_op:
      instances += 1
    else:
      sub_op_group = op['ops'][idx - instances : idx]
      if len(sub_op_group) > 1:
        log.debug(f'[join_short_or_same_name_op] Group: {[op["name"] for op in sub_op_group]}')
        new_op = h_op.join_ops(sub_op_group)
        h_time.add_time(new_op)
        new_sub_ops.append(new_op)
      else:
        log.debug(f'[join_short_or_same_name_op] Individual: {prev_sub_op["name"]}')
        new_sub_ops.append(prev_sub_op)
      instances = 1

  log.debug(f'[join_short_or_same_name_op] Replaced:  {[op["name"] for op in op["ops"]]} with {[op["name"] for op in new_sub_ops]}')
  op['ops'] = new_sub_ops
  h_time.add_relative_timing(op)
