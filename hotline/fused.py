import os

from hotline.hotline import *


def adjust_fused_kernel_time(op, **kwargs):
  """Adjust the ts and dur of fused kernel slices. Example
  Module ops:
    Conv
    Relu
  Both of these module ops can share the same kernel slice "volta_scudnn_128x64_relu_small_nn_v1". We need to figure out how many module ops are contained in 1 kernel slice. We do this so that we divide the runtime of the kernel slice by the number of ops it is included it. The frontend will then show it multiple times, but with each adjusted to be shorter so that the total runtime equals the original time of the slice if displayed just once.

  We figure out how many module ops are contained in 1 kernel slice by finding consecutive slice_idx's that have the same fused kernel name.

  Can operate individually on any non-leaf node because the logic will work the same further down or further up the module tree, the only difference is the number of slices. The required information to is available either way. Since the slices have been copy()'d, they never get double updated.
    """
  if 'ops' not in op or 'resources' not in op:
    return
  log.debug(f'[fused_length] Processing: {op.get("idx")}, {op["name"]}, {op["type"]}')
  for res_name in op['resources'].keys():
    # if not 'cpu' in res_name:
    #   continue
    slices = h_slice.get_slices(op, filter_by_resource=res_name)

    # Get fused slices, idxs, and names
    fused_slices = [slice for slice in slices if slice.get('possibly_is_fused')]
    fused_slices_idx = [idx for idx, slice in enumerate(slices) if slice.get('possibly_is_fused')]
    fused_slice_names = [slice['name'] for slice in fused_slices]  # TODO: Confirm fusion match using more than just is_same_name and is_consecutive, also use type, ts, dur. Or just don't modify the slice so sameness can be detected here using == and is_consecutive.

    # Loop over fused slices
    fusion_length = 1
    for idx, (slice_idx, name) in enumerate(zip(fused_slices_idx, fused_slice_names)):
      # Operate differently on the last iteration because we compare to position + 1 and we don't want to overbound the list. Instead we want to trigger the code that runs after the end of a series of fused kernels.
      is_last = idx == len(fused_slices_idx) - 1
      if is_last:
        is_consecutive = False
        is_same_name = False
      else:
        # Figure out if the next slice is fused with this one
        next_slice_idx = fused_slices_idx[idx + 1]
        is_consecutive = slice_idx + 1 == next_slice_idx
        next_name = fused_slice_names[idx + 1]
        is_same_name = name == next_name
      if is_consecutive and is_same_name:
        fusion_length += 1
      else:
        # Now that we have identified a fused group, set on each slice the slice idx involved, the fusion length (ie. number of slices in the fusion), and adjust timings (is this needed?) so that we don't include the full time of repeated slices, since a slice is repeated to represent each op contained in the fused slice.
        # Update timings so that each slice is a fraction of the time
        if fusion_length > 1:
          for i in range(fusion_length):
            lookback_idx = idx - i
            if not fused_slices[lookback_idx].get('adjusted'):
              log.info(f'[fused_length] idx={idx}, slice_idx={slice_idx}, name={name}, fusion_length={fusion_length}, is_consecutive={is_consecutive}, is_same_name={is_same_name}')
              fused_slices[lookback_idx]['fused_length'] = fusion_length
              shortened_dur = int(fused_slices[lookback_idx]['dur'] / fusion_length)
              ts_offset = int(shortened_dur * (fusion_length - i -1))
              log.info(f'[fused_length] name={fused_slices[lookback_idx]["name"]}, lookback_idx={lookback_idx} ts {fused_slices[lookback_idx]["ts"]} -> {fused_slices[lookback_idx]["ts"] + ts_offset}, dur {fused_slices[lookback_idx]["dur"]} -> {shortened_dur}')
              fused_slices[lookback_idx]['ts'] += ts_offset
              fused_slices[lookback_idx]['dur'] = shortened_dur
              fused_slices[lookback_idx]['adjusted'] = True  # only apply this logic once

        # We have found that this is not a fused kernel so we reset our fusion length to 1 meaning no fusion.
        fusion_length = 1





def add_fused_id(op, **kwargs):
  test_mode = os.environ.get('TEST_MODE')
  if 'ops' not in op or 'resources' not in op:
    return
  if test_mode:
    # This function uses op['id'] key but this doesn't exist in test mode because it is random. todo: Fix this eventually.
    return
  log.debug(f'[add_fused_id] Processing: {op.get("idx")}, {op["name"]}, {op["type"]}')
  for res_name in op['resources'].keys():
    prev_last_slice = None
    fused_ops = []
    # Loop over ops comparing the starts/ends of the op's slices and check if the slices are the same. Sameness means kernel fusion, because it performs a task in both ops, aka. the slice crosses the typical boundary between ops.
    for idx, sub_op in enumerate(op['ops']):
      if 'resources' not in sub_op or res_name not in sub_op['resources']:
        continue
      sub_op_slices = sub_op['resources'][res_name].get('slices',[])
      if len(sub_op_slices) < 2:
        continue
      first_slice = sub_op_slices[0]
      last_slice = sub_op_slices[-1]

      # Look for fusion at the start of the op
      if prev_last_slice == first_slice:
        sub_op['fused_start'] = True
      prev_last_slice = last_slice

      # Look for fusion at the end of the op
      next_first_slice = None
      is_last_op = idx + 1 >= len(op['ops'])
      if not is_last_op:
        next_op = op['ops'][idx + 1]
        if res_name in next_op['resources']:
          next_first_slice = next_op['resources'][res_name]['slices'][0]
        else:
          next_first_slice = None
      if next_first_slice == last_slice:
        sub_op['fused_end'] = True
        # Add fused ids
        fused_ops.append(next_op)
        fused_ops.append(sub_op)
        fused_ids = [fused_op['id'] for fused_op in fused_ops]
        for fused_op in fused_ops:
          fused_op['fused_ids'] = list(set(fused_ids))
      else:
        fused_ops = []

      # TODO: trigger fused ID when last position

