import re

from IPython import embed

from hotline.hotline import *

import hotline.slice as h_slice

def shorten_name(name, length=29):
  if len(name) > length:
    name = name[:length] + '...'
  name = name.replace('/','|')
  name = name.replace('\\','|')
  return name


def rename_slice(name):
  rules = [
    {
      'match_pattern': re.compile('\(\d+\): '), # Match examples: '(188): ', '(1): ,'
      'logic_name': 'line of code'
    },
    {
      'match_pattern': '<built-in function', # Example: <built-in function print>
      'logic_name': 'built-in function'
    },
    {
      'match_pattern': '<built-in method', # Example: <built-in method acquire of _multiprocessing.SemLock object at 0x7f86f5bc91f0>
      'logic_name': 'built-in method'
    },
    {
      'match_pattern': 'autograd::', # Example: torch::autograd::AccumulateGrad
      'logic_name': 'remove'
    },
    {
      'match_pattern': 'engine::', # Example: autograd::engine::evaluate_function: MeanBackward1
      'logic_name': 'remove'
    },
    {
      'match_pattern': 'evaluate_function: ', # Example: autograd::engine::evaluate_function: MeanBackward1
      'logic_name': 'remove'
    },
    {
      'match_pattern': 'torch::', # Example: torch::autograd::AccumulateGrad
      'logic_name': 'remove'
    },
    # Example: void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, at::detail::Array<char*, 3> >(int, at::native::CUDAFunctor_add<float>, at::detail::Array<char*, 3>)
    {
      'match_pattern': 'void ',
      'logic_name': 'remove'
    },
    {
      'match_pattern': 'at::',
      'logic_name': 'remove'
    },
    {
      'match_pattern': 'native::',
      'logic_name': 'remove'
    },
    {
      'match_pattern': 'detail::',
      'logic_name': 'remove'
    },
    {
      'match_pattern': '(anonymous namespace)::',
      'logic_name': 'remove'
    },
    # Example from Transformer: cunn_SpatialSoftMaxBackward<float, float, float, LogSoftMaxBackwardEpilogue>(float*, float*, float*, unsigned int, unsigned int, unsigned int)
    {
      'match_pattern': 'float',
      'logic_name': 'remove'
    },
    {
      'match_pattern': 'unsigned',
      'logic_name': 'remove'
    },
    {
      'match_pattern': 'int',
      'logic_name': 'remove'
    },
    {
      'match_pattern': ',',
      'logic_name': 'remove'
    },
    {
      'match_pattern': '*',
      'logic_name': 'remove'
    },
    {
      'match_pattern': 'detail::',
      'logic_name': 'remove'
    },
    {
      'match_pattern': '(',
      'logic_name': 'remove'
    },
    {
      'match_pattern': ')',
      'logic_name': 'remove'
    },
    {
      'match_pattern': '< ',
      'logic_name': 'remove'
    },
    {
      'match_pattern': ' >',
      'logic_name': 'remove'
    },
    {
      'match_pattern': '<>',
      'logic_name': 'remove'
    },
    {
      'match_pattern': 'Backward',
      'logic_name': 'remove'
    },
  ]
  logic_name = None

  # Find the first matching pattern to identify which logic we should apply
  for rule in rules:
    is_match = False
    if isinstance(rule['match_pattern'], re.Pattern):
      is_match = re.search(rule['match_pattern'], name)
    elif isinstance(rule['match_pattern'], str):
      is_match = rule['match_pattern'] in name

    if is_match:
      logic_name = rule['logic_name']
    else:
      continue

    # Apply renaming logic
    if logic_name == 'line of code':
      # Input example: 'torch/utils/data/dataloader.py(1173): _get_data'
      name = name.split("/")[-1]  # dataloader.py(1173): _get_data
      file_name = name.split('(')[0]  # dataloader.py(1173): _get_data
      func_name = name.split('): ')[1]  # dataloader.py(1173): _get_data
      func_name = func_name.lstrip('<').rstrip('>')  # remove <>
      file_name = file_name.lstrip('<').rstrip('>')  # remove <>
      name = f'{func_name} {file_name}'  # dataloader.py _get_data

    elif logic_name == 'built-in function':
      # Input example: <built-in function print>
      name = name.replace('<built-in function ', '')  # print>
      name = name.rstrip('>')  # print

    elif logic_name == 'built-in method':
      # Input example: <built-in method acquire of _multiprocessing.SemLock object at 0x7f86f5bc91f0>
      name = name.replace('<built-in method ', '')  # acquire of _multiprocessing.SemLock object at 0x7f86f5bc91f0>
      func_name = name.split(" of ")[0]  # acquire
      rest = name.split(" of ")[1]  # _multiprocessing.SemLock object at 0x7f86f5bc91f0>
      rest = rest.split(".")[-1]  #  SemLock object at 0x7f86f5bc91f0>
      obj_name = rest.split(" object at ")[0]  # SemLock
      name = f'{func_name} {obj_name}'  # SemLock acquire

    elif logic_name == 'remove':
      # Input example: autograd::engine::evaluate_function: MeanBackward1
      name = name.replace(rule['match_pattern'], '')  # MeanBackward1

  return name.strip()


def rename_op(op, **kwargs):
  # RNN-T specific name improvements
  if op['name'] == 'rnn' and op['type'] == 'LSTM':
    # reverse it because LSTM is more helpful
    op['type'] = 'rnn'
    op['name'] = 'LSTM'
  elif op['name'] == 'conv' and op['type'] == 'MaskConv':
    # reverse it because MaskConv is more helpful
    op['type'] = 'conv'
    op['name'] = 'MaskConv'
  elif op['name'] == 'module' and op['type'] == 'BatchNorm1d':
    # reverse it because BatchNorm1d is more helpful
    op['type'] = 'module'
    op['name'] = 'BatchNorm1d'
  elif op['name'] == 'module':
    # Anytihng is more helpful than "module"
    op['name'] = op['type']
    op['type'] = 'module'


def calc_predominant_name(slices, minimum_depth=True):
  if len(slices) <= 1:
    return rename_slice(slices[0]['name'])

  if minimum_depth:
    slices = h_slice.get_slices_at_depth(slices, 'minimum')

  count = {}  # count number of slices with the same name
  dur_sum = {}  # sum duration of slices with the same name
  for slice in slices:
    dur_sum[slice['name']] = dur_sum.get(slice['name'], 0) + slice['dur']  # set or increase the sum for this name
    count[slice['name']] = count.get(slice['name'], 0) + 1

  count_total = sum(count.values())
  dur_total = sum(dur_sum.values())
  dur_percent = {} # percent of total duration for slices with the same name
  for name, sum_ in dur_sum.items():
    # if dur_total == 0:
    #   dur_percent[name] = 1.0
    # else:
    if dur_total == 0:  # don't divide by zero
      dur_percent[name] = 0
    else:
      dur_percent[name] = sum_ / dur_total

  # Penalize those with high counts. This is so that a single long op does not get edged out by many small ops. This reflects our belief that single long ops are usually more interesting. This was added to make "scatter (24.4%)" in ResNet50 forward pass the predominant name when it edged out by "_named_members (25.7%)".
  penalized_percent = {}
  for name, count_ in count.items():
    penalized_percent[name] = dur_percent[name] - count_ / count_total
  top_penalized_percent = max(penalized_percent.values())

  # Tie break by count if more than 1 of the same max value
  tie_break_needed = len([val for val in dur_percent.values() if val == top_penalized_percent]) > 1
  if tie_break_needed:
    top_dur_name = [name for name, c in count.items() if c == max(count.values())][0]
  else:
    top_dur_name = [name for name, val in penalized_percent.items() if val == top_penalized_percent][0]

  top_dur_percent = dur_percent[top_dur_name]
  if top_dur_percent < 0.01:
    # Two decimal places if less than 1%
    top_dur_percent_str = "{:.2f}".format(top_dur_percent*100) + '%'
  else:
    # Zero decimal places if greater than 1%
    top_dur_percent_str = "{:.0f}".format(top_dur_percent*100) + '%'

  op_count_minus_1 = len(dur_percent.keys()) - 1  # Unique names minus one because of the format "first_name and N others" needs N-1.
  s = 's' if op_count_minus_1 > 1 else ''

  if top_dur_percent == 1.0 or op_count_minus_1 == 0:
    # When top name is 100% of runtime just use that name OR there's only one op count.
    predominant_name = rename_slice(top_dur_name)
  elif sum(dur_sum.values()) == 0 and op_count_minus_1 > 0:
    # Handle when all slices have dur 0. Remove (%) from name.
    predominant_name = f'{rename_slice(top_dur_name)} and {op_count_minus_1} other{s}…'
  else:
    predominant_name = f'{rename_slice(top_dur_name)}({top_dur_percent_str}) and {op_count_minus_1} other{s}…'

  return predominant_name


def match(word, compare_to_list):
  for compare_to in compare_to_list:
    if isinstance(compare_to, re.Pattern):
      is_match = re.search(compare_to, word.lower())
    elif isinstance(compare_to, str):
      is_match = compare_to.lower() in word.lower()
    if is_match:
      return compare_to  # return the matched string/pattern
  return False


def add_predominant_name(op, **kwargs):
  if 'resources' not in op or not op['type'] == 'generated':
    return

  log.debug(f'[add_predominant_name] Processing: {op.get("idx")}, {op["name"]}, {op["type"]}')

  slices = h_slice.get_slices(op, filter_by_resource='cpu')  # NOTE: Currently only use cpu ops to generate names because they often have easier to understand names. Resource specific names get added later.
  predominant_name = h_name.calc_predominant_name(slices)

  log.debug(f'{op["name"]} --> {predominant_name}')
  op['name'] = predominant_name
  op['pretty_name'] = predominant_name


def rename_model_pass_op(op, **kwargs):
  log.debug(f'[rename_model_pass_op] Processing: {op.get("idx")}, {op["name"]}, {op["type"]}')
  if 'is_model_pass' in op:
    op['pretty_name'] = f'{op["is_model_pass"]} Pass'

def add_pretty_name(op, **kwargs):
  log.debug(f'[add_pretty_name] Processing: {op.get("idx")}, {op["name"]}, {op["type"]}')
  if 'pretty_name' in op:
    # Already got a pretty name, nothing to do
    return
  if op['name'].isnumeric():
    op['pretty_name'] = f'{op["type"]}-{op["name"]}'
  else:
    op['pretty_name'] = op['name']
  op['pretty_name'] = op['pretty_name']


def standardize_name(name):
  name = name.lower()
  remove_list = ['1d', '2d', '3d', '_']
  for remove in remove_list:
    name = name.replace(remove, '')
  return name


def fuzzy_op_match(slice, op, is_backward, tp):
  """Decide whether the current slice matches the target op.

  Args:
    slice: perfetto slice format
    op: pytorch module op format
  """
  # Clean up names to be more matching
  slice_name = standardize_name(slice['name'])
  compare_to = standardize_name(op['type'])

  # Remap names to known matches
  op_name_map = {
    # formatting of this dict:
    #   'lowercase type in pytorch model def': 'name in trace',

    'linear': ['addmm', 'mmbackward', 'sgemm', 'matmul'],
    'adaptiveavgpool': ['mean'],
    'avgpool': ['mean'],

    # Forward ResNet 4xGPU
    'conv': ['winograd', 'implicit_convolve_sgemm', re.compile('cudnn.*x')],  # ex. volta_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1
    'batchnorm': ['cudnn::bn'],
    'maxpool': ['max_pool'],

    # Forward RNN-T 4xGPU
    'hardtanh': ['clamp'],
    'relu': ['clamp'],

    # Forward WMT Transformer
    'batchmatmul': ['bmm', 'addmm'],  # Add ability to ignore "baddbmm"
    'batchedaddmatmul': ['baddbmm'],
    'tile': ['reshape'],
    'concatenate': ['cat'],
  }
  # Backward Transformer
  if is_backward:
    op_name_map['multiheadattention'] = ['tbackward']  # TODO can be removed now?
  # Forward RNN-T 4xGPU
  op_name_map['batchnorm'].append('batch_norm_transform_input_channels_last_kernel')
  if not is_backward:
    op_name_map['lstm'] = ['void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4>)']  # check is this needed? probably not
  else:
    op_name_map['lstm'] = ['rnn']

  if compare_to.lower() in op_name_map:
    compare_to_list = op_name_map[compare_to] + [compare_to.lower()]
  else:
    compare_to_list = [compare_to]

  # If current is a cudaLaunchKernel, compare the target to the name of the launched kernel
  if slice['name'] == 'cudaLaunchKernel':
    launch_slice_id = slice["slice_id"]
    flow_slice = tp.query_dict(f'SELECT * FROM DIRECTLY_CONNECTED_FLOW({launch_slice_id});')[0]
    kernel_slice = tp.query_dict(f'SELECT * FROM slice WHERE id = {flow_slice["slice_in"]};')[0]
    slice_name = kernel_slice['name'].lower()

  fused_names = [  # also see fused.py
    re.compile('cudnn.*x.*relu'),  # Example conv+relu fused kernel found in ResNet50 forward: volta_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1
    re.compile('cat.*dog'),  # for tests
    re.compile('fish.*turtle'),  # for tests
  ]
  is_fused = match(slice_name, fused_names)

  is_match = match(slice_name, compare_to_list)
  if is_match and is_fused:
    slice['possibly_is_fused'] = True  # A fused kernel may or may not be utilized for all it's affordances. There may be a conv_relu where the relu isn't present in the pytorch model and the relu is not applied.
    log.debug(f'FUSED FOUND: {slice_name} <-> {is_fused}')

  log.debug(f"[fuzzy] Match={is_match}, (Slice <-> Matched / Type / Mapped) {slice_name[:30]} <-> {op['name']} / {op['type']} / {is_match}")

  return is_match, is_fused


def set_resource_specific_name(op, **kwargs):
  log.debug(f'[set_resource_specific_name] Processing: {op.get("idx")}, {op["name"]}, {op["type"]}')
  if 'ops' in op or 'resources' not in op or op['type'] != 'generated':
    # Only apply to leaf nodes which have been generated
    return
  for res_name, res in op['resources'].items():
    if 'slices' in res:
      res['res_name'] = calc_predominant_name(res['slices'])
      log.debug(f'[set_resource_specific_name] {op["name"]} --> {res["res_name"]}')
