import inspect
import os.path
import copy
import sys
import torchinfo

from hotline.hotline import *

def convert_model_to_heirachical_dict(model, device, dataloader):
    """ Convert DNN model from pytorch module to heirachical dict format.
    Tested to work with torchinfo==1.7.0
    """
    data_batch = next(iter(dataloader))
    data_shape = []
    if isinstance(data_batch, dict):
      for key in data_batch.keys():
        data_shape.append(tuple(data_batch[key].shape))
    else:
      data_shape = tuple(data_batch[0].shape)
    log.info(f'data_batch shape for torch_info: {data_shape}')

    try:
      torch_info = torchinfo.summary(model, depth=sys.maxsize, mode='train', device=device, verbose=0)
    except Exception as e:
      log.error(f'torchinfo fallback failed')
      raise e

    # Function to convert torchinfo object to hotline dictionary
    def torchinfo_op_to_hotline_dict(op_info):
      op = {
        'name': op_info.var_name, # ex. 0 or layer1
        'type': op_info.class_name, # ex. Conv2d or Sequential
        'is_model_op': True,
        # Non-essential fields:
        # 'input_size': op_info.input_size, # ex. [32, 1, 28, 28]
        # 'output_size': op_info.output_size, # ex. [32, 32, 28, 28]
        # 'kernel_size': op_info.kernel_size, # ex. [5, 5]
        # 'trainable_params': op_info.trainable_params, # ex. 832
        # 'num_params': op_info.num_params, # ex. 832
        # 'param_bytes': op_info.param_bytes, # ex. 3328
        # 'output_bytes': op_info.output_bytes, # ex. 3211264
        # 'macs': op_info.macs, # ex. 20873216 multiply–accumulate (MAC) operations
      }
      # # Extra torchinfo fields:
      # if op_info.is_leaf_layer:
        # op['repr'] = repr(op_info.module) # ex. 'Linear(in_features=1000, out_features=10, bias=True)'
      # """Example:
      #   {'bias':   {'kernel_size': [10], 'num_params': 10},
      #    'weight': {'kernel_size': [1000, 10], 'num_params': 10,000'}
      # """
      #   if hasattr(op_info, 'inner_layers'):
      #     for key, value in op_info.inner_layers.items():
      #       op['inner_layers_' + key] = value
      return op

    # Recursively convert torchinfo heirarchy to hotline heirarchy
    model_ops = []
    def recursive_add_op(op_info, ops):
      op = torchinfo_op_to_hotline_dict(op_info)
      if hasattr(op_info, 'children'):
        for sub_op_info in op_info.children:
          if op_info.depth + 1 != sub_op_info.depth:
            continue
          if 'ops' not in op:
            op['ops'] = []
          recursive_add_op(sub_op_info, op['ops'])
      ops.append(op)
    recursive_add_op(torch_info.summary_list[0], model_ops)


    # # Add extra torchinfo to top-level hotline dictionary
    # """ Example:
    # ==============
    # Total params: 3,199,106
    # Trainable params: 3,199,106
    # Non-trainable params: 0
    # Total mult-adds (M): 443.11
    # # ==============  # only available if providing data_shape or data_batch
    # # Input size (MB): 0.10
    # # Forward/backward pass size (MB): 9.89
    # # Params size (MB): 12.80
    # # Estimated Total Size (MB): 22.79
    # # ==============
    # """
    # ops[0]['torchinfo_stats_str'] = repr(torch_info).split("\n")[-11:-1]
    # torchinfo_stats = vars(copy.copy(torch_info))
    # torchinfo_stats.pop('summary_list', None)
    # torchinfo_stats.pop('formatting', None)
    # ops[0]['torchinfo_stats'] = torchinfo_stats
    torchinfo_stats_str = repr(torch_info).split("\n")[-4:-1]

    return model_ops, torchinfo_stats_str


def get_module_location(module):
    """Get the filename and line number where a PyTorch module is defined."""
    # Get the module that contains the module
    module_module = inspect.getmodule(module)
    # Get the filename of the module
    filename = inspect.getfile(module_module)
    # Get the full absolute filepath
    filepath = os.path.abspath(filename)
    # Get the source code of the module
    source = inspect.getsource(module_module)
    # Split the source code into lines
    lines = source.split('\n')
    # Search for the line that contains the module definition
    for i, line in enumerate(lines):
        if 'class ' + module.__class__.__name__ + '(' in line:
            return filepath, i + 1
