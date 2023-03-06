import traceback
import inspect
import orjson
import torch

from IPython import embed
from pprint import pprint

from hotline.hotline import *

was_step_called = False

is_annotate_enabled = True
def disable_annotate():
  """Call this function right after importing hotline to prevent hotline.annotate from doing anything so that no additional annotations are put into the trace.

  Usage:
  import hotline
  hotline.h_annotate.disable_annotate()
  """
  global is_annotate_enabled
  is_annotate_enabled = False


def _remove_manual_annotations(slices):
  return [slice for slice in slices if not slice['name'].startswith('hid=')]


def remove_manual_annotations(input_trace_file):
  # Read trace
  with open(input_trace_file, "rb") as f:
    trace = orjson.loads(f.read())

  # Remove slices that include "hotline id" because they are manual annotations
  trace['traceEvents'] = _remove_manual_annotations(trace['traceEvents'])

  # Write trace
  with open(input_trace_file, 'wb') as f:
    f.write(orjson.dumps(trace))  # 4x faster


def get_manual_annotation_by_idx(tp_manual, idx):
  manual_annotation = tp_manual.query_dict(f'select * from slices where name like "hid={idx} %"')
  if manual_annotation:
    return manual_annotation[0]
  else:
    return {}


def get_annotation_idx_by_name(tp_manual, name):
  name = name.lower()
  manual_annotation = tp_manual.query_dict(f'select * from slices where LOWER(slices.name) like "hid=% {name}%"')
  if manual_annotation:
    name = manual_annotation[0]['name']
    idx = int(name.split(' ')[0].split('=')[1])
    return idx


class HotlineAnnotate:
  """
  Usage Examples:
  with hotline.annotate('linear'):
    x = self.linear(x)

  with hotline.annotate(module._get_name()):
    x = module(x)

  for idx, layer in enumerate(self.layers):
    with hotline.annotate(f'Layer{idx}'):
      x = layer(x)

  x = hotline.annotate_module_list(self.nn_operations, x)
  """
  def __init__(self):
    self.step()

  def step(self):
    """Reset the counter to 1 after each training iteration step.

    Important: Run this after torch_profiler.step() b/c it will trigger hotline.analyze to run if it's time and before hotline.annotate.step() resets our gathered annotation model.
    """
    self.idx = 1
    global was_step_called
    was_step_called = True
    self.reset()

  def reset(self):
    self.top_op = {
      'name': 'top',
      'type': 'top',
      'ops': []
    }
    self.parents = [self.top_op]

  def print_tree(self):
    h_tree.pre_order_depth_first(self.top_op, h_print.print_op_in_tree)
    # pprint(self.top_op)

  def annotate_module_list(self, module_list, x):
    """Annotate each module in a module list.

    Usage Example:
    out = hotline.annotate_module_list(self.nn_operations, input)
    """
    for module in module_list:
      module_name = module.__class__.__name__  # ex: Linear
      with self.make_wrapper()(module_name):
        x = module(x)
    return x

  def make_wrapper(self):
    class Wrapper:
      def __init__(self, name, preserve_name=False):
        if name.startswith('ignore'):
          # You can prefix an annotation name with "ignore" to prevent it from being used.
          self.ignore = True
          return
        else:
          self.ignore = False

        if preserve_name:
          self.annotation_name = name
        else:
          self.annotation_name = f'hid={self.outer.idx} {name}'

        # Get the traceback
        traceback_list = traceback.extract_stack()
        # Exclude the bottom-most frame
        traceback_list = traceback_list[:-1]
        traceback_str = ''.join(traceback.format_list(traceback_list))

        # Get the caller's file name and line number
        caller_frame = inspect.currentframe().f_back
        source_file_name = caller_frame.f_code.co_filename
        # Get the full absolute path of the caller's file
        source_file_name = os.path.abspath(source_file_name)
        source_file_num = caller_frame.f_lineno

        self.op = {
            'idx': self.outer.idx,
            'name': name,
            'type': name,
            'is_model_op': True,
            'source_stack_trace': traceback_str,
            'source_file_name': source_file_name,
            'source_file_num': source_file_num,
        }
        self.outer.idx += 1

      def __enter__(self):
        if self.ignore:
          return
        # make this op a child of the parent
        if 'ops' in self.outer.parents[-1]:
          self.outer.parents[-1]['ops'].append(self.op)
        else:
          self.outer.parents[-1]['ops'] = [self.op]
        self.outer.parents.append(self.op)  # make the tree 1 level deeper
        if is_annotate_enabled:
          self.torch_annotation = torch.autograd.profiler.record_function(self.annotation_name)
          return self.torch_annotation.__enter__()

      def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ignore:
          return
        self.outer.parents.pop()  # make the tree 1 level less deep
        if is_annotate_enabled:
          return self.torch_annotation.__exit__(exc_type, exc_val, exc_tb)

    Wrapper.outer = self
    Wrapper.step = self.step
    Wrapper.print_tree = self.print_tree
    Wrapper.annotate_module_list = self.annotate_module_list
    return Wrapper


class HotlineAnnotateModule(torch.nn.Module):
  """
  Annotate in __init__ definition rather than in forward() function.

  INCOMPLETE
  TODO: Need to place in the correct location of the ops model tree, hopefully with and idx.
  TODO: Need to support nn.Sequential and nn.ModuleList

  Usage Example:
    Replace this code:
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes * block.expansion, stride),
            norm_layer(planes * block.expansion),
        )
    With this code:
        downsample = nn.Sequential(
            hotline.HotlineAnnotateModule('conv', conv1x1(self.inplanes, planes * block.expansion, stride)),
            hotline.HotlineAnnotateModule('batch_norm', norm_layer(planes * block.expansion)),
        )
  """
  def __init__(self, annotation_name, base_class, *args, **kwargs):
    super(HotlineAnnotateModule, self).__init__()
    self.base_class = base_class
    self.torch_annotation = torch.autograd.profiler.record_function(annotation_name)

  def forward(self, *args, **kwargs):
    self.torch_annotation.__enter__()  # start the annotation
    out = self.base_class.forward(*args, **kwargs)  # apply the original forward function
    self.torch_annotation.__exit__(None, None, None)  # end the annotation
    return out
