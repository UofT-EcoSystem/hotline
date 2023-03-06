from hotline.hotline import *

import hotline.detect_model as detect_model



def post_order_depth_first(this_op, apply_fn, *args, parent_op=None, **kwargs):
  """Post-order tree traversal. A kind of depth-first search. Is a recursive function.

  https://en.wikipedia.org/wiki/Tree_traversal
  Used to step through the model definition in a forward order, leafs first, then parents. """
  for op in this_op['ops']:
    if 'ops' in op:
      post_order_depth_first(op, apply_fn, *args, parent_op=this_op, **kwargs)
    apply_fn(op, *args, parent_op=this_op, **kwargs)
  if this_op['type'] == 'root':
    apply_fn(this_op, *args, parent_op=this_op, **kwargs)


def pre_order_depth_first(this_op, apply_fn, *args, parent_op=None, level=-1, **kwargs):
  """Pre-order tree traversal. A kind of depth-first search. Is a recursive function.

  https://en.wikipedia.org/wiki/Tree_traversal
  Used to step through the model definition in a forward order, depth first."""
  level += 1
  results = []
  if this_op['type'] == 'root':
    results.append(apply_fn(this_op, *args, parent_op=this_op, level=level, **kwargs))
  for op in this_op['ops']:
    results.append(apply_fn(op, *args, parent_op=this_op, level=level, **kwargs))
    if 'ops' in op:
      results.extend(pre_order_depth_first(op, apply_fn, *args, parent_op=this_op, level=level, **kwargs))
  return results


def parallel_pre_order_depth_first(this_op, apply_fn, executor, model_ops, tp):
  """Pre-order tree traversal. A kind of depth-first search. Is a recursive function.

  https://en.wikipedia.org/wiki/Tree_traversal
  Used to step through the model definition in a forward order, depth first."""
  results = []
  if this_op['type'] == 'root':
    results.append(executor.submit(detect_model.detect_model, this_op, model_ops, tp))
  for op in this_op['ops']:
    results.append(executor.submit(detect_model.detect_model, op, model_ops, tp))
    if 'ops' in op:
      results.extend(parallel_pre_order_depth_first(op, apply_fn, executor, model_ops, tp))
  return results

class DepthFirstTreeIterator:
    def __init__(self, tree):
        self.tree = tree
        self.stack = [self.tree]
        self.current = None

    def __iter__(self):
        return self

    def __next__(self):
        if not self.stack:
            raise StopIteration
        self.current = self.stack.pop(0)
        if 'ops' in self.current:
            self.stack = self.current['ops'] + self.stack
        return self.current
