import sys
import json
import os
import time
import shutil
import torch
import inspect
import humanize
import concurrent.futures

import pandas as pd
import numpy as np
from pathlib import Path
from IPython import embed
from pprint import pprint
from datetime import datetime, timedelta
from tabulate import tabulate
from distutils.dir_util import copy_tree

import hotline
from hotline.util import log, decorator, Wrapper # needs to be first
import hotline.slice as h_slice  # needs to be second
import hotline.op as h_op
import hotline.time as h_time
import hotline.name as h_name
import hotline.read as h_read
import hotline.util as h_util
import hotline.tree as h_tree
import hotline.print as h_print
import hotline.write as h_write
import hotline.fused as h_fused
import hotline.torch as h_torch
import hotline.denoise as h_denoise
import hotline.accuracy as h_accuracy
import hotline.annotate as h_annotate
import hotline.perfetto as h_perfetto
import hotline.detect_model as detect_model
import hotline.detect_training_loop as detect_training_loop


def analyze(torch_model, dataloader, run_name='', output_dir='./', ui_dir=None, worker_name='worker0', metadata={}, test_accuracy=False):
  """ Usage:  with torch.profiler.profile(... on_trace_ready=hotline.analyze(model) ... ) as p: """
  if torch_model:
    model_name = torch_model.__class__.__name__
  else:
    model_name = 'unknown'
  if not run_name:
    run_name = model_name

  output_dir = os.environ.get('HOTLINE_OUTPUT_DIR', output_dir)
  output_dir = f'{output_dir}/results'
  output_dir = os.path.normpath(output_dir)
  ui_dir = ui_dir if ui_dir is not None else output_dir+'/ui'  # default to output_dir/ui
  ui_dir = os.environ.get('HOTLINE_UI_DIR', ui_dir)  # example setting: /home/dans/hotline/ui

  # Get the caller's file name and line number
  caller_frame = inspect.currentframe().f_back
  source_file_name = caller_frame.f_code.co_filename
  # Get the full absolute path of the caller's file
  source_file_name = os.path.abspath(source_file_name)
  source_file_num = caller_frame.f_lineno

  def handler_fn(prof) -> None:
    """ Inspired by https://pytorch.org/docs/stable/_modules/torch/profiler/profiler.html#tensorboard_trace_handler """
    if not os.path.isdir(output_dir):
      try:
        os.makedirs(output_dir, exist_ok=True)
      except Exception:
        raise RuntimeError("Can't create directory: " + output_dir)

    t = time.time()
    # t_ms = int(t * 1000)
    t_ms = ''
    print(f"The current time in milliseconds: {t_ms}")

    file_name = f"{run_name}.{t_ms}{worker_name}.pt.trace.json"
    trace_filepath = os.path.join(output_dir, file_name)
    log.info(f'Writing trace to: {trace_filepath}')
    prof.export_chrome_trace(trace_filepath)

    log.info(f'Starting Hotline for run: {run_name}')
    hotline = Hotline(trace_filepath, output_dir, ui_dir, torch_model, run_name, model_name, dataloader, metadata=metadata, backend='torch', test_accuracy=test_accuracy, source_file_name=source_file_name, source_file_num=source_file_num)
    hotline.analyze()

    runtime = time.time() - t
    print(f'hotline.analyze() runtime: {runtime:.2f}s')

    # Start webserver to serve results
    # from pathlib import Path
    # LOCATION_OF_HOTLINE_INSTALL = Path.cwd()
    # $LOCATION_OF_HOTLINE_INSTALL./node_modules/.bin/parcel src/index.html --port 7234

  return handler_fn

class Hotline:
  """Identify backward pass operations using the model architecture as guide."""
  def __init__(self, trace_filepath, output_dir, ui_dir, torch_model, run_name, model_name, dataloader, metadata=None, backend='torch', test=False, num_gpus=None, test_accuracy=False, source_file_name=False, source_file_num=False):
    self.trace_filepath = trace_filepath
    self.output_dir = output_dir
    self.ui_traces_path = f'{ui_dir}/dist/traces/results/{run_name}'
    self.ui_model_path = f'{ui_dir}/src/results/{run_name}.js'
    self.results_summary_csv_filepath = f'{self.output_dir}/results_summary.csv'
    self.ui_source_code_path = f'{ui_dir}/dist/traces/results/code/{run_name}'
    self.torch_model = torch_model
    self.run_name = run_name
    self.model_name = metadata.get('model', model_name)
    self.dataloader = dataloader
    self.metadata = metadata
    self.backend = backend
    if not test and self.torch_model:
      self.device = next(self.torch_model.parameters()).device
    self.fn_runtimes = []
    self.test = test
    self.is_test_accuracy = test_accuracy
    self.view_manual_annotations = False  # Feature flag (default: False) to skip deleting manual annotations so they can be inspected in perfetto when debugging accuracy code.
    self.source_file_name = source_file_name
    self.source_file_num = source_file_num

    self.resource_types = ['cpu', 'gpu']
    self.num_gpus = num_gpus if num_gpus else torch.cuda.device_count()
    log.info(f'Num GPUs: {self.num_gpus}')
    self.slice_idx = 0
    self.drop_flow_view = ''
    self.last_found_was_fused = False
    self.count_sum_greater_than_1 = 0
    self.count_sum_less_than_1 = 0
    self.test_mode = os.environ.get('TEST_MODE')

    # percent of parent runtime to be considered a tiny op
    self.tiny_op_threshold = 0.05
    self.max_generated_depth = 2
    self.remove_slice_args = True # For speedup and disk space saving at cost of less information when opened with perfetto
    self.write_model_ops_to_file = True  # For testing only


  @decorator
  def load_trace_processor(self):
      self.tp = h_perfetto.load_trace_processor(trace_bytes=self.slices_bytes)


  @decorator
  def create_perfetto_indexes(self):
      self.slice_index = h_perfetto.create_slice_index(self.tp)
      self.flow_index = h_perfetto.create_flow_index(self.tp)
      self.track_index, self.thread_index, self.process_index = h_perfetto.create_track_indexes(self.tp)


  @decorator
  def load_raw_trace(self):
      """This must execute before load_trace_processor() so that convert_ids_int_string will run to fix a weird bug."""
      self.raw_slices, self.raw_slice_index, self.slices_bytes = h_read.load_raw_trace(self.trace_filepath, remove_slice_args=self.remove_slice_args)


  @decorator
  def convert_model_to_heirachical_dict(self):
    if self.test:
      self.model_ops = self.torch_model

    else:
      self.model_ops, self.stats_str = h_torch.convert_model_to_heirachical_dict(self.torch_model, self.device, self.dataloader)
      # h_tree.post_order_depth_first(self.model_ops[0], detect_model.remove_repeating_ops)

      if self.write_model_ops_to_file:
        h_write.write_model_ops_to_file(self.model_ops, self.output_dir, self.run_name)

    # log.info('self.model_ops:')
    # pprint(self.model_ops)
    # h_tree.pre_order_depth_first(self.model_ops[0], h_print.print_op_in_tree)


  @decorator
  def print_runtime_tree(self):
    h_tree.pre_order_depth_first(self.top_op, h_print.print_op_in_tree_with_runtime)
    log.info('\n\nTraining Loop Breakdown:')
    h_tree.pre_order_depth_first(self.top_op, h_print.print_op_in_tree_with_runtime, top_level_only=True)


  def _add_op_idx(self, op, counter, **kwargs):
    if self.is_test_accuracy:
      # When testing accuracy we want to control the idx of the ops so we can compare to the manual annotations. Here we set the op idx to the idx of the manual annotation for the training loop. The other ops, the model ops have their idx set when the model is created.
      if op['type'] == 'training loop':
        idx = h_annotate.get_annotation_idx_by_name(self.tp_with_manual_annotations, op["name"])
        if idx:
          op['idx'] = idx
      if op.get('is_backward_op'):
        return

      return

    # Only apply to training loop and model ops
    if op['type'] == 'training loop' or op.get('is_model_op'):
      if op.get('is_backward_op'):
        return
      idx = getattr(self, counter)
      op['idx'] = idx
      setattr(self, counter, idx + 1)  # Increment counter with variable name of counter. Equivalent to: self.counter += 1


  @decorator
  def add_op_idx(self):
    self.op_idx = 1
    h_tree.pre_order_depth_first(self.top_op, self._add_op_idx, 'op_idx')
    # h_tree.pre_order_depth_first(self.top_op, h_print.print_op_in_tree)
    self.idx_to_op_map = {op['idx']: op for op in h_tree.DepthFirstTreeIterator(self.top_op) if 'idx' in op}
    # self.model_op_idx = 1
    # h_tree.pre_order_depth_first(self.model_ops[0], self._add_op_idx, 'model_op_idx')

  @decorator
  def detect_training_loop(self):
    # These annotations will be used later to set the filename and lineinfo on the training loop ops
    annotations = hotline.annotate.outer.top_op['ops']
    self.training_loop_ops = detect_training_loop.detect_training_loop(self.tp, annotations)


  @decorator
  def add_interesting_info_to_top_op(self):
    """ Collect interesting metrics and measurements """
    # Grab self variables safe to print in the UI's top_op details
    safe_vars = {}
    self_vars = vars(self)
    for key, value in self_vars.items():
      if isinstance(value, (int, float, str, bool)):
        safe_vars[key] = value
    self.top_op['config'] = safe_vars
    self.top_op['metadata'] = {}
    for key, value in self.metadata.items():
      if isinstance(value, (int, float, str, bool)):
        self.top_op[f'metadata.{key}'] = value

    # Add additional interesting values
    self.top_op['trace_disk_size'] = humanize.naturalsize(os.stat(self.trace_filepath).st_size)
    self.top_op['trace_event_count'] = humanize.intcomma(len(self.raw_slices))
    self.top_op['pytorch_version'] = torch.__version__
    self.top_op['gpu_model'] = torch.cuda.get_device_name(0)
    self.top_op['gpu_cuda_version'] = torch.version.cuda
    folder_size = sum(f.stat().st_size for f in Path(self.ui_traces_path).glob('**/*') if f.is_file())
    self.top_op['hotline_traces_trace_disk_size'] = humanize.naturalsize(folder_size)
    file_count = len([f for f in Path(self.ui_traces_path).rglob('*') if f.is_file()])
    self.top_op['hotline_annotation_count'] = humanize.intcomma(file_count)
    self.top_op['processed_datetime'] = datetime.now().strftime('%d/%m/%Y %H:%M:%S')

    runtime_with_profiling, runtime_without_profiling, runtime_profiling_overhead_factor = h_time.format_iteration_runtime(self.metadata.get('runtime'))
    self.top_op['runtime_without_profiling'] = runtime_without_profiling
    self.top_op['runtime_with_profiling'] = runtime_with_profiling
    self.top_op['runtime_profiling_overhead_factor'] = runtime_profiling_overhead_factor

    print('\nRuntime of hotline analyze functions:')
    total_time = h_time.print_fn_runtimes(self.fn_runtimes)
    self.top_op['hotline_analysis_time'] = h_time.format_time_as_str(total_time)


  def add_interesting_info_to_op(self, op, **kwargs):
    if 'resources' not in op and op['resources']:
      return

    # Get longest runtime and earliest ts for this op
    longest_dur = 0
    earliest_ts = float('inf')
    for res_name, res in op['resources'].items():
      time = res['time']
      if time['dur'] >= longest_dur:
        longest_res = res_name
        bound_by = 'GPU-Bound' if 'gpu' in res_name else 'CPU-Bound'
      longest_dur = max(longest_dur, time['dur'])
      earliest_ts = min(earliest_ts, time['ts'])
      # Save runtime per resource
      time['runtime_str'] = h_time.format_time_as_str(time['dur']/1000)

    op['runtime'] = longest_dur
    op['runtime_str'] = h_time.format_time_as_str(longest_dur/1000)
    op['start_timestamp'] = h_time.epoch_to_time_str(earliest_ts)
    op['recommendations'] = 'None'
    op['bound_by'] = bound_by
    op['longest_res'] = longest_res
    op['trace_event_count'] = len(h_slice.get_slices(op))


  @decorator
  def add_interesting_info_to_ops(self):
    h_tree.pre_order_depth_first(self.top_op, self.add_interesting_info_to_op)


  @decorator
  def write_hierarchical_model(self):
    dir_path = os.path.dirname(self.ui_model_path)
    if not os.path.exists(dir_path):
      log.info(f'Creating directory: {dir_path}')
      os.makedirs(dir_path)

    # Remove slices from dict because the front-end doesn't need those right now
    h_tree.pre_order_depth_first(self.top_op, h_slice.remove_slices)  # DELETE SLICES

    log.info(f'Exporting model to: {self.ui_model_path}')
    h_write.write_hierarchical_model(self.top_op, self.ui_model_path)


  @decorator
  def write_traces(self):
    log.info(f'Exporting traces to: {self.ui_traces_path}')
    shutil.rmtree(self.ui_traces_path, ignore_errors=True)
    if not os.path.exists(self.ui_traces_path):
      log.info(f'Creating directory: {self.ui_traces_path}')
      os.makedirs(self.ui_traces_path)
    slices_bytes = getattr(self, 'slices_bytes_with_manual_annotations', self.slices_bytes)
    h_tree.pre_order_depth_first(self.top_op, h_write.write_trace, self.ui_traces_path, self.run_name, self.raw_slice_index, slices_bytes, self.track_index)

  @decorator
  def write_source_codes(self):
    log.info(f'Exporting traces to: {self.ui_source_code_path}')
    shutil.rmtree(self.ui_source_code_path, ignore_errors=True)
    if not os.path.exists(self.ui_source_code_path):
      log.info(f'Creating directory: {self.ui_source_code_path}')
      os.makedirs(self.ui_source_code_path)
    h_tree.pre_order_depth_first(self.top_op, h_write.write_source_code, self.ui_source_code_path, self.run_name)


  @decorator
  def add_longest_parent_per_resource_types(self):
    h_tree.pre_order_depth_first(self.top_op, h_time.add_longest_parent_per_resource_type, self.resource_types)


  @decorator
  def add_predominant_names(self):
    h_tree.pre_order_depth_first(self.top_op, h_name.add_predominant_name)


  @decorator
  def adjust_fused_kernel_times(self):
    h_tree.pre_order_depth_first(self.top_op, h_fused.adjust_fused_kernel_time)


  @decorator
  def add_fused_ids(self):
    h_tree.pre_order_depth_first(self.top_op, h_fused.add_fused_id)


  @decorator
  def add_times(self):
    h_tree.pre_order_depth_first(self.top_op, h_time.add_time)


  @decorator
  def add_relative_timings(self):
    h_tree.pre_order_depth_first(self.top_op, h_time.add_relative_timing)
    # global count_sum_greater_than_1
    # log.info(f'[add_relative_timing] Num levels with time sum > 100%: {count_sum_greater_than_1}')
    # global count_sum_less_than_1
    # log.info(f'[add_relative_timing] Num levels with time sum < 100%: {count_sum_less_than_1}')


  @decorator
  def create_top_level_op(self):
    name = self.model_name if self.model_name else self.run_name
    if 'inference' not in name.lower():
      name = name + ' Training Iteration'
    self.top_op = h_op.create_top_level_op(self.training_loop_ops, name)

    if len(self.training_loop_ops) == 1 and self.training_loop_ops[0]['name'] == 'default':
      # We don't have a training loop, so we just use the model ops such as in the case of inference.
      self.top_op['ops'] = self.model_ops
      self.top_op['is_model_pass'] = True

    self.top_op['source_file_name'] = self.source_file_name
    self.top_op['source_file_num'] = self.source_file_num

  @decorator
  def set_resource_specific_names(self):
    h_tree.pre_order_depth_first(self.top_op, h_name.set_resource_specific_name)


  @decorator
  def add_random_ids(self):
    if self.test_mode:
      # In test mode we don't want to generate random IDs because randomness makes our testing code think something changed.
      return
    h_tree.pre_order_depth_first(self.top_op, h_op.add_random_id)


  @decorator
  def adjust_async_times(self):
    h_tree.pre_order_depth_first(self.top_op, h_time.adjust_async_time)


  @decorator
  def rename_ops(self):
    h_tree.pre_order_depth_first(self.top_op, h_name.rename_op)


  @decorator
  def sort_resources(self):
    h_tree.pre_order_depth_first(self.top_op, h_op.sort_resource)


  @decorator
  def locate_forward_backward_annotations(self):
    h_tree.pre_order_depth_first(self.top_op, detect_model.locate_forward_backward_annotations, self.num_gpus)


  @decorator
  def detect_model(self):
    multithread = False
    if multithread:
      with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_gpus*2) as self.executor:
        results = h_tree.parallel_pre_order_depth_first(self.top_op, detect_model.detect_model, self.executor, self.model_ops, self.tp)
        for result in results:
          if isinstance(result, concurrent.futures.Future):
            result.result()
    else:
      results = h_tree.pre_order_depth_first(self.top_op, detect_model.detect_model, self.model_ops, self.tp)


  @decorator
  def check_for_missing_ops(self):
    h_tree.pre_order_depth_first(self.top_op, h_op.check_for_missing_op)


  @decorator
  def delete_missing_ops(self):
    h_tree.pre_order_depth_first(self.top_op, h_op.delete_missing_ops)


  @decorator
  def add_gpu_slices(self):
    h_tree.pre_order_depth_first(self.top_op, h_perfetto.add_gpu, self.tp, self.slice_index, self.flow_index)


  @decorator
  def remove_lonely_resources(self):
    h_tree.pre_order_depth_first(self.top_op, detect_model.remove_lonely_resources)


  @decorator
  def join_short_or_same_name_ops(self):
    h_tree.post_order_depth_first(self.top_op, h_denoise.join_short_or_same_name_op, self.tiny_op_threshold)


  @decorator
  def detect_arbitrary_annotationss(self):
    h_tree.pre_order_depth_first(self.top_op, h_denoise.detect_arbitrary_annotations, self.max_generated_depth, self.tiny_op_threshold)


  @decorator
  def raise_sub_ops_of_dominant_ops(self):
    raise_op_threshold = 0.75
    h_tree.post_order_depth_first(self.top_op, h_denoise.raise_sub_ops_of_dominant_op, raise_op_threshold)


  @decorator
  def add_pretty_names(self):
    h_tree.pre_order_depth_first(self.top_op, h_name.add_pretty_name)


  @decorator
  def rename_model_pass_ops(self):
    h_tree.pre_order_depth_first(self.top_op, h_name.rename_model_pass_op)


  @decorator
  def split_default_resource(self):
    h_tree.pre_order_depth_first(self.top_op, h_op.split_default_resource, self.track_index)



  def results_summary(self):
    results_summary = {}
    for _key in ['config','metadata']:
      for key, value in self.top_op[_key].items():
        if isinstance(value, (int, float, str, bool)):
          results_summary[f'{_key}.{key}'] = value
    for key, value in self.top_op.items():
      if isinstance(value, (int, float, str, bool)):
        results_summary[f'{key}'] = value

    results_summary['results_size_on_disk'] = humanize.naturalsize(os.stat(self.ui_model_path).st_size)

    new_df = pd.DataFrame(results_summary, index=[0])

    log.info("\nTable of this run's results:")
    kv_table = [{'key': k, 'value': v} for k, v in results_summary.items()]
    print(tabulate(kv_table, tablefmt='orgtbl'))

    # Write all results to csv
    filepath = self.results_summary_csv_filepath
    if os.path.isfile(filepath):
      df = pd.read_csv(filepath)
    else:
      df = pd.DataFrame()
    df = df.append(new_df)
    df.to_csv(filepath, index=False)

    h_print.print_interest_results_table(df)


  @decorator
  def create_model_from_annotations(self):
    self.model_ops = []
    if self.test:
      self.model_ops = self.torch_model
    else:
      ops = hotline.annotate.outer.top_op['ops']
      if not ops:
        return
      forward_op = [op for op in ops if op['name'].lower() == 'forward']
      if forward_op:
        # This removes the training loop ops from the model
        self.model_ops = forward_op[0].get('ops', [])
      else:
        # Otherwise, use all the ops in the definition
        self.model_ops = ops

      if self.write_model_ops_to_file:
        h_write.write_model_ops_to_file(self.model_ops, self.output_dir, self.run_name)

    # h_tree.pre_order_depth_first(self.model_ops[0], h_op.print_ops_as_tree)


  @decorator
  def test_accuracy_setup(self):
    if not self.is_test_accuracy:
      return

    # Load the trace with manual annotations in perfetto
    raw_slices, raw_slice_index, slices_bytes = h_read.load_raw_trace(self.trace_filepath, remove_slice_args=self.remove_slice_args)
    self.tp_with_manual_annotations = h_perfetto.load_trace_processor(trace_bytes=slices_bytes)
    self.flow_index_with_manual_annotations = h_perfetto.create_flow_index(self.tp_with_manual_annotations)
    self.slices_bytes_with_manual_annotations = slices_bytes

    # Make a copy of trace with manual annotations
    manual_trace_path = self.trace_filepath + 'with_manual_annotations.json'
    log.info('Saved trace with manual annotations to: ' + manual_trace_path)
    shutil.copyfile(self.trace_filepath, manual_trace_path)

    if self.view_manual_annotations:
      # Keep manual annotations in the output trace so it can be viewed in perfetto. The accuarcy check is also skipped.
      self.is_test_accuracy = False
    else:
      # Remove manual annotations so that hotline can process the trace as if they weren't there
      h_annotate.remove_manual_annotations(self.trace_filepath)


  @decorator
  def test_accuracy(self):
    """ Must run before write_hierarchical_model() which removes events from ops."""
    if not self.is_test_accuracy:
      return

    h_accuracy.print_manual_to_detected_mapping_table(self.tp_with_manual_annotations, self.idx_to_op_map)

    self.top_op['total_accuracy_str'] = h_accuracy.test_accuracy(self.tp_with_manual_annotations, self.idx_to_op_map, self.flow_index_with_manual_annotations, self.metadata)


  def analyze(self):
    log.info(f'run_name: {self.run_name}')
    log.info(f'Begin analyzing: {self.trace_filepath}')

    # Setup
    self.test_accuracy_setup()
    self.load_raw_trace()
    self.load_trace_processor()
    self.create_perfetto_indexes()
    self.convert_model_to_heirachical_dict()
    # self.create_model_from_annotations()  # creates self.model_ops

    # Annotation
    self.detect_training_loop()  # creates self.training_loop_ops
    self.create_top_level_op()  # creates self.top_op using self.training_loop_ops
    # h_tree.post_order_depth_first(self.top_op, detect_model.remove_repeating_ops)
    self.split_default_resource()
    self.detect_arbitrary_annotationss()
    self.locate_forward_backward_annotations()
    self.detect_model()  # finds ops with 'is_model_pass' and traverses the model and slices within that op (forward/backward)
    self.delete_missing_ops() # delete missing ops OR self.check_for_missing_ops()
    self.detect_arbitrary_annotationss()
    self.add_gpu_slices()
    self.remove_lonely_resources()
    self.delete_missing_ops()


    # Analysis
    self.add_op_idx()
    self.rename_ops()
    self.add_random_ids()
    self.add_times()
    self.adjust_async_times()
    self.add_relative_timings()
    self.raise_sub_ops_of_dominant_ops()
    self.join_short_or_same_name_ops()
    self.set_resource_specific_names()
    self.add_pretty_names()
    self.rename_model_pass_ops()
    self.add_longest_parent_per_resource_types()
    self.sort_resources()

    # Write Outputs
    # self.print_runtime_tree()
    self.test_accuracy()
    self.write_traces()
    self.write_source_codes()
    self.add_interesting_info_to_top_op()
    self.add_interesting_info_to_ops()
    self.write_hierarchical_model()
    self.results_summary()
    log.info(f'Run {self.run_name}, model {self.model_name}, done.')
