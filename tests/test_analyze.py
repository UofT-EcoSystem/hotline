import datetime
import orjson
import traceback
import json
import os

from IPython import embed

import hotline as h

# traces = [
#   '/home/dans/cpath/traces/test_training_loop_detection.json',
  # '/home/dans/cpath/traces/MyConvNet-MNIST.worker0.pt.trace.json',
  # '/home/dans/cpath/traces/perfetto_resnet_1gpu.pt.trace.json',
  # '/home/dans/cpath/traces/perfetto_resnet_4gpu_with_stacks_shapes_memory.pt.trace.json',
  # '/home/dans/cpath/traces/perfetto_RNN_record_shapes-True__profile_memory-False__with_stack-False.pt.trace.json',
# ]

# To get model_ops.json, set in hotline.py
# self.write_model_ops_to_file = True

tests = [
  # {
  #   'model_name': 'ConvNet',
  #   'run_name': 'ConvNet',
  #   'output_dir': 'results/ConvNet',
  #   'trace': '/home/dans/cpath/traces/MyConvNet-MNIST.worker0.pt.trace.json',
  #   'model': '/home/dans/cpath/traces/model_ops_MyConvNet-MNIST.json',
  #   'input_data_shape': None,
  #   'num_gpus': 1,
  # },

  # {
  #   'model_name': 'ResNet50-1xGPU',
  #   'run_name': 'ResNet50-1xGPU',
  #   'output_dir': 'results/ResNet50-1xGPU',
  #   'trace': '/home/dans/cpath/traces/resnet_1gpu.pt.trace.json',
  #   'model': '/home/dans/cpath/traces/model_ops_resnet50.json',
  #   'input_data_shape': None,
  #   'num_gpus': 1,
  # },

  # {
  #   'model_name': 'ResNet50-2xGPU',
  #   'run_name': 'ResNet50-2xGPU',
  #   'output_dir': 'results/ResNet50-2xGPU',
  #   'trace': '/home/dans/cpath/traces/resnet50_2gpu.pt.trace.json',
  #   'model': '/home/dans/cpath/traces/model_ops_resnet50.json',
  #   'input_data_shape': None,
  #   'num_gpus': 2,
  # },

  # {
  #   'model_name': 'ResNet50-4xGPU',
  #   'run_name': 'ResNet50-4xGPU',
  #   'output_dir': 'results/ResNet50-4xGPU',
  #   'trace': '/home/dans/cpath/traces/resnet_4gpu.pt.trace.json',
  #   # 'trace': '/home/dans/cpath/traces/resnet_4gpu_with_stacks_shapes_memory.pt.trace.json',  # Too big and slow, not needed
  #   'model': '/home/dans/cpath/traces/model_ops_resnet50.json',
  #   'input_data_shape': None,
  #   'num_gpus': 4,
  # },

  # {
  #   'model_name': 'RNN-T-4xGPU',
  #   'run_name': 'RNN-T-4xGPU',
  #   'output_dir': 'results/RNN-T-4xGPU',
  #   'trace': '/home/dans/cpath/traces/RNN_record_shapes-True__profile_memory-False__with_stack-False.pt.trace.json',
  #   'model': '/home/dans/cpath/traces/model_ops_rnn-t.json',
  #   'input_data_shape': None,
  #   'num_gpus': 4,
  # },

  {
    'model_name': 'wmt-transformer-1xGPUs-small',
    'run_name': 'current',
    'output_dir': 'results/current',
    # 'trace': '/home/dans/algorithmic-efficiency/results/current/transformer_no_attention.pt.trace.json',
    'trace': '/home/dans/algorithmic-efficiency/results/current/transformer_1gpu-small2.pt.trace.json',
    'model': '/home/dans/cpath/traces/model_ops_transformer_hand_written-small2.json',
    'input_data_shape': None,
    'num_gpus': 1,
  },

  # {
  #   'model_name': 'wmt-transformer-2xGPUs-small',
  #   'run_name': 'current',
  #   'output_dir': 'results/current',
  #   'trace': '/home/dans/cpath/traces/wmt-transformer-2xGPUs-small.pt.trace.json',
  #   # 'trace': '/home/dans/cpath/traces/wmt-transformer-2xGPUs-small-with_stack.pt.trace.json',
  #   'model': '/home/dans/cpath/traces/model_ops_wmt-transformer-small.json',
  #   'input_data_shape': None,
  #     # - batch_sizes = {'wmt': 128}
  #     # + batch_sizes = {'wmt': 2}
  #     # - nhead: int = 16,
  #     # + nhead: int = 2,
  #     # - d_hid: int = 4096,
  #     # + d_hid: int = 64,
  #     # - nlayers: int = 6,
  #     # + nlayers: int = 2,
  #   'num_gpus': 2,
  # },

  # {
  #   'model_name': 'wmt-transformer-4xGPUs',
  #   'run_name': 'current',
  #   'output_dir': 'results/current',
  #   'trace': '/home/dans/cpath/traces/wmt-transformer-4xGPUs.pt.trace.json',
  #   # 'trace': '/home/dans/cpath/traces/wmt-transformer-4xGPUs-with_stack.worker0.pt.trace.json',
  #   # 'model': '/home/dans/cpath/traces/model_ops_wmt-transformer.json',
  #   # 'model': '/home/dans/cpath/traces/model_ops_wmt-transformer-ORIGINAL.json',
  #   'model': '/home/dans/cpath/traces/model_ops_wmt-transformer-no_repeats.json',
  #   'input_data_shape': None,
  #     # - batch_sizes = {'wmt': 128}
  #     # + batch_sizes = {'wmt': 32}
  #     # nhead: int = 16,
  #     # d_hid: int = 4096,
  #     # nlayers: int = 6,
  #   'num_gpus': 4,
  # },

  # cp traces/ui_transformer_4gpu_for_paper.js ui/src/models/results/current/current.js
]

first_time = datetime.datetime.now()

for test in tests:
  print(f'\nBeginning test: {test["model_name"]}')
  # Open the model def
  with open(test['model'], 'rb') as f:
    model = orjson.loads(f.read())

  try:
    hotline = h.Hotline(test['trace'], test['output_dir'], model, test['run_name'], test['model_name'], test['input_data_shape'], backend='torch', test=True, num_gpus=test['num_gpus'])
    hotline.analyze()
  except Exception as e:
    print(f'\nFailed to analyze: {test}\n')
    print(traceback.format_exc())


this_time = datetime.datetime.now()
tdelta = this_time - first_time
print(f'TOTAL RUNTIME: {tdelta}')