from IPython import embed
import cpath_tracer

model_def = [
  {'name': 'Fusion Test', 'type': 'Animals', 'ops': [
    # {'name': 'safari', 'type': 'safari', 'ops': [
      {'name': 'bird', 'type': 'bird'},
      {'name': 'cat', 'type': 'cat'},
      {'name': 'dog', 'type': 'dog'},
      {'name': 'fish', 'type': 'fish'},
      {'name': 'turtle', 'type': 'turtle'},
      {'name': 'rabbit', 'type': 'rabbit'},
    ]},
  # ]},
]
model_name = 'Test Fused Model'
output_dir = 'actual_results/Test_FusedModel'
input_trace_file = 'traces/Test_FusedModel.pt.trace.json'

# query = ['SELECT * FROM slice WHERE cat != "Kernel" AND ts BETWEEN 1649470253311129000 AND 1649470253341007000 AND depth = 0 AND track_id = 4;']
query = ['SELECT * FROM slice ORDER BY ts;']

tracer = cpath_tracer.Tracer(input_trace_file, output_dir, model_def, model_name, resource_types=['cpu'])

tracer.add_op_idxs()
tracer.find_model_in_slices(query)
tracer.add_gpu_slices()
tracer.convert_leaf_slices_to_ops()
# tracer.export_traces()
tracer.add_random_ids()
tracer.add_fused_ids_start_end_flags()
tracer.adjust_fused_kernel_times()
tracer.add_times()
tracer.add_relative_timings()
tracer.add_longest_parent_per_resource_types()
tracer.add_pretty_names()
tracer.export_hierarchical_model()

