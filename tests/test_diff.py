from IPython import embed
import cpath_tracer

model_def = [
  {'name': 'Diff Test', 'type': 'Diff Test', 'ops': [
    {'name': 'forward', 'type': 'forward'},
  ]},
]
model_name = 'Diff Test'
output_dir1 = 'actual_results/diff_model1'
output_dir2 = 'actual_results/diff_model2'
diff_output_dir = 'actual_results/diff_output'

input_trace_file1 = 'traces/diff_test1.json'
input_trace_file2 = 'traces/diff_test2.json'

def analyze_trace(model_def, model_name, input_trace_file, output_dir):
  query = ['SELECT * FROM slice WHERE track_id=0 ORDER BY ts;']

  tracer = cpath_tracer.Tracer(input_trace_file, output_dir, model_def, model_name, forward=True)

  tracer.find_model_in_slices(query)
  tracer.add_section_op_from_slicess()
  tracer.add_gpu_slices()
  tracer.add_op_idxs()
  tracer.add_random_ids()
  tracer.add_fused_ids()
  tracer.adjust_fused_kernel_times()
  tracer.add_times()
  tracer.add_relative_timings()
  tracer.raise_sub_ops_of_dominant_ops()
  tracer.join_short_or_same_name_ops()
  tracer.detect_patterns()
  tracer.add_longest_parent_per_resource_types()
  tracer.add_pretty_names()
  tracer.add_predominant_names()
  tracer.export_traces()
  tracer.export_hierarchical_model()
  return tracer

tracer1 = analyze_trace(model_def, model_name+'1', input_trace_file1, output_dir1)
tracer2 = analyze_trace(model_def, model_name+'2', input_trace_file2, output_dir2)

tracer1.diff(tracer2, diff_output_dir)

