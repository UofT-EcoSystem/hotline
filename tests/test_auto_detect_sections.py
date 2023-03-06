from IPython import embed
import cpath_tracer

model_def = [
  {'name': 'Test Case', 'type': 'Test Case', 'ops': [
    {'name': 'forward', 'type': 'forward'},
  ]},
]
model_name = 'Test Case'
output_dir = 'actual_results/Auto_Detect_Sections_Test'

# input_trace_file = 'traces/perfetto_resnet_4gpu_optim.pt.trace.json'
# input_trace_file = 'traces/Auto_Detect_Sections_Test.pt.trace.json'
input_trace_file = 'traces/Raise_and_Join.json'
# input_trace_file = 'traces/Predominant_Name.json'
# input_trace_file = 'traces/Test_Sections_With_Gpu.json'
# input_trace_file = 'traces/test_set_resource_specific_names.json'

query = ['SELECT * FROM slice WHERE track_id=0 ORDER BY ts;']

tracer = cpath_tracer.Tracer(input_trace_file, output_dir, model_def, model_name, forward=True)
tracer.find_model_in_slices(query)
tracer.analyze()
