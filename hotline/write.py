import orjson
import os
import json

from hotline.hotline import *


def write_model_ops_to_file(model_ops, output_dir, run_name):
  model_ops_filepath = os.path.join(output_dir, f'{run_name}_model_ops.json')
  log.info(f'\n\nSaving model definition to file: {model_ops_filepath}\n\n')
  with open(model_ops_filepath, 'w') as f:
      json.dump(model_ops, f, indent=2)

def write_hierarchical_model(top_op, filepath):
  # Write JS module to disk
  with open(filepath, 'w') as outfile:
    outfile.write('export const model =\n')
    # json.dump([top_op], outfile, indent=2)  # slower
    json.dump([top_op], outfile)
  log.info(f'Wrote: {filepath}')

export_idx = 0
def write_trace(op, ui_traces_path, run_name, raw_slice_index, slices_bytes, track_index, **kwargs):
  # if 'ops' in op:  # only apply to non-leaf nodes
  #   return
  global export_idx
  export_idx += 1

  save_file = f'{ui_traces_path}/{run_name}.{export_idx}.pt.trace.json'

  if export_idx == 1:
    # This is the top op, save the original trace
    with open(save_file, "wb") as f:
      f.write(slices_bytes.getbuffer())
  else:
    slices = h_slice.get_slices(op)
    if slices:
      # Convert perfetto processed traces to raw traces from pytorch profiler
      raw_slices = []
      for sub_slice in slices:
        key = h_read.get_key(sub_slice, processed=True, track_index=track_index)
        try:
          raw_slice = raw_slice_index[key]
        except KeyError as e:
          log.error(f'❌❌❌ failed to lookup key: {key}')
        raw_slices.append(raw_slice)
      raw_slices = h_read.sanitize_trace(raw_slices)

      with open(save_file, 'wb') as f:
        f.write(orjson.dumps(raw_slices))  # 4x faster

  op['trace_file'] = save_file.split('/ui/dist/traces')[-1]
  op['trace_disk_size'] = humanize.naturalsize(os.stat(save_file).st_size)


def write_source_code(op, ui_source_code_path, run_name, **kwargs):
  if 'source_file_name' in op and op['source_file_name']:
    src = op['source_file_name']
    dst = f'{ui_source_code_path}{src}'
    op['ui_source_code_path'] = dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)

