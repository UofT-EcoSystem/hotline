import io
import orjson
import time
import json
import copy

from perfetto.trace_processor import TraceProcessor

from hotline.hotline import *


def get_key(slice, processed=False, track_index=None):
  """A key that can corrolated between legacy raw JSON and new trace processor JSON types"""
  if processed:
    ts = h_time.safe_time_convert(slice["ts"])
    dur = h_time.safe_time_convert(slice.get('dur',''))
    cat = slice.get('category','')
    tid = track_index[slice["track_id"]]["tid"]
  else:
    ts = slice["ts"]
    dur = slice.get('dur','')
    cat = slice.get('cat','')
    tid = slice.get('tid','')
  if dur == 0: # legacy json can have duration missing, while new json has dur=0
    dur = '' # revert to legacy
  key = f'{ts}{dur}{cat}{tid}{slice["name"]}'
  key = key.replace("None", "")
  return key


def convert_ids_int_string(slices):
  for slice in slices:
    if 'id' in slice:
      slice['id'] = str(slice['id'])


def convert_negative_tids_to_positive(slices):
  for slice in slices:
    if 'tid' in slice and isinstance(slice['tid'], int):
      slice['tid'] = abs(slice['tid'])


def remove_args(slices):
  [slice.pop('args', None) for slice in slices]


def load_raw_trace(input_trace_file, remove_slice_args=False):
  with open(input_trace_file, "rb") as f:
    raw_slices = orjson.loads(f.read())

  if isinstance(raw_slices, dict) and 'traceEvents' in raw_slices:
    raw_slices = raw_slices.pop('traceEvents', None) # Perfetto doesn't want this format produced by PyTorch
  convert_ids_int_string(raw_slices) # Convert IDs from int to string. Without this perfetto fails to JSON load trace with IDs stored as integers.
  convert_negative_tids_to_positive(raw_slices) # Convert negative 'tid' values to positive. Without this perfetto combines together the slices with different tids into one track

  if remove_slice_args:
    remove_args(raw_slices) # For speedup

  #   # Write back changes to disk so that Perfetto can read it in the correct format.
  #   with open(input_trace_file, 'wb') as f:
  #     # json.dump(raw_slices, f, indent=2)
  #     # json.dump(raw_slices, f)  # speedup without indents
  #     slices_bytes = orjson.dumps(raw_slices)
  #     f.write(slices_bytes) # speedup with orjson 4x speedup
  # time.sleep(0.1) # Wait for the file to be written to disk before we try to read it.
  # slices_bytes = None

  slices_bytes = orjson.dumps(raw_slices)
  slices_bytes = io.BytesIO(slices_bytes)  # 4x speedup using orjson

  # Remove tracks that have only 1 slice because these are deemed noise
  raw_slices = h_slice.remove_tracks_with_n_slices(raw_slices, 1)

  raw_slice_index = {}
  for s in raw_slices:
    # makes very little performance difference
    key = get_key(s)
    raw_slice_index[key] = s

  return raw_slices, raw_slice_index, slices_bytes


def sanitize_trace(trace):
  for index, slice in enumerate(trace):
    for key, value in slice.items():
      # Replace None with NULL since the trace processor sometimes gives us "cat": null, but the trace viewer gives an error and can't open the trace.
      if value is None:
        slice[key] = 'NULL'
    trace[index] = slice
  return trace
