import traceback

from perfetto.trace_processor import TraceProcessor

from hotline.hotline import *

# Example from slice table
# {
#   'id': 459,
#   'type': 'internal_slice',
#   'ts': 1665165714853849000,
#   'dur': 15000,
#   'track_id': 2,
#   'category': 'cpu_op',
#   'name': 'aten::add_',
#   'depth': 2,
#   'stack_id': 1711056705702287,
#   'parent_stack_id': 1794751843672294,
#   'parent_id': 389,
#   'arg_set_id': 455,
#   'thread_ts': None,
#   'thread_dur': None,
#   'thread_instruction_count': None,
#   'thread_instruction_delta': None,
#   'cat': 'cpu_op',
#   'slice_id': 459
# }

interesting_fields = 'SELECT ts, dur, track_id, category, name, depth, cat, slice_id, id FROM slice'


def load_trace_processor(trace_filepath=None, trace_bytes=None):
  input_trace = trace_bytes or trace_filepath
  try:
    tp = TraceProcessor(input_trace)
  except ConnectionResetError as e:
    # This happens sometimes so retry once
    tp = TraceProcessor(input_trace)

  def query_dict(query):
    query = query.lower().replace('select * from slice', interesting_fields) # gives a 15% speedup
    try:
      query_iterator = tp.query(query)
    except Exception as e:
      print('[ERROR] Unable to run query: %s' % query)
      # print('[ERROR] Unable to run query.')
      print(traceback.format_exc())
      print('\n\n')
      raise e
    return [item.__dict__ for item in query_iterator]

  tp.query_dict = query_dict

  return tp


def get_profile_step_slice(tp):
  """Scope to a certain ProfilerStep"""
  query = f'SELECT * FROM slices WHERE name LIKE "ProfilerStep%";'  # PyTorch specific
  slice = tp.query_dict(query)[0]
  start = slice['ts']
  end = slice['ts'] + slice['dur']
  between_time_range = f' ts BETWEEN {start} AND {end}'

  return slice, between_time_range


def sql_in_string(ids, col_name):
  ids_str = '(' + ', '.join([str(id) for id in ids]) + ')'  # make a string like '(1, 2, 3)'
  in_ids_str = f' {col_name} IN {ids_str}'  # example: ' track_id IN (2, 4)'
  return in_ids_str


def track_ids_for_process_of_slice(tp, slice):
  """Get track ids for the process that a slice (ex. ProfilerStep) is found on. Said another way, get all the track ids contained within a process given a track id. For example, what timelines exist in the ProfilerStep process.
  """
  neighbouring_track_ids  = f'''
  select id from THREAD_TRACK where utid in (
    select utid from THREAD where upid=(
      SELECT upid from thread WHERE utid=(
        select utid from thread_track WHERE id={slice["track_id"]}
      )
    )
  );
  '''
  tracks = tp.query_dict(neighbouring_track_ids)
  primary_track_ids = [track['id'] for track in tracks]
  on_primary_tracks = sql_in_string(primary_track_ids, 'track_id')
  return primary_track_ids, on_primary_tracks


def get_process_and_thread_name_for_each_track(tp):
  """Get process and thread name for each track.
  Note docs: https://perfetto.dev/docs/analysis/trace-processor#thread-and-process-identifiers
  """
  query = f'''
  SELECT
    thread_track.id as track_id,
    process.pid, thread.tid,
    process.name AS process_name,
    thread.name AS thread_name,
    process.type AS process_type,
    thread.type AS thread_type,
    thread.utid,
    process.upid
  FROM thread_track
  JOIN thread on thread_track.utid = thread.utid
  JOIN process on thread.upid = process.upid;
  '''
  track_info = tp.query_dict(query)
  return track_info


def add_gpu(op, tp, slice_index, flow_index, **kwargs):
  """Based on CPU threads, find GPU kernels.

  TODO: Several assumptions are made that need fixing:
    - Assumes flows connect from cpu to gpu (what if flow is in reverse or between cpus?)
    - Assumes cpu/gpu resource names (cpu1 -> gpu1) (what if cpu1 -> gpu[1,2,3,4]? what if cpu1 -> cpu2?)
  """
  if 'resources' not in op:
    return
  log.debug(f'[add_gpu] Processing: {op.get("idx")}, {op["name"]}, {op["type"]}')
  res_names = list(op['resources'].keys())
  for res_name in res_names:
    slices = h_slice.get_slices(op, filter_by_resource=res_name)
    if not slices:
      continue

    # Get kernel slice ids
    kernel_slice_ids = []
    slice_ids = [op["id"] for op in slices]
    for slice_id in slice_ids:
      kernel_slice_ids.extend(flow_index.get(slice_id, []))

    # Get kernel slices
    for kernel_slice_id in kernel_slice_ids:
      kernel_slice = slice_index[kernel_slice_id]
      track_id = kernel_slice['track_id']
      h_op.append_list_to_resource(op, 'gpu' + str(track_id), 'slices', kernel_slice)



def add_slices(op, slices, tp):
  """ Queries each slice given for nested slices then adds all them to an op.

  TODO: improve function name and/or break this into two functions?
  """
  # Get descendant slices
  d_slices = []
  for slice in slices:
    try:
      d_slice = tp.query_dict(f'SELECT * FROM descendant_slice({slice["id"]});')
      d_slices.extend(d_slice)
    except:
      print(traceback.format_exc())

  sub_slices = sorted(slices + d_slices, key=lambda d: d['ts'])
  for slice in sub_slices:
    h_op.append_list_to_resource(op, 'cpu' + str(slice['track_id']), 'slices', slice)


def create_slice_index(tp):
  slices = tp.query_dict('SELECT * FROM slice')
  slice_index = {}
  for slice in slices:
    slice_index[slice['id']] = slice
  return slice_index

def create_flow_index(tp):
  # From CPU (key) to GPU (value)
  flows = tp.query_dict('SELECT slice_in, slice_out FROM flow')
  flow_index = {}
  for flow in flows:
    if flow['slice_out'] not in flow_index:
      flow_index[flow['slice_out']] = [ flow['slice_in'] ]
    else:
      flow_index[flow['slice_out']].append(flow['slice_in'])
  return flow_index

def create_track_indexes(tp):
  track_info = get_process_and_thread_name_for_each_track(tp)
  track_index = {}
  for track in track_info:
    track_index[track['track_id']] = track

  thread_index = {}
  for track in track_info:
    thread_index[track['tid']] = track

  process_index = {}
  for track in track_info:
    if track['pid'] not in process_index:
      process_index[track['pid']] = [track]
    else:
      process_index[track['pid']].append(track)

  return track_index, thread_index, process_index


def get_connected_slices(tp, flow_index, slices):
    # Get connected slice ids
  connected_slice_ids = []
  sub_slice_ids = [op["id"] for op in slices]
  for sub_slice_id in sub_slice_ids:
    connected_slice_ids.extend(flow_index.get(sub_slice_id, []))

  # Get connected slices from flow index
  ids_str = ','.join([str(id) for id in connected_slice_ids])  # Example string: 23448, 27178, 27205
  connected_slices = tp.query_dict(f'SELECT * FROM slices WHERE id IN ({ids_str})')
  return connected_slices

def get_descendant_and_connected_slices(tp, flow_index, slice_id):
  sub_slices = tp.query_dict(f'SELECT * FROM descendant_slice({slice_id})')
  connected_slices = get_connected_slices(tp, flow_index, sub_slices)

  slices = []
  slices.extend(sub_slices)
  slices.extend(connected_slices)
  return slices

def get_slices_on_other_threads_on_same_process_between_time_range(tp, slice):
  """Get slices on other threads on same process between time range."""
  # Get other thread ids on same process
  track_ids, on_tracks = track_ids_for_process_of_slice(tp, slice)
  track_ids.remove(slice['track_id'])
  on_tracks = sql_in_string(track_ids, 'track_id')

  # Get time range
  start = slice['ts']
  end = slice['ts'] + slice['dur']
  between_time_range = f' ts BETWEEN {start} AND {end}'

  # Get slices
  slices = tp.query_dict(f'SELECT * FROM slices WHERE {on_tracks} and {between_time_range}')
  return slices
