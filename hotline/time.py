import functools
import operator

from datetime import timedelta
from datetime import datetime
from tabulate import tabulate

from hotline.hotline import *


def safe_time_convert(time):
  time = str(time)
  if time[-3:] == '000':
    # raise ValueError(f'When converting between the new nanosecond format and the old microsecond format, we found a number that doesn\'t include the expected "000". Given value: {time}')
    return int(time[:-3])
  else:
    return int(time)


def slices_time_stats(slices):
  # get ts when first slice starts and last slice finishes
  min_start_ts = min([slice["ts"] for slice in slices])
  max_stop_ts = max([slice["ts"] + slice['dur'] for slice in slices])
  ts = min_start_ts
  dur = max_stop_ts - min_start_ts
  # idle_dur = calc_idle_dur(slices, span_dur=dur)
  # dur -= idle_dur  # too dangerous because of slices at multiple levels
  return ts, dur


def add_time(op, **kwargs):
  """Add first slice start time, last slice finish time, and duration."""
  if 'resources' not in op:
    return
  log.debug(f'[add_time] Processing: {op.get("idx")}, {op["name"]}, {op["type"]}')
  for res_name, res in op['resources'].items():
    slices = h_slice.get_slices(op, filter_by_resource=res_name)
    if slices:
      ts, dur = slices_time_stats(slices)
      h_op.add_dict_to_resource(
        op, res_name, 'time',
        ts=ts, dur=dur,
      )


def calc_idle_dur(slices, span_dur=None):
  """Optionally provide the span_dur of the slices so it doesn't have to be calculated again."""
  if span_dur is None:
    _, span_dur = slices_time_stats(slices)

  slices = h_slice.get_slices_at_depth(slices, 'minimum')
  slices_dur = sum([slice['dur'] for slice in slices])
  # TODO: Check for overlapping slices on same or different tracks -> BAD

  idle_dur = span_dur - slices_dur
  return idle_dur


def calc_async_dur(slices):
  async_op_names = ['cudaMemcpyAsync', 'cudaStreamSynchronize']
  slices_of_interest = [slice for slice in slices if slice['name'] in async_op_names]
  async_dur = sum([slice['dur'] for slice in slices_of_interest]) - (len(slices_of_interest)*1000) # remove all the async time except 1 unit per slice of interest. TODO: Calculate how much time overlap there is within the slices of interest and cancel that out.
  return async_dur

count_sum_greater_than_1 = 0
count_sum_less_than_1 = 0
def add_relative_timing(op, **kwargs):
  log.debug(f'[add_relative_timing] Processing: {op.get("idx")}, {op["name"]}, {op["type"]}')
  if not 'ops' in op or 'resources' not in op:
    return
  global count_sum_greater_than_1
  global count_sum_less_than_1
  try:
    max_dur = max([res['time']['dur'] for _, res in op['resources'].items()])
  except:
    log.error('[add_relative_timing] Exception!')
    embed()
  closest_to_100_percent = 9999
  for res_name in op['resources'].keys():
    sum_ = 0
    last_sub_op_time = None
    for idx, sub_op in enumerate(op['ops']):
      if 'resources' not in sub_op or res_name not in sub_op['resources']:
        continue
      sub_op_time = sub_op['resources'][res_name]['time']
      sub_op_dur = sub_op_time['dur']
      if max_dur == 0:
        relative_dur = 0  # if max_dur is 0 aka. the parent op has 0 dur, that means all of these sub_ops must have 0 dur as well.
      else:
        relative_dur = sub_op_dur / max_dur
      sub_op_time['relative_dur'] = relative_dur

      # Calc gap
      if not last_sub_op_time:
        gap = sub_op_time['ts'] - op['resources'][res_name]['time']['ts']
      else:
        # calculate relative gap
        gap = sub_op_time['ts'] - last_sub_op_time['ts'] - last_sub_op_time['dur']
      # Reduce gap if any async slices are present
      async_dur = calc_async_dur(op['resources'][res_name]['slices'])
      gap -= async_dur

      # Calc and set relative gap
      if max_dur == 0:
        relative_gap = 0
      else:
        relative_gap = gap / max_dur
      if relative_gap < 0:
        log.debug(f'Gap between ops is less than 1! ⚠️ Is something being included twice? Perhaps fusion? fused={"fused_ids" in sub_op.keys()}')
        sub_op_time['relative_gap_to_previous'] = 0
      else:
        sub_op_time['relative_gap_to_previous'] = relative_gap
      last_sub_op_time = sub_op_time  # last op happens earlier in time

      if sub_op['name'] == 'Backward' and sub_op_time['relative_dur'] < 0.02 and 'cpu' in res_name:
        alt_cpu_res_name = [res_n for res_n in list(sub_op['resources'].keys()) if res_n != res_name and 'cpu' in res_n][-1]
        _, dur = slices_time_stats(sub_op['resources'][alt_cpu_res_name]['slices'])
        async_dur = calc_async_dur(sub_op['resources'][alt_cpu_res_name]['slices'])
        dur -= async_dur
        relative_dur = dur / max_dur
        sub_op_time['relative_dur'] = relative_dur
        sub_op_time['relative_gap_to_previous'] = 0

      # Never allow the first op in a level to have a gap so that everything has their start time adjusted to the left
      if idx == 0:
        sub_op_time['relative_gap_to_previous'] = 0

    # END LOGIC
    # BEGIN DIAGNOSTIC LOGGING
      sum_+= relative_dur
    closeness_to_100_percent = abs(sum_ - 1.0)  # should equal 0.0 if exactly 100% duration
    closest_to_100_percent = min(closest_to_100_percent, closeness_to_100_percent)
    log.debug(f'[add_relative_timing] Sum: {sum_} for res_name={res_name} op={op["name"]}')  # Should equal 1, aka. 100%
  if closest_to_100_percent > 0.02:
    log.debug(f'[add_relative_timing] Sum not 100%, off by (0 is best): {closest_to_100_percent} for op={op["name"]}')
    if sum_ > 1.0:
      count_sum_greater_than_1 += 1
    if sum_ < 1.0:
      count_sum_less_than_1 += 1


def print_fn_runtimes(fn_runtimes):
  sorted_fn_runtimes = sorted(fn_runtimes, key=lambda d: d['time'])
  times = [item['time'] for item in fn_runtimes]
  total_time = functools.reduce(operator.add, times)
  sorted_fn_runtimes.append({'name':'TOTAL', 'time': total_time})
  for item in sorted_fn_runtimes:
    item['%'] = item['time'] / total_time
  print(tabulate(sorted_fn_runtimes,
            headers={'name':'name','%':'%','time':'time'},
            tablefmt='orgtbl'))

  # Convert format
    #  FROM: {'name': 'add_op_idxs','time': timedelta(microseconds=60)}
    #  TO: {'add_op_idxs': timedelta(microseconds=60)}
  fn_runtimes_dict = {}
  for item in fn_runtimes:
    fn_runtimes_dict[item['name']] = item['time']

  # # Remove some times by name
  total_time = (total_time
    # Remove accuracy calculation
    - fn_runtimes_dict.get('test_accuracy', timedelta(0))
    - fn_runtimes_dict.get('test_accuracy_setup', timedelta(0))
    # Remove Disk I/O
    # - fn_runtimes_dict.get('load_raw_trace', timedelta(0))
    # - fn_runtimes_dict.get('load_trace_processor', timedelta(0))
    # - fn_runtimes_dict.get('write_traces', timedelta(0))
  )

  return total_time


def epoch_to_time_str(ts):
  unix_epoch = ts
  seconds = unix_epoch / 1000000000  # convert nanoseconds to seconds
  microseconds = (unix_epoch % 1000000000) / 1000  # convert remaining nanoseconds to microseconds

  time = datetime.fromtimestamp(seconds).strftime('%H:%M:%S.%f')[:-3]  # format time string
  microsecond_str = str(int(microseconds)).zfill(3)  # zero-pad microseconds

  formatted_time = f"{time}.{microsecond_str}"  # combine time and microseconds

  return formatted_time  # output: 17:19:05.931


def add_longest_parent_per_resource_type(op, resource_types, parent_op=None, **kwargs):
  if not parent_op or 'resources' not in op or 'resources' not in parent_op:
    return
  for res_type in resource_types:
    parent_has_res_type = any([res_name for res_name in parent_op["resources"].keys() if res_type in res_name])
    if not parent_has_res_type:
      continue
    res_names = [res_name for res_name, res in parent_op['resources'].items() if res_type in res_name]
    res_durs = [
      res['time']['dur'] #- calc_idle_dur(res['slices'], span_dur=res['time']['dur'])
         for res_name, res in parent_op['resources'].items()
          if res_type in res_name
    ]
    max_dur_res_idx = res_durs.index(max(res_durs))
    max_dur_res_name = res_names[max_dur_res_idx]
    for res_name in res_names:
      if res_name not in op['resources']:
        continue
      is_longest = res_name == max_dur_res_name
      op['resources'][res_name]['time']['parent_is_longest'] = is_longest



def adjust_async_time(op, **kwargs):
  if 'resources' not in op:
    return
  log.debug(f'[adjust_async_time] Processing: {op.get("idx")}, {op["name"]}, {op["type"]}')

  for res_name in op['resources'].keys():
    async_dur = calc_async_dur(op['resources'][res_name]['slices'])
    if async_dur:
      op['resources'][res_name]['time']['dur'] = op['resources'][res_name]['time']['dur'] - async_dur
      log.debug(f'[adjust_async_time] Reduced {op["name"]} by {async_dur} dur.')


def format_time_3_digits(input_str):
  """Generate a time string that always has 3 digits like "123" or "12.3" or "1.23"
  
  Examples: 
  from "123.56 s" to "123 s"
  from "123.567 s" to "123 s"
  from "12.356 s" to "12.3 s"
  from "1.2356 s" to "1.23 s"
  from "1.00 s" to "1 s"
  from "1.20 s" to "1.2 s" TODO: support removing trailing zero
  """
  try:
    time, unit = input_str.split()
    time = float(time)
    if time >= 100:
      output_str = "{:.0f}".format(time)
    elif time >= 10:
      output_str = "{:.1f}".format(time)
    else:
      output_str = "{:.2f}".format(time)
    output_str = output_str + f' {unit}'
    output_str = output_str.replace('.00 ', ' ')
    output_str = output_str.replace('.0 ', ' ')
  except (ValueError, IndexError):
    output_str = input_str
  return output_str

def format_time_as_str(time):
  """ Input: datetime.timedelta
      Output: string"""
  if isinstance(time, (float, int)):
    time = timedelta(microseconds=time)

  if time.total_seconds() < 0.001:
    minimum_unit = 'microseconds'
  elif time.total_seconds() < 1:
    minimum_unit = 'milliseconds'
  elif time.total_seconds() < 60: # 60 seconds
    minimum_unit = 'seconds'
  else:
    minimum_unit = 'minutes'

  time = humanize.precisedelta(time, minimum_unit=minimum_unit)
  time = time.replace('microseconds', 'us')
  time = time.replace('microsecond', 'us')
  time = time.replace('milliseconds', 'ms')
  time = time.replace('millisecond', 'ms')
  time = time.replace('seconds', 's')
  time = time.replace('second', 's')
  time = time.replace('minutes', 'm')
  time = time.replace('minute', 'm')
  time = time.replace(' and', ',')

  time = format_time_3_digits(time)
  return time


def calc_runtime_mean_std_err(timedeltas):
  # Remove outliers not within 50% of median
  times = np.array([time / timedelta(microseconds=1) for time in timedeltas])  # convert timedeltas to ints
  median = np.median(times)
  deviation = np.abs(times - median)
  mask = deviation < 0.5 * median
  outlier_free = times[mask]
  mean = np.mean(outlier_free)
  mean_str = format_time_as_str(mean)

  # Calculate standard error
  std_err = (np.std(outlier_free) / np.sqrt(len(outlier_free))) / mean * 100
  if std_err >= 1:
    std_err_str = f'{std_err:.0f}%'  # ex: 1%
  else:
    std_err_str = f'{std_err:.1f}%'  # ex: 0.5%

  mean_err_str = f'{mean_str} ±{std_err_str}' # ex: 1.2 ms ±0.5%
  return mean, std_err, mean_err_str


def format_iteration_runtime(runtimes):
  if not runtimes:
    return None, None, None

  if len(runtimes) == 40:
    runtimes_without_profiling = runtimes[10:20]  # the first 20 runtimes are without profiling, we take the last 10 for measurements, first 10 are warmups
    runtimes_with_profiling = runtimes[30:40]  # the last 20 runtimes are with profiling, we take the last 10 for measurements, first 10 are warmups

    without_prof_mean, without_prof_std_err, without_prof_mean_err_str = calc_runtime_mean_std_err(runtimes_without_profiling)
    with_prof_mean, with_prof_std_err, with_prof_mean_err_str = calc_runtime_mean_std_err(runtimes_with_profiling)

    # Calculate profiling overhead (difference as % with and without profiling)
    overhead = (with_prof_mean / without_prof_mean) - 1
    if overhead < 0:
      overhead_str = f'{abs(overhead):.2f}× faster'
    else:
      overhead_str = f'{abs(overhead):.2f}× slower'  # ex: 0.21× slower

    return with_prof_mean_err_str, without_prof_mean_err_str, overhead_str

  else:
    # use last runtime, it was run with profiling. Other times pytorch profiler performing warmups or waits.
    return format_time_as_str(runtimes[-1]), None, None
