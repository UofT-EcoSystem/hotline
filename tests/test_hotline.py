from absl import logging
import sys
import os
import glob
import importlib
import difflib  # could be improved by difftastic CLI tool https://github.com/Wilfred/difftastic
from IPython import embed
import hashlib

# Config
expected_results_parent_folder = 'expected_results/'  # folder containing all expected results
# override_expected_results = ['layers_backward_resnet50'] # set to a folder to limit tests to that folder
override_expected_results = [] # if set to empty list this means you want to use all input folders
ignore_files = []  # Skip comparing differences for files that have these substrings.
actual_results_parent_folder = 'actual_results/'  # folder containing all expected results
dry_run = True # don't actually call the main function of cpath


def colored(r, g, b, text):
  return f"\033[38;2;{r};{g};{b}m{text}\033[38;2;255;255;255m"

def print_red(text):
  print(colored(255, 0, 0, text))

def print_green(text):
  print(colored(0, 255, 0, text))

def print_diff(diff):
  for line in diff:
    line = line.replace('\n','')
    if line[0] == '-':
      print_red(line)
    elif line[0] == '+':
      print_green(line)
    else:
      print(line)


def all_same( items ):
    return len( set( items ) ) == 1

def compare_hash(files):
  digests = []
  for filename in files:
    hasher = hashlib.md5()
    with open(filename, 'rb') as f:
      buf = f.read()
      hasher.update(buf)
      hash = hasher.hexdigest()
      digests.append(hash)
  is_same = all_same(digests)
  return is_same


class CpathTestError(Exception):
  pass

# Get folders in
expected_results = list(glob.iglob(f'{expected_results_parent_folder}*', recursive=False))
expected_results = [folder.replace(expected_results_parent_folder, '') for folder in expected_results]
if override_expected_results:
  expected_results = override_expected_results

# Each folder is a DNN model to test
for model in expected_results:

  # calls main analysis script (this should process traces and generate outputs)
  if not dry_run:
    # import folder name as module (assumes this is the case)
    module_name = model
    module = importlib.import_module(module_name)
    module.main()

  # Gather trace file paths for this folder
  search_path = os.path.join(expected_results_parent_folder, model, '**/*.json')
  expected_files = list(glob.iglob(search_path, recursive=True))
  search_path = os.path.join(actual_results_parent_folder, model, '**/*.json')
  actual_files = list(glob.iglob(search_path, recursive=True))

  # Compare number of expected and actual trace files
  # expected_lines = [file.replace(expected_results_parent_folder, '') for file in expected_files]
  # actual_lines = [file.replace(actual_results_parent_folder, '') for file in actual_files]
  # diff = list(difflib.unified_diff(expected_lines, actual_lines, fromfile='expected', tofile='actual'))
  # if diff:
  #   print_diff(diff)
  #   msg = f'[{colored(255, 0, 0, "FAIL")}] The number of input and output files differ. Were files added or removed?'
  #   raise CpathTestError(msg)

  # Compare each trace file
  for expected_file, actual_file in zip(expected_files, actual_files):
    if any(ignore_file in actual_file for ignore_file in ignore_files):
      continue
    print(f'\nComparing file: {expected_file} \n            to: {actual_file}')
    is_same_hash = compare_hash([expected_file, actual_file])
    if not is_same_hash:
      print('FAILED hash comparison!')
      expected_lines = open(expected_file,"r").readlines()
      actual_lines = open(actual_file,"r").readlines()

      diff = list(difflib.unified_diff(expected_lines, actual_lines, fromfile='expected', tofile='actual'))
      if diff:
        print_diff(diff)
        msg = f'[{colored(255, 0, 0, "FAIL")}] This file has a difference: {actual_file}\n as compared to the expected: {expected_file}'
        raise CpathTestError(msg)

msg = f'[{colored(0, 255, 0, "INFO")}] All tests passed.'
print(msg)
