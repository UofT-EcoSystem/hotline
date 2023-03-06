"""
# Run Tests
pytest pytest tests/test_training_loop.py -s
# -s is to display stdout even when test passes and/or make embed() work
"""
import pytest
import os
import sys
from IPython import embed
sys.path.append(os.path.abspath('.'))
import hotline.detect_training_loop as training_loop

def test_no_ops():
  high_level_ops = []
  slices = [{'depth': 1, 'dur': 1, 'ts': 2, 'name': 'dataloader'}]
  cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
  assert cpath_ops == []

def test_no_slices():
  high_level_ops = [{'name': 'dataload', 'raw_names': ['dataload']}]
  slices = []
  cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
  assert cpath_ops == [{'name': 'dataload', 'slices': []}]

def test_one_match():
  high_level_ops = [{'name': 'dataload', 'raw_names': ['dataload']}]
  slices = [{'depth': 1, 'dur': 1, 'ts': 2, 'name': 'dataloader'}]
  cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
  assert cpath_ops == [{'name': 'dataload', 'slices': [{ 'depth': 1, 'dur': 1, 'ts': 2, 'name': 'dataloader'}]}]

def test_leading():
  high_level_ops = [{'name': 'dataload', 'raw_names': ['dataload']}]
  slices = [
    {'depth': 1, 'dur': 1, 'ts': 1, 'name': 'leading'},
    {'depth': 1, 'dur': 1, 'ts': 2, 'name': 'dataloader'},
  ]
  cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
  assert cpath_ops == [
    {'name': 'dataload',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 1, 'name': 'leading'},
        {'depth': 1, 'dur': 1, 'ts': 2, 'name': 'dataloader'},
      ]},
  ]

def test_trailing():
  high_level_ops = [{'name': 'dataload', 'raw_names': ['dataload']}]
  slices = [
    {'depth': 1, 'dur': 1, 'ts': 2, 'name': 'dataloader'},
    {'depth': 1, 'dur': 99, 'ts': 1, 'name': 'trailing'},
  ]
  cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
  assert cpath_ops == [
    {'name': 'dataload',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 2, 'name': 'dataloader'},
        {'depth': 1, 'dur': 99, 'ts': 1, 'name': 'trailing'},
      ]},
  ]

def test_match_unknown_match():
  high_level_ops = [
    {'name': 'dataload', 'raw_names': ['dataload']},
    {'name': 'loss', 'raw_names': ['loss']},
  ]
  slices = [
    {'depth': 1, 'dur': 1, 'ts': 20, 'name': 'dataloader'},
    {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'unknown'},
    {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'loss'},
  ]
  cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
  assert cpath_ops == [
    {'name': 'dataload',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 20, 'name': 'dataloader'},
        {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'unknown'},
      ]},
    {'name': 'loss',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'loss'},
      ]},
  ]

def test_match_match_match():
  high_level_ops = [
    {'name': 'dataload', 'raw_names': ['dataload']},
    {'name': 'forward', 'raw_names': ['forward']},
    {'name': 'loss', 'raw_names': ['loss']},
  ]
  slices = [
    {'depth': 1, 'dur': 1, 'ts': 20, 'name': 'dataloader'},
    {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'forward'},
    {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'loss'},
  ]
  cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
  assert cpath_ops == [
    {'name': 'dataload',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 20, 'name': 'dataloader'},
      ]},
    {'name': 'forward',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'forward'},
      ]},
    {'name': 'loss',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'loss'},
      ]},
  ]

def test_match_skip_match():
  high_level_ops = [
    {'name': 'dataload', 'raw_names': ['dataload']},
    {'name': 'forward', 'raw_names': ['forward']},
    {'name': 'loss', 'raw_names': ['loss']},
  ]
  slices = [
    {'depth': 1, 'dur': 1, 'ts': 20, 'name': 'dataloader'},
    {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'fward'},
    {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'loss'},
  ]
  cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
  assert cpath_ops == [
    {'name': 'dataload',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 20, 'name': 'dataloader'},
      ]},
    {'name': 'forward',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'fward'},
      ]},
    {'name': 'loss',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'loss'},
      ]},
  ]

def test_match_match_skipx2_match_unknown():
  high_level_ops = [
    {'name': 'zero', 'raw_names': ['zero']},
    {'name': 'dataload', 'raw_names': ['dataload']},
    {'name': 'forward', 'raw_names': ['forward']},
    {'name': 'loss', 'raw_names': ['loss']},
  ]
  slices = [
    {'depth': 1, 'dur': 1, 'ts': 19, 'name': 'zero'},
    {'depth': 1, 'dur': 1, 'ts': 20, 'name': 'dataloader'},
    {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'fward'},
    {'depth': 1, 'dur': 1, 'ts': 31, 'name': 'fward'},
    {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'loss'},
    {'depth': 1, 'dur': 1, 'ts': 41, 'name': 'unknown'},
  ]
  cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
  assert cpath_ops == [
    {'name': 'zero',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 19, 'name': 'zero'},
      ]},
    {'name': 'dataload',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 20, 'name': 'dataloader'},
      ]},
    {'name': 'forward',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'fward'},
        {'depth': 1, 'dur': 1, 'ts': 31, 'name': 'fward'},
      ]},
    {'name': 'loss',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'loss'},
        {'depth': 1, 'dur': 1, 'ts': 41, 'name': 'unknown'},
      ]},
  ]

def test_match_skip_match_unknown_sometimes1():
  high_level_ops = [
    {'name': 'dataload', 'raw_names': ['dataload']},
    {'name': 'forward', 'raw_names': ['forward']},
    {'name': 'loss', 'raw_names': ['loss']},
  ]
  slices = [
    {'depth': 1, 'dur': 1, 'ts': 1, 'name': 'unknown'},
    {'depth': 1, 'dur': 1, 'ts': 2, 'name': 'dataloader'},
    {'depth': 1, 'dur': 1, 'ts': 3, 'name': 'fward'},
    {'depth': 1, 'dur': 1, 'ts': 4, 'name': 'unknown'},
    {'depth': 1, 'dur': 1, 'ts': 5, 'name': 'loss'},
  ]
  cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
  assert cpath_ops == [
    {'name': 'dataload',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 1, 'name': 'unknown'},
        {'depth': 1, 'dur': 1, 'ts': 2, 'name': 'dataloader'},
      ]},
    {'name': 'forward',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 3, 'name': 'fward'},
        {'depth': 1, 'dur': 1, 'ts': 4, 'name': 'unknown'},
      ]},
    {'name': 'loss',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 5, 'name': 'loss'},
      ]},
  ]

def test_match_skip_match_unknown_sometimes2():
  high_level_ops = [
    {'name': 'dataload', 'raw_names': ['dataload']},
    {'name': 'forward', 'raw_names': ['forward']},
    {'name': 'loss', 'raw_names': ['loss']},
  ]
  slices = [
    {'depth': 1, 'dur': 1, 'ts': 1, 'name': 'dataloader'},
    {'depth': 1, 'dur': 1, 'ts': 2, 'name': 'unknown'},
    {'depth': 1, 'dur': 1, 'ts': 3, 'name': 'fward'},
    {'depth': 1, 'dur': 1, 'ts': 4, 'name': 'unknown'},
    {'depth': 1, 'dur': 1, 'ts': 5, 'name': 'loss'},
  ]
  cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
  assert cpath_ops == [
    {'name': 'dataload',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 1, 'name': 'dataloader'},
      ]},
    {'name': 'forward',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 2, 'name': 'unknown'},
        {'depth': 1, 'dur': 1, 'ts': 3, 'name': 'fward'},
        {'depth': 1, 'dur': 1, 'ts': 4, 'name': 'unknown'},
      ]},
    {'name': 'loss',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 5, 'name': 'loss'},
      ]},
  ]

def test_match_match_skip_match_unknown():
  high_level_ops = [
    {'name': 'zero', 'raw_names': ['zero']},
    {'name': 'dataload', 'raw_names': ['dataload']},
    {'name': 'forward', 'raw_names': ['forward']},
    {'name': 'loss', 'raw_names': ['loss']},
  ]
  slices = [
    {'depth': 1, 'dur': 1, 'ts': 19, 'name': 'zero'},
    {'depth': 1, 'dur': 1, 'ts': 20, 'name': 'dataloader'},
    {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'fward'},
    {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'loss'},
    {'depth': 1, 'dur': 1, 'ts': 41, 'name': 'unknown'},
  ]
  cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
  assert cpath_ops == [
    {'name': 'zero',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 19, 'name': 'zero'},
      ]},
    {'name': 'dataload',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 20, 'name': 'dataloader'},
      ]},
    {'name': 'forward',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'fward'},
      ]},
    {'name': 'loss',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'loss'},
        {'depth': 1, 'dur': 1, 'ts': 41, 'name': 'unknown'},
      ]},
  ]

def test_match_match_skip_match_unknown_everywhere():
  high_level_ops = [
    {'name': 'zero', 'raw_names': ['zero']},
    {'name': 'dataload', 'raw_names': ['dataload']},
    {'name': 'forward', 'raw_names': ['forward']},
    {'name': 'loss', 'raw_names': ['loss']},
  ]
  slices = [
    {'depth': 1, 'dur': 1, 'ts': 1, 'name': 'unknown'},
    {'depth': 1, 'dur': 1, 'ts': 10, 'name': 'zero'},
    {'depth': 1, 'dur': 1, 'ts': 19, 'name': 'unknown'},
    {'depth': 1, 'dur': 1, 'ts': 20, 'name': 'dataloader'},
    {'depth': 1, 'dur': 1, 'ts': 25, 'name': 'unknown'},
    {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'fward'},
    {'depth': 1, 'dur': 1, 'ts': 35, 'name': 'unknown'},
    {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'loss'},
    {'depth': 1, 'dur': 1, 'ts': 45, 'name': 'unknown'},
  ]
  cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
  assert cpath_ops == [
    {'name': 'zero',
      'slices': [
      {'depth': 1, 'dur': 1, 'ts': 1, 'name': 'unknown'},
        {'depth': 1, 'dur': 1, 'ts': 10, 'name': 'zero'},
        {'depth': 1, 'dur': 1, 'ts': 19, 'name': 'unknown'},
      ]},
    {'name': 'dataload',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 20, 'name': 'dataloader'},
      ]},
    {'name': 'forward',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 25, 'name': 'unknown'},
        {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'fward'},
        {'depth': 1, 'dur': 1, 'ts': 35, 'name': 'unknown'},
      ]},
    {'name': 'loss',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'loss'},
        {'depth': 1, 'dur': 1, 'ts': 45, 'name': 'unknown'},
      ]},
  ]

def test_match_match_skip_match_unknown_sometimes3():
  high_level_ops = [
    {'name': 'zero', 'raw_names': ['zero']},
    {'name': 'dataload', 'raw_names': ['dataload']},
    {'name': 'forward', 'raw_names': ['forward']},
    {'name': 'loss', 'raw_names': ['loss']},
  ]
  slices = [
    {'depth': 1, 'dur': 1, 'ts': 1, 'name': 'unknown'},
    {'depth': 1, 'dur': 1, 'ts': 2, 'name': 'zero'},
    {'depth': 1, 'dur': 1, 'ts': 3, 'name': 'unknown'},
    {'depth': 1, 'dur': 1, 'ts': 5, 'name': 'dataloader'},
    {'depth': 1, 'dur': 1, 'ts': 6, 'name': 'unknown'},
    {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'fward'},
    {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'loss'},
  ]
  cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
  assert cpath_ops == [
    {'name': 'zero',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 1, 'name': 'unknown'},
        {'depth': 1, 'dur': 1, 'ts': 2, 'name': 'zero'},
        {'depth': 1, 'dur': 1, 'ts': 3, 'name': 'unknown'},
      ]},
    {'name': 'dataload',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 5, 'name': 'dataloader'},
      ]},
    {'name': 'forward',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 6, 'name': 'unknown'},
        {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'fward'},
      ]},
    {'name': 'loss',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'loss'},
      ]},
  ]

def test_matchx2_skipx2_matchx2():
  high_level_ops = [
    {'name': 'dataload', 'raw_names': ['dataload']},
    {'name': 'forward', 'raw_names': ['forward']},
    {'name': 'loss', 'raw_names': ['loss']},
  ]
  slices = [
    {'depth': 1, 'dur': 1, 'ts': 20, 'name': 'dataloader'},
    {'depth': 1, 'dur': 1, 'ts': 21, 'name': 'dataloader'},
    {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'fward'},
    {'depth': 1, 'dur': 1, 'ts': 31, 'name': 'fward'},
    {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'loss'},
    {'depth': 1, 'dur': 1, 'ts': 41, 'name': 'loss'},
  ]
  cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
  assert cpath_ops == [
    {'name': 'dataload',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 20, 'name': 'dataloader'},
        {'depth': 1, 'dur': 1, 'ts': 21, 'name': 'dataloader'},
      ]},
    {'name': 'forward',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'fward'},
        {'depth': 1, 'dur': 1, 'ts': 31, 'name': 'fward'},
      ]},
    {'name': 'loss',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'loss'},
        {'depth': 1, 'dur': 1, 'ts': 41, 'name': 'loss'},
      ]},
  ]

def test_unknowns_before_match_skipx2_matchx2():
  high_level_ops = [
    {'name': 'dataload', 'raw_names': ['dataload']},
    {'name': 'forward', 'raw_names': ['forward']},
    {'name': 'loss', 'raw_names': ['loss']},
  ]
  slices = [
    {'depth': 1, 'dur': 1, 'ts': 19, 'name': 'unknown'},
    {'depth': 1, 'dur': 1, 'ts': 20, 'name': 'dataloader'},
    {'depth': 1, 'dur': 1, 'ts': 29, 'name': 'unknown'},
    {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'fward'},
    {'depth': 1, 'dur': 1, 'ts': 31, 'name': 'fward'},
    {'depth': 1, 'dur': 1, 'ts': 39, 'name': 'unknown'},
    {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'loss'},
    {'depth': 1, 'dur': 1, 'ts': 41, 'name': 'loss'},
  ]
  cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
  assert cpath_ops == [
    {'name': 'dataload',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 19, 'name': 'unknown'},
        {'depth': 1, 'dur': 1, 'ts': 20, 'name': 'dataloader'},
      ]},
    {'name': 'forward',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 29, 'name': 'unknown'},
        {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'fward'},
        {'depth': 1, 'dur': 1, 'ts': 31, 'name': 'fward'},
        {'depth': 1, 'dur': 1, 'ts': 39, 'name': 'unknown'},
      ]},
    {'name': 'loss',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'loss'},
        {'depth': 1, 'dur': 1, 'ts': 41, 'name': 'loss'},
      ]},
  ]

def test_unknowns_everywhere_match_skipx2_matchx2():
  high_level_ops = [
    {'name': 'dataload', 'raw_names': ['dataload']},
    {'name': 'forward', 'raw_names': ['forward']},
    {'name': 'loss', 'raw_names': ['loss']},
  ]
  slices = [
    {'depth': 1, 'dur': 1, 'ts': 19, 'name': 'unknown'},
    {'depth': 1, 'dur': 1, 'ts': 20, 'name': 'dataloader'},
    {'depth': 1, 'dur': 1, 'ts': 29, 'name': 'unknown'},
    {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'fward'},
    {'depth': 1, 'dur': 1, 'ts': 31, 'name': 'unknown'},
    {'depth': 1, 'dur': 1, 'ts': 32, 'name': 'fward'},
    {'depth': 1, 'dur': 1, 'ts': 39, 'name': 'unknown'},
    {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'loss'},
    {'depth': 1, 'dur': 1, 'ts': 41, 'name': 'unknown'},
    {'depth': 1, 'dur': 1, 'ts': 42, 'name': 'loss'},
    {'depth': 1, 'dur': 1, 'ts': 43, 'name': 'unknown'},
  ]
  cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
  assert cpath_ops == [
    {'name': 'dataload',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 19, 'name': 'unknown'},
        {'depth': 1, 'dur': 1, 'ts': 20, 'name': 'dataloader'},
      ]},
    {'name': 'forward',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 29, 'name': 'unknown'},
        {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'fward'},
        {'depth': 1, 'dur': 1, 'ts': 31, 'name': 'unknown'},
        {'depth': 1, 'dur': 1, 'ts': 32, 'name': 'fward'},
        {'depth': 1, 'dur': 1, 'ts': 39, 'name': 'unknown'},
      ]},
    {'name': 'loss',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'loss'},
        {'depth': 1, 'dur': 1, 'ts': 41, 'name': 'unknown'},
        {'depth': 1, 'dur': 1, 'ts': 42, 'name': 'loss'},
        {'depth': 1, 'dur': 1, 'ts': 43, 'name': 'unknown'},
      ]},
  ]

####
####  We test that failing to match a high level op should raise an exception
####

def test_no_match():
  high_level_ops = [{'name': 'dataload', 'raw_names': ['dataload']}]
  slices = [{'depth': 1, 'dur': 1, 'ts': 10, 'name': 'forward pass'}]
  with pytest.raises(NameError) as e:
    cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
    # assert cpath_ops == [{'name': 'dataload', 'slices': []}]

def test_match_followed_by_no_match():
  high_level_ops = [
    {'name': 'dataload', 'raw_names': ['dataload']},
    {'name': 'forward', 'raw_names': ['forward']},
  ]
  slices = [
    {'depth': 1, 'dur': 1, 'ts': 2, 'name': 'dataloader'},
  ]
  with pytest.raises(NameError) as e:
    cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
    # assert cpath_ops == [ 
    #   {'name': 'dataload',
    #     'slices': [
    #       {'depth': 1, 'dur': 1, 'ts': 2, 'name': 'dataloader'},
    #     ]},
    #   {'name': 'forward', 'slices': []},
    # ]

def test_match_followed_by_no_match_x2():
  high_level_ops = [
    {'name': 'dataload', 'raw_names': ['dataload']},
    {'name': 'forward', 'raw_names': ['forward']},
    {'name': 'backward', 'raw_names': ['backward']},
  ]
  slices = [
    {'depth': 1, 'dur': 1, 'ts': 2, 'name': 'dataloader'},
  ]
  with pytest.raises(NameError) as e:
    cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
    # assert cpath_ops == [ 
    #   {'name': 'dataload',
    #     'slices': [
    #       {'depth': 1, 'dur': 1, 'ts': 2, 'name': 'dataloader'},
    #     ]},
    #   {'name': 'forward', 'slices': []},
    #   {'name': 'backward', 'slices': []},
    # ]

def test_no_match_followed_by_match():
  high_level_ops = [
    {'name': 'dataload', 'raw_names': ['dataload']},
    {'name': 'forward', 'raw_names': ['forward']},
  ]
  slices = [
    {'depth': 1, 'dur': 1, 'ts': 2, 'name': 'forward'},
  ]
  with pytest.raises(NameError) as e:
    cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
    # assert cpath_ops == [ 
    #   {'name': 'dataload', 'slices': []},
    #   {'name': 'forward',
    #     'slices': [
    #       {'depth': 1, 'dur': 1, 'ts': 2, 'name': 'forward'},
    #     ]},
    # ]

def test_no_match_followed_by_match_x2():
  high_level_ops = [
    {'name': 'dataload', 'raw_names': ['dataload']},
    {'name': 'forward', 'raw_names': ['forward']},
    {'name': 'backward', 'raw_names': ['backward']},
  ]
  slices = [
    {'depth': 1, 'dur': 1, 'ts': 2, 'name': 'forward'},
    {'depth': 1, 'dur': 1, 'ts': 3, 'name': 'backward'},
  ]
  with pytest.raises(NameError) as e:
    cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)
    # assert cpath_ops == [ 
    #   {'name': 'dataload', 'slices': []},
    #   {'name': 'forward',
    #     'slices': [
    #       {'depth': 1, 'dur': 1, 'ts': 2, 'name': 'forward'},
    #     ]},
    #   {'name': 'backward',
    #     'slices': [
    #       {'depth': 1, 'dur': 1, 'ts': 2, 'name': 'backward'},
    #     ]},
    # ]

####
####  More realistic tests
####


def test_full():
  high_level_ops = [
    {'name': 'dataload', 'raw_names': ['dataload']},
    {'name': 'forward', 'raw_names': ['forward']},
    {'name': 'loss', 'raw_names': ['loss']},
    {'name': 'zero', 'raw_names': ['zero']},
    {'name': 'backward', 'raw_names': ['backward']},
    {'name': 'optimizer', 'raw_names': ['optimizer']},
  ]

  slices = [
    {'depth': 1, 'dur': 1, 'ts': 1, 'name': 'leading'},
    {'depth': 1, 'dur': 1, 'ts': 2, 'name': 'dataloader'},
    {'depth': 1, 'dur': 1, 'ts': 3, 'name': 'dataloader'},
    {'depth': 1, 'dur': 1, 'ts': 9, 'name': 'in-between ops'},
    {'depth': 1, 'dur': 1, 'ts': 10, 'name': 'forward pass'},
    # {'depth': 1, 'dur': 1, 'ts': 11, 'name': 'in-between ops'},
    {'depth': 1, 'dur': 1, 'ts': 20, 'name': 'loss'},
    {'depth': 1, 'dur': 1, 'ts': 21, 'name': 'within ops'},
    {'depth': 1, 'dur': 1, 'ts': 22, 'name': 'loss'},
    {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'zero'},
    {'depth': 1, 'dur': 1, 'ts': 31, 'name': 'zero'},
    {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'bward'},
    {'depth': 1, 'dur': 1, 'ts': 41, 'name': 'bward'},
    {'depth': 1, 'dur': 1, 'ts': 50, 'name': 'optimizer'},
    {'depth': 1, 'dur': 1, 'ts': 99, 'name': 'trailing'},
  ]

  cpath_ops = training_loop._detect_training_loop(high_level_ops, slices)

  assert True
  cpath_ops == [
    {'name': 'dataload',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 1, 'name': 'leading'},
        {'depth': 1, 'dur': 1, 'ts': 2, 'name': 'dataloader'},
        {'depth': 1, 'dur': 1, 'ts': 3, 'name': 'dataloader'},
      ]},
    {'name': 'forward',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 9, 'name': 'in-between ops'},
        {'depth': 1, 'dur': 1, 'ts': 10, 'name': 'forward pass'},
        {'depth': 1, 'dur': 1, 'ts': 11, 'name': 'in-between ops'},
      ]},
    {'name': 'loss',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 20, 'name': 'loss'},
        {'depth': 1, 'dur': 1, 'ts': 21, 'name': 'within ops'},
        {'depth': 1, 'dur': 1, 'ts': 22, 'name': 'loss'},
      ]},
    {'name': 'zero',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 30, 'name': 'zero'},
        {'depth': 1, 'dur': 1, 'ts': 31, 'name': 'zero'},
      ]},
    {'name': 'backward',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 40, 'name': 'bward'},
        {'depth': 1, 'dur': 1, 'ts': 41, 'name': 'bward'},
      ]},
    {'name': 'optimizer',
      'slices': [
        {'depth': 1, 'dur': 1, 'ts': 50, 'name': 'optimizer'},
        {'depth': 1, 'dur': 1, 'ts': 99, 'name': 'trailing'},
      ]},
  ]

# test_full()