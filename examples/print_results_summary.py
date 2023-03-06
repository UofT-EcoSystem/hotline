import os
import sys
import hotline

import pandas as pd

def main():
  if len(sys.argv) < 2:
    print('❌ Please provide a FILEPATH to a results_summary.csv')
    return

  filepath = f'{sys.argv[1]}'
  if os.path.isfile(filepath):
    df = pd.read_csv(filepath)
    hotline.h_print.print_interest_results_table(df)
  else:
    print(f'❌ {filepath} does not exist')

main()