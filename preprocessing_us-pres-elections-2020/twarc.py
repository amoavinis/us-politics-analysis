import os
import sys
from tqdm import tqdm
args = sys.argv[1:]

if len(args)==0:
    input_dir = 'inputs'
    output_dir = 'outputs'
else:
    input_dir = args[0]
    output_dir = args[1]

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
files = os.listdir(input_dir)

for f in tqdm(files):
    filename = input_dir+'/'+f
    os.system('twarc hydrate ' + filename + ' > ' + output_dir + '/' + f + '_out.jsonl')
