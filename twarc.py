import os
import sys

args = sys.argv[1:]

if len(args)==0:
    input_dir = 'inputs'
    output_dir = 'outputs'
else:
    input_dir = args[0]
    output_dir = args[1]
    
files = os.listdir(input_dir)

for f in files:
    filename = input_dir+'/'+f
    os.system('twarc hydrate ' + filename + ' > ' + output_dir + '/' + f + '_out.jsonl')
