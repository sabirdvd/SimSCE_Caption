


import openai
import sys
import argparse
from api_secrets import API_KEY


openai.api.key = API_KEY


parser=argparse.ArgumentParser()
parser.add_argument('--input',  default='', help='input_prompt', type=str,required=True)
parser.add_argument('--output', default='', help='', type=str,required=True)
args = parser.parse_args()



with open(args.input,'rU') as f:
        for line in f:
                file1.append(line.rstrip())



f=open(args.output, "w")
for i in range(len(file1)):
	temp =[]
	responces = openai.Completion.create(engine="text-davinci-001", prompt=file1[i], max_tokens=35)

	temp.append(responces)
	result= +str(w)
	f.write(result)
	f.write('\n')
	print w
f.close()
