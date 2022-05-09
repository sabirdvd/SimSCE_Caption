import sys
import argparse
from niacin.augment import RandAugment
from niacin.text import en


parser=argparse.ArgumentParser()
parser.add_argument('--input',  default='', help='input_text', type=str,required=True)
parser.add_argument('--output', default='', help='', type=str,required=True)
args = parser.parse_args()


augmentor = RandAugment([
    en.add_synonyms,
    en.add_hyponyms,
    #en.add_misspelling,
    en.swap_words,
    #en.add_contractions,
    #en.add_whitespace,
], n=2, m=15, shuffle=False)

text = []

with open(args.input,'rU') as f:
        for line in f:
                text.append(line.rstrip())


f=open(args.output, "w")
for i in range(len(text)):
    temp =[]
    for data in text:
        for tx in augmentor:
            data = tx(data)
        print(data)

        temp.append(data)
        result= str(data)
        f.write(result)

        f.write('\n')
f.close()
