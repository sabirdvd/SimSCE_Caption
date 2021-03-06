# Unspervied-SimSCE via BLIP nucleus sampling


Please refer to this medium blog post for more detail.

```
Please note that you need to export OPENAI_API_KEY="YOUR_OPENAI_KEY before running GPT-3 scripts.
```

<!---Supervised learning model fine-tuning is like a kinda cheating and doesn't generalize well specially on the COCO dataset -->

Supervised learning model fine-tuning is like a kinda cheating and doesn't generalize well specially on the semantic similairty dataset. In this project, we are trying to learn semantic similarity in an unsupervised manner via a synthetic caption dataset (e.g., nucleus sampling ([Holtzman et al 2020](https://arxiv.org/abs/1904.09751))). The advantage of nucleus sampling is that, unlike traditional beam serach that generates safe captions, it generates more surprising diverse generate caption. The reserach question is: can we learn a good sentence represention from 3 million synthetic caption? 

(1) If the answer is yes, then we could combined this with wiki data to learn general semantic simairlty task like UNIMO [(Li et al 2021)](https://medium.com/@iee_53136/paper-summary-unimo-towards-unified-modal-understanding-and-generation-via-cross-modal-8c9e881c2012) and MCSE [(Zhang et al. 2022)](https://arxiv.org/pdf/2204.10931.pdf).



```
python 3.6
torch 1.7.1 with cuda 10.1
```



Note that the problem with [Conceptinal Caption](https://ai.google.com/research/ConceptualCaptions/) is the noisy alignment between text and images. 


## Dataset  

Download 3.3 million images from [Conceptinal Caption](https://ai.google.com/research/ConceptualCaptions/) size around 300G+ using this code [igorbrigadir](https://github.com/igorbrigadir/DownloadConceptualCaptions)

Place data from: https://ai.google.com/research/ConceptualCaptions/download in this folder

```Train_GCC-training.tsv``` Training Split (3,318,333)

```Validation_GCC-1.1.0-Validation.tsv```  Validation Split (15,840)



And run
```
download_data.py
```

Note that, the download is slow, so it will take a couple of days.


## Caption generation 

Download BLIP with nucleus sampling 

```
git clone https://github.com/salesforce/BLIP
pip install -r requirements.txt

git clone https://github.com/sabirdvd/BLIP_image_caption_demo.git
cp caption_Inference_L.py ../

# for beam search 
python caption_Inference_L.py

cp caption_Infer_NS_L.py /BLIP 
#nucleus sampling 
python caption_Infer_NS_L.py

```
<!---
Please refer to my [blog](https://github.com/sabirdvd/BLIP_image_caption_demo) about nucleus sampling and the original [BLIP paper](https://arxiv.org/pdf/2201.12086.pdf)  
-->

## Text generation 
Another idea is to use a language model (i.e., GPT2/3 to increase the data) via generation or prompting.

Generation Example GPT-2 huggingface, the result is nosiy (unhuman like)

```
python GPT-2_gen.py
```

```
Input: a man hitting a tennis ball

a man hitting a tennis ball. The ball broke. The man said he hit it because of bad luck. The man who hit it got his ball
```

GPT-3 davinci-001  (max_length=35)

```
python GPT-3_prompt.py 
```

Caption prompt   ** prompt ** 

```
**Write more about this caption:**

a man hitting a tennis ball

This caption is about a man hitting a tennis ball. The man is hitting the tennis ball against a wall. He is doing this to improve his tennis skills. 


**a man hitting a tennis ball**

A tennis ball is a spherical object that is used in the sport of tennis. It is hit by a player's hand and then hits a racket before being sent over the net.

A tennis ball is a ball that is used in the sport of tennis. It is round and has a diameter of about 68.5 mm (2.7 in). Tennis balls are usually made of a synthetic rubber, but can also be made of natural rubber.

**Tell me more about this text:**

a couple of people riding horses across a sandy beach

This text describes a scene of a couple of people riding horses across a sandy beach. The horses are galloping and the riders are enjoying the view. This image is peaceful and calming, and it is easy to imagine being a part of it.

**Tell more about this sentence:**

a man hitting a tennis ball 

A man hitting a tennis ball is playing tennis.
```

gpt-j-6B  (max_length=35)


```
**Write more about this caption:**

Write more about this caption: a man hitting a tennis ball with some friends at the end of a long hike, late afternoon in the mountains in New Mexico, USA.

Input: a man hitting a tennis ball

a man hitting a tennis ball into the air, and then his return shots.

``` 

This concluded that the bigger the model the more robust against context. 

To run this locally you need GPU and 16GRAM+ (recomanded 32GRAM) , please have look at this [hugging face post](https://huggingface.co/blog/how-to-generate)  for diffrent generation method (e.g, beam serach)


## Text data augmentation

A fast and cheap way to increase the data is using [data augmentation strategies](https://niacin.readthedocs.io/en/latest/) 

```python 

from niacin.augment import RandAugment
from niacin.text import en

augmentor = RandAugment([
    en.add_synonyms,
    en.add_hyponyms,
    en.add_misspelling,
    en.swap_words,
    en.add_contractions,
    en.add_whitespace,
], n=2, m=15, shuffle=False)


```
input text 

```python 
text = ["a man hitting a tennis ball"]
    
for data in text:
    for tx in augmentor:
        data = tx(data)
    print(data)

output 

a old boy hitting a tennis ball


```

Note, that all the techniques to increase the data mentioned above will be applied over a human-written caption (COCO-caption). 


## Training 

Dowload SimSCE 

```
git clone https://github.com/princeton-nlp/SimCSE.git
pip install -r /content/SimCSE/requirements.txt
```

replaced the original ```trainers.py``` with the provided one 



Download the generated Caption dataset from Conceptinal Caption (CC) with light filtering (e.g., without caption that less than 40 character) 
the result is 2M (2255927 captions) and without any filter (2864924)

```
wget clone https://www.dropbox.com/s/pc1uv2rf6nqdp57/CC_caption_40.txt.zip
```
For CC+wiki 3M (3255928) 

```
wget clone https://www.dropbox.com/s/1whxunhaze7hkk2/CC_caption_40%2Bwiki.txt.zip
```

For CC+wiki+COCO-Caption  3.5M (366984)

```
wget clone https://www.dropbox.com/s/k7oqwr9a1a0h8x1/CC_caption_40%2Bwiki%2BCOCO.txt.zip
```

For COCO-caption (413915)  (human labled) 

```
wget clone https://www.dropbox.com/s/6gfu2esshvnj4sm/caption_only.txt.zip
```
For COCO-caption +wiki 1.4M (1413915)

```
wget clone  https://www.dropbox.com/s/wc4k677wp24kzhh/COCO%2Bwiki.txt.zip
```



Image caption corpus does not train a good language model 
[(Tan and Bansal, 2020)](https://aclanthology.org/2020.emnlp-main.162.pdf). Therefore we add 7M sentence from wiki to the caption dataset 

```COCO-caption+wiki+CC+8Mwiki 11M (11541667)``` 

```
wget clone https://www.dropbox.com/s/xhfx32sjy2z5bpa/11M_wiki_7M%2BCC%2BCOCO.txt.zip
```



Download Eva tools 

```
cd SentEval/data/downstream/
bash download_dataset.sh
``` 



Download [bert-base-uncased huggingface](https://huggingface.co/bert-base-uncased/tree/main) version TF->pytorch
```
git clone https://huggingface.co/bert-base-uncased
```
Download bert-base-uncased huggingface version TF->pytorch

```
git clone https://huggingface.co/roberta-base
```

run  

```
python train.py \                                                                                                                     
    --model_name_or_path bert-base-uncased \
    --train_file CC_caption_40.txt \
    --output_dir result/my-unsup-simcse-caption\
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --learning_rate 3e-5 \
    --max_seq_length 10 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    "$@"
```
<!---
 [colab](https://colab.research.google.com/drive/14BNaxj1yrNolAoy9uuABiFrwwTzniHCx?usp=sharing) demo here
-->

Pre-trained model 

```
wget clone https://www.dropbox.com/s/dezln4bzjguxbn3/my-unsup-simcse-caption.zip
```




## Inference 

Follwing the SimSCE  

```python 
In [2]: tokenizer = AutoTokenizer.from_pretrained("pre-trained")

In [3]: model = AutoModel.from_pretrained("pre-trained")

In [4]: texts = [
   ...:     "There's a kid on a skateboard.",
   ...:     "A kid is skateboarding.",
   ...:     "A kid is inside the house."
   ...: ]
   ...: inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
   ...:
   ...: # Get the embeddings
   ...: with torch.no_grad():
   ...:     embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
   ...:
   ...: # Calculate cosine similarities
   ...: # Cosine similarities are in [-1, 1]. Higher means more similar
   ...: cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
   ...: cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])
   ...:
   ...: print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[1], cosine_sim_0_1))
   ...: print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[2], cosine_sim_0_2))
Cosine similarity between "There's a kid on a skateboard." and "A kid is skateboarding." is: 0.815
Cosine similarity between "There's a kid on a skateboard." and "A kid is inside the house." is: 0.553

```




## Preliminary results --  stsb_spearman  (work in progress)

wiki only 

| Model | stsb |
| ------------- | ------------- |
| Baseline: SimCSE-BERT  | **76.9**   |
|this work  | 71.4 |


Our result is better than SOTA which trained only on caption dataset. 

| Model | stsb |
| ------------- | ------------- |
| Baseline: SimCSE-BERT COCO-caption  | 67.8  |
| MCSE-BERT COCO-caption [(Zhang et al. 2022)](https://arxiv.org/pdf/2204.10931.pdf)  | 71.6 |
|this work COCO-caption | 71.7 |
|this work CC | **73.8** |

However, when combining the against textual visual + wiki, the visual feature + wiki have better result (Zhang et al. 2022)


| Model | stsb |
| ------------- | ------------- |
| Baseline: SimCSE-BERT wiki+COCO-caption  | 73.9 |
| MCSE-BERT wiki+COCO-caption  [(Zhang et al. 2022)](https://arxiv.org/pdf/2204.10931.pdf)  | **78.5** |
|this work wiki+COCO-caption | 74.0 |
| this work+COCO+CC| 74.5      |









## Reference


