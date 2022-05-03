# Unspervied-SimSCE Caption via BLIP nucleus sampling


In this project, we are trying   to learn semantic similarity in an unsupervised manner via a synthetic caption dataset (e.g., nucleus sampling ([Holtzman et al 2020](https://arxiv.org/abs/1904.09751))). The advantage of nucleus sampling is that, unlike traditional beam serach that generates safe captions, it generates more surprising diverse generate caption. The reserach question is: can we learn a good sentence represention from 3 million synthetic caption? 

(1) If the answer is yes, then we could combined this with wiki data to learn general semantic simairlty task like UNIMO [(Li et al 2021)](https://medium.com/@iee_53136/paper-summary-unimo-towards-unified-modal-understanding-and-generation-via-cross-modal-8c9e881c2012) and MCSE [(Zhang et al. 2022)](https://arxiv.org/pdf/2204.10931.pdf).





Note that the problem with [Conceptinal Caption](https://ai.google.com/research/ConceptualCaptions/) is the noisy alignment between text and images. 

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
Please refer to my [blog](https://github.com/sabirdvd/BLIP_image_caption_demo) about nucleus sampling and the original [BLIP paper](https://arxiv.org/pdf/2201.12086.pdf)  


Download 3.3 million images from [Conceptinal Caption](https://ai.google.com/research/ConceptualCaptions/) 300G
Not that, the  download is slow, so it will take a couple of days. 

## Training 

Dowload SimSCE 

```
git clone https://github.com/princeton-nlp/SimCSE.git
pip install -r /content/SimCSE/requirements.txt
```


```
cd SimSCE
```

Download the generated Caption dataset from Conceptinal Caption (CC) with light filtering (e.g., without caption that less than 40 character) 
the result is 2M (2255927 captions) and without any filter (2864924)

```
!wget clone https://www.dropbox.com/s/pc1uv2rf6nqdp57/CC_caption_40.txt.zip
```
For CC+wiki 3M (3255928) 

```
!wget clone https://www.dropbox.com/s/1whxunhaze7hkk2/CC_caption_40%2Bwiki.txt.zip
```

For CC+wiki+COCO-Caption  3.5M (366984)

```
!wget clone https://www.dropbox.com/s/k7oqwr9a1a0h8x1/CC_caption_40%2Bwiki%2BCOCO.txt.zip
```

For COCO-caption (413915)  (human labled) 

```
!wget clone https://www.dropbox.com/s/6gfu2esshvnj4sm/caption_only.txt.zip
```
For COCO-caption +wiki 1.4M (1413915)

```
!wget clone  https://www.dropbox.com/s/wc4k677wp24kzhh/COCO%2Bwiki.txt.zip
```

Download Eva tools 

```
cd SentEval/data/downstream/
bash download_dataset.sh
``` 

Download bert-base-uncased huggingface version TF->pytorch
```
!git clone https://huggingface.co/bert-base-uncased
```



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

## Preliminary results Acc of stsb_spearman  (work in progress)

Our reult better that SoTa that trained on only caption 


| Model | stsb |
| ------------- | ------------- |
| SimCSE-BERT COCO-caption  | 67.8  |
| MCSE-BERT COCO-caption [(Zhang et al. 2022)](https://arxiv.org/pdf/2204.10931.pdf)  | 71.6 |
|this work COCO-caption | 71.7 |
|this work CC | **73.8** |

However, when combining the against textual visual + wik, the visual feature + wiki have better result (Zhang et al. 2022)


| Model | stsb |
| ------------- | ------------- |
| SimCSE-BERT wiki+COCO-caption  | 73.9 |
| MCSE-BERT wiki+COCO-caption  [(Zhang et al. 2022)](https://arxiv.org/pdf/2204.10931.pdf)  | **78.5** |
|this work wiki+COCO-caption | 74.0 |
| this work+COCO+CC| 74.5      |









## Reference


