# unspervied-SimSCE Caption via BLIP nucleus sampling

In this project, we are trying   to learn semantic similarity in an unsupervised manner via a synthetic caption dataset (e.g., nucleus sampling). The advantage of nucleus sampling is that, unlike traditional beam serach that generates safe captions, it generates more surprising diverse generate caption. The reserach question is:  can we learn a good sentence represention from 3 million synthetic caption? 

(1) If the answer is yes, then we could combined this we reqular data to learn general semantic simairlty task .

Note that the problem with [Conceptinal Caption](https://ai.google.com/research/ConceptualCaptions/) is the noisy alignment between text and images. 

Dowload SimSCE 

```
git clone https://github.com/princeton-nlp/SimCSE.git
pip install -r /content/SimCSE/requirements.txt
```

Download BLIP with nucleus sampling

```
git clone https://github.com/salesforce/BLIP
pip install -r requirements.txt

git clone https://github.com/sabirdvd/BLIP_image_caption_demo.git
cp caption_Inference_L.py ../
python caption_Inference_L.py

```

Download 3.3 million images [Conceptinal Caption](https://ai.google.com/research/ConceptualCaptions/) 300G


## Train 

```
cd SimSCE
```

Download extracted Caption with less than 40 cha filtering 

```
!wget clone https://www.dropbox.com/s/pc1uv2rf6nqdp57/CC_caption_40.txt.zip
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
python train.py \                                                                                                                      ok  SimSCE py  
    --model_name_or_path bert-base-uncased \
    --train_file CC_caption_40.txt \
    --output_dir result/my-unsup-simcse-bert-base-uncased \
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
