# QAEngineering

### set up docker
```
docker build Docker --build-arg user=root --build-arg password=makefog -t aia:std-ssh
docker run -it --rm --runtime=nvidia -p 6022:22 -p 6006:6006 -p 8888:8888 -d aia:std-ssh
```

In project root directory:
```
sh steup.sh
``` 

### start training:

```
python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=1 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=80 \
  --doc_stride=40 \
  --output_dir=/tmp/squad_base/
```