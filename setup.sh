
apt install git

export REFERENCE_DIR=reference

mkdir -p $REFERENCE_DIR

cd $REFERENCE_DIR && git clone https://github.com/google-research/bert.git

mkdir -p model
curl -o model/multilingual_L-12_H-768_A-12.zip https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip
unzip model/multilingual_L-12_H-768_A-12.zip -d model/bert_base

export SQUAD1_PATH=data/raw/squad_v1_1
mkdir -p $SQUAD1_PATH
curl -o $SQUAD1_PATH/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
curl -o $SQUAD1_PATH/dev-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
curl -o $SQUAD1_PATH/evaluate-v1.1.py https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py

export BERT_BASE_DIR=$REFERENCE_DIR/model/bert_base/multilingual_L-12_H-768_A-12
export SQUAD_DIR=$REFERENCE_DIR/$SQUAD1_PATH

PYTHONPATH=$PYTHONPATH:$PWD/$REFERENCE_DIR
PYTHONPATH=$PYTHONPATH:$PWD/$REFERENCE_DIR/bert
PYTHONPATH=$PYTHONPATH:$PWD


export PYTHONPATH=$PYTHONPATH


echo $PYTHONPATH
echo $BERT_BASE_DIR
echo $SQUAD_DIR

