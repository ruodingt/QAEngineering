export REFERENCE_DIR=reference
export BERT_BASE_DIR=$REFERENCE_DIR/model/bert_base/multilingual_L-12_H-768_A-12
export SQUAD1_PATH=data/raw/squad_v1_1
export SQUAD_DIR=$REFERENCE_DIR/$SQUAD1_PATH

PYTHONPATH=
PYTHONPATH=$PYTHONPATH:$PWD/$REFERENCE_DIR
PYTHONPATH=$PYTHONPATH:$PWD/$REFERENCE_DIR/bert
PYTHONPATH=$PYTHONPATH:$PWD
export PYTHONPATH=$PYTHONPATH

echo $PYTHONPATH
echo $BERT_BASE_DIR
echo $SQUAD_DIR
