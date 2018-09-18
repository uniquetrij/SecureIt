cd models/research
PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ../..
#python create_tf_record.py --data_dir=`pwd` --output_dir=`pwd`
python models/research/object_detection/train.py --logtostderr --pipeline_config_path=pipeline.config --train_dir=train
