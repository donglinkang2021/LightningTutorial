code_dir="03-SAVING_AND_LOADING_CHECKPOINTS_BASIC"

nohup python $code_dir/train.py > $code_dir/output.log &

# python read_log.py --log_file $code_dir/output.log