code_dir="04-EARLY_STOPPING"

nohup python $code_dir/train.py > $code_dir/output.log &

# python read_log.py --log_file $code_dir/output.log