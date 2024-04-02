code_dir="02-VALIDATE_AND_TEST_A_MODEL_BASIC"

# nohup python $code_dir/train.py > $code_dir/output.log &

python read_log.py --log_file $code_dir/output.log