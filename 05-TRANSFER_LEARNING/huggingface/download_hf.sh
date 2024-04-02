export HF_ENDPOINT=https://hf-mirror.com
data_dir="/opt/data/private/linkdom/model"
huggingface-cli download --resume-download google-bert/bert-base-uncased --local-dir $data_dir/bert-base-uncased