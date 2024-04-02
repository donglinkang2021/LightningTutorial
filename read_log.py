import argparse

def read_log(log_file):
    with open(log_file, "r", encoding="utf-8") as file:
        data = file.read()
        file_size = len(data) / 1024 / 1024
        print(f"nohup.out size: {file_size:.2f} MB")
        data = data.strip().split("\n")
        print("\n".join(data[:20]))
        print("\n".join(data[-10:]))

if __name__ == "__main__":
    # python read_log.py --log_file 01-TRAIN_A_MODEL_BASIC/output.log
    # add description to the argument
    parser = argparse.ArgumentParser(description="Read the log file")
    # add argument
    parser.add_argument("--log_file", type=str, help="The log file")
    # parse the argument
    args = parser.parse_args()
    # read the log file
    read_log(args.log_file)