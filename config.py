import argparse


class Config:
    parser = argparse.ArgumentParser(description="for gluon weight inference only")
    parser.add_argument(
        "--gluon_weight_path", default="./weights/ko-news-clf-gluon-weight.pth"
    )
    parser.add_argument("--data_path", default="./data/sample.csv")
    parser.add_argument("--save_path", default=None)
    parse = parser.parse_args()
    parser.print_help()
    params = {
        "GLUON_WEIGHT_PATH": parse.gluon_weight_path,
        "SAVE_PATH": parse.save_path,
        "DATA_PATH": parse.data_path,
    }
    gluon_cfg = {"max_len": 128, "batch_size": 32}
