from argparse import ArgumentParser
import os
import predict


def parse_args():
    parser = ArgumentParser(description="English Words Segmentation")
    parser.add_argument("--path", type=str)
    args_ = vars(parser.parse_args())
    if 'path' not in args_ or args_['path'] is None:
        parser.error('You need to specify the path using "--path" flag')
    else:
        return args_


def parse_file(input_file_path):
    with open(input_file_path, encoding='utf-8', mode='r') as file_:
        lines = file_.readlines()
    return [line.strip() for line in lines]


def predict_outputs(params):
    file_path = params['path']
    output_path = os.path.join(os.getcwd(), 'en.wiki.gold.test')
    RESOURCES_PATH = os.path.join(os.getcwd(), 'resources')
    model_path = os.path.join(RESOURCES_PATH, 'bilstm_model.pt')
    test_x = parse_file(file_path)
    predict.tokenize_outputs(model_path, test_x, output_path)


if __name__ == '__main__':
    args = parse_args()
    predict_outputs(args)
