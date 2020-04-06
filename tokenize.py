from argparse import ArgumentParser
import os
import predict
from utils import load_pickle


def parse_args():
    parser = ArgumentParser(description="English Words Segmentation")
    parser.add_argument("--path", type=str)
    args = vars(parser.parse_args())
    if args['path'] not in args or args['path'] is None:
        parser.error('You need to specify the path using "--path" flag')
    else:
        return args



def parse_file(input_file_path):
    lines = []
    with open(input_file_path, encoding='utf-8', mode='r') as file_:
        lines = file_.readlines()
    return [line.strip() for line in lines]

def predict_outputs(params):
    #TODO: Check if it works params['path']
    file_path = params['path']
    output_path = os.path.join(os.getcwd(), 'en.wiki.gold.test')
    RESOURCES_PATH = os.path.join(os.getcwd(), 'resources')
    model_path = os.path.join(RESOURCES_PATH, 'bilstm_model.pt')
    dataset_path = os.path.join(RESOURCES_PATH, 'train_dataset.pkl')
    train_dataset = load_pickle(dataset_path)

    test_x = parse_file(file_path)
    predict.tokenize_outputs(model_path, test_x, train_dataset, output_path)
    # TODO: Check https://github.com/jidasheng/bi-lstm-crf


if __name__ == '__main__':
    args = parse_args()
    predict_outputs(args)