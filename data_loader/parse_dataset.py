import os

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

from utilities import configure_workspace, save_pickle


class WikiDataset(Dataset):
    def __init__(self, input_file_path: str, gold_file_path: str,
                 max_char_len:int =256, is_crf:bool =False, TASK:str='BIS'):
        configure_workspace()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_char_len
        self.parse_dataset(input_file_path, gold_file_path)
        self.get_unigrams()
        self.create_vocabulary(is_crf)
        self.encode_labels(TASK)
        self.encoded_data = None
        self.vocab_size = len(self.char2idx.keys())
        self.out_vocab_size = len(self.label2idx.keys())

        print(f'Input Vocabulary Size: {self.vocab_size}')
        print(f'Output Vocabulary Size: {self.out_vocab_size}')

    def parse_dataset(self, input_file_path, gold_file_path):
        lines = []
        with open(input_file_path, encoding='utf-8', mode='r') as file_:
            lines = file_.readlines()
        self.data_x = [line.strip() for line in lines]

        lines_ = []
        with open(gold_file_path, encoding='utf-8', mode='r') as gold_file_:
            lines_ = gold_file_.readlines()

        self.data_y = [line.strip() for line in lines_]

    def encode_labels(self, TASK='BIS'):
        if TASK == 'BIS':
            self.label2idx = {'<PAD>':0, 'B': 1, 'I': 2, 'S': 3}
            self.idx2label = {0: '<PAD>', 1: 'B', 2: 'I', 3: 'S'}
        elif TASK == 'BI':
            self.label2idx = {'<PAD>':0, 'B': 1, 'I': 2}
            self.idx2label = {0: '<PAD>', 1: 'B', 2: 'I'}
        else:
            raise NotImplementedError

    def get_unigrams(self):
        chars = []
        for sentence in self.data_x:
            chars.extend([word for word in sentence])
        self.unigrams = sorted(list(set(chars)))

    def create_vocabulary(self, is_crf):
        self.char2idx = dict()
        self.char2idx['<PAD>'] = 0
        self.char2idx['<UNK>'] = 1
        start_= 2
        if is_crf:
            self.char2idx['<BOS>'] = 2
            self.char2idx['<EOS>'] = 3
            start_= 4
        self.char2idx.update({val: key for (key, val) in enumerate(self.unigrams,
                                                  start=start_)})
        self.idx2char = {val: key for (key, val) in self.char2idx.items()}

    """
    def char_padding(self, sentence_):
        if len(sentence_) > self.max_len:
            sentence_ = sentence_[:self.max_len]
        else:
            for _ in range(self.max_len - len(sentence_)):
                sentence_.append(self.label2idx.get('<PAD>'))
        return sentence_
    """

    def vectorize_data(self):
        """
        Converts data_x from a seq of tokens (str) to a seq of indices (int)
        """
        train_x = []
        for sentence in self.data_x:
            sentence_ = [self.char2idx.get(char, 1) for char in sentence]
            train_x.append(sentence_)
            # train_x.append(self.char_padding(sentence_))
        self.train_x = train_x

        train_y = []
        for sentence in self.data_y:
            sentence_ = [self.label2idx.get(label) for label in sentence]
            train_y.append(sentence_)
            # train_y.append(self.char_padding(sentence_))
        self.train_y = train_y
        self.encode_data()
    
    def encode_data(self):
        self.encoded_data = list()
        # data_x.shape = [samples_num, max_chars_sentence]
        assert len(self.train_x) == len(self.train_y)
        for i in range(len(self.train_x)):
            # train_x = torch.LongTensor(self.train_x[i]).to(self.device)
            # train_y = torch.LongTensor(self.train_y[i]).to(self.device)
            self.encoded_data.append({"inputs": self.train_x[i], "outputs": self.train_y[i]})

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        if self.encoded_data is None:
            raise RuntimeError("Trying to retrieve elements, but dataset is not vectorized yet")
        return self.encoded_data[idx]

    @staticmethod
    def decode_output(logits: torch.Tensor, idx2label):
        max_indices = torch.argmax(logits, -1).tolist()  # shape = (batch_size, max_len)
        predictions = list()
        for indices in max_indices:
            predictions.append([idx2label[i] for i in indices])
        return predictions

    @staticmethod
    def decode_data(data: torch.Tensor, idx2label):
        data_ = data.tolist()
        return [idx2label.get(idx, None) for idx in data_]

    @staticmethod
    def pad_collate(batch):
        data_x, data_y = [], []
        for item in batch:
            data_x.append(item.get('inputs'))
            data_y.append(item.get('outputs'))
            seq_lengths = torch.LongTensor(list(map(len, data_x)))
            seq_tensor = Variable(torch.zeros((len(data_x), seq_lengths.max()))).long()
            for idx, (seq, seqlen) in enumerate(zip(data_x, seq_lengths)):
                seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
            seq_tensor = seq_tensor[perm_idx]

            lbl_length = torch.LongTensor(list(map(len, data_y)))
            lbl_tensor = Variable(torch.zeros((len(data_y), lbl_length.max()))).long()
            for idx, (lbl, lbllen) in enumerate(zip(data_y, lbl_length)):
                lbl_tensor[idx, :lbllen] = torch.LongTensor(lbl)
            lbl_lengths, perm_idx = lbl_length.sort(0, descending=True)
            lbl_tensor = lbl_tensor[perm_idx]
            return seq_tensor.to('cuda'), lbl_tensor.to('cuda'), seq_lengths


if __name__ == '__main__':
    DATA_PATH = os.path.join(os.getcwd(), 'data')
    RESOURCES_PATH = os.path.join(os.getcwd(), 'resources')

    file_path = os.path.join(DATA_PATH, 'en.wiki.sentences.train')
    gold_file_path = os.path.join(DATA_PATH, 'en.wiki.gold.train')

    train_dataset = WikiDataset(file_path, gold_file_path)
    train_dataset.vectorize_data()

    train_x = torch.LongTensor(train_dataset.train_y)
    print(f'train_y shape is: {train_x.shape}')
    # x.shape = [number of samples, max characters/sentence] = [31_553 , 256]
    train_y = torch.LongTensor(train_dataset.train_y)
    print(f'train_y shape is: {train_y.shape}')
    # y.shape = [number of samples, max characters/sentence] = [31_553 , 256]

    print('\n========== Validation Dataset ==========')
    dev_file_path = os.path.join(DATA_PATH, 'en.wiki.sentences.dev')
    dev_gold_file_path = os.path.join(DATA_PATH, 'en.wiki.gold.dev')
    dev_dataset = WikiDataset(dev_file_path, dev_gold_file_path)
    dev_dataset.vectorize_data()

    dev_x = torch.tensor(dev_dataset.train_x)
    print(f'dev_x shape is: {dev_x.shape}')
    # x.shape = [number of samples, max characters/sentence] = [3_994 , 256]
    dev_y = torch.tensor(dev_dataset.train_y)
    print(f'dev_y shape is: {dev_y.shape}')
    # y.shape = [number of samples, max characters/sentence] = [3_994 , 256]

    char2idx_path_save = os.path.join(RESOURCES_PATH, 'char2idx.pkl')
    save_pickle(char2idx_path_save, train_dataset.char2idx)

    idx2char_path_save = os.path.join(RESOURCES_PATH, 'idx2char.pkl')
    save_pickle(idx2char_path_save, train_dataset.idx2char)

    label2idx_path_save = os.path.join(RESOURCES_PATH, 'label2idx.pkl')
    save_pickle(label2idx_path_save, train_dataset.label2idx)

    idx2label_path_save = os.path.join(RESOURCES_PATH, 'idx2label.pkl')
    save_pickle(idx2label_path_save, train_dataset.idx2label)
