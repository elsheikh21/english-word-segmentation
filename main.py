import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from utilities import save_pickle
from evaluator import compute_scores, pprint_confusion_matrix
from model import HyperParameters, BaselineModel, load_pretrained_embeddings, BiLSTM_CRF
from data_loader import WikiDataset
from training import Trainer

if __name__ == '__main__':
    DATA_PATH = os.path.join(os.getcwd(), 'data')
    RESOURCES_PATH = os.path.join(os.getcwd(), 'resources')

    file_path = os.path.join(DATA_PATH, 'en.wiki.sentences.train')
    gold_file_path = os.path.join(DATA_PATH, 'en.wiki.gold.train')

    print('========== Training Dataset ==========')
    train_dataset = WikiDataset(file_path, gold_file_path)
    train_dataset.vectorize_data()

    # train_x = torch.LongTensor(train_dataset.train_x)
    # print(f'train_x shape is: {train_x.shape}')
    # x.shape = [number of samples, max characters/sentence] = [31_553, 256]
    # train_y = torch.LongTensor(train_dataset.train_y)
    # print(f'train_y shape is: {train_y.shape}')
    # y.shape = [number of samples, max characters/sentence] = [31_553, 256]

    char2idx_path_save = os.path.join(RESOURCES_PATH, 'char2idx.pkl')
    save_pickle(char2idx_path_save, train_dataset.char2idx)

    label2idx_path_save = os.path.join(RESOURCES_PATH, 'label2idx.pkl')
    save_pickle(label2idx_path_save, train_dataset.label2idx)

    print('\n========== Validation Dataset ==========')
    dev_file_path = os.path.join(DATA_PATH, 'en.wiki.sentences.dev')
    dev_gold_file_path = os.path.join(DATA_PATH, 'en.wiki.gold.dev')
    dev_dataset = WikiDataset(dev_file_path, dev_gold_file_path)
    dev_dataset.char2idx = train_dataset.char2idx
    dev_dataset.idx2char = train_dataset.idx2char
    dev_dataset.label2idx = train_dataset.label2idx
    dev_dataset.idx2label = train_dataset.idx2label
    dev_dataset.vectorize_data()

    # dev_x = torch.LongTensor(dev_dataset.train_x)
    # print(f'dev_x shape is: {dev_x.shape}')
    # x.shape = [number of samples, max characters/sentence] = [3_994 , 256]
    # dev_y = torch.LongTensor(dev_dataset.train_y)
    # print(f'dev_y shape is: {dev_y.shape}')
    # y.shape = [number of samples, max characters/sentence] = [3_994 , 256]

    embeddings_size = 300
    fname = os.path.join(RESOURCES_PATH, 'wiki.en.vec')
    pretrained_embeddings = load_pretrained_embeddings(fname, train_dataset.char2idx, embeddings_size)

    hyperparams = HyperParameters()
    hyperparams.vocab_size = train_dataset.vocab_size
    hyperparams.num_classes = train_dataset.out_vocab_size
    hyperparams.embeddings = pretrained_embeddings

    baseline_model = BaselineModel(hyperparams)
    print('\n========== Model Summary ==========')
    print(baseline_model)

    train_dataset_ = DataLoader(train_dataset,
                                batch_size=hyperparams.batch_size,
                                shuffle=True,
                                collate_fn=WikiDataset.pad_collate)

    dev_dataset_ = DataLoader(dev_dataset,
                              batch_size=hyperparams.batch_size,
                              shuffle=False,
                              collate_fn=WikiDataset.pad_collate)


    trainer = Trainer(
        model=baseline_model,
        loss_function=nn.CrossEntropyLoss(
            ignore_index=train_dataset.label2idx['<PAD>']),
        optimizer=Adam(baseline_model.parameters(), lr=1e-6, weight_decay=1e-5),
        label_vocab=train_dataset.label2idx
    )

    trainer.train(train_dataset_, dev_dataset_, epochs=2)
    save_model_path = os.path.join(RESOURCES_PATH, f'{baseline_model.name}_model.pt')
    torch.save(baseline_model.state_dict(), save_model_path)

    # load model
    # model = BaselineModel(hyperparams)
    # model.load_state_dict(torch.load(save_model_path))

    # evaluate model
    # test_loss, test_acc = trainer.evaluate(test_dataset)
    # print(f"Test set\nLoss: {test_loss:.5f}, Acc: {test_acc * 100:.5f}%")

    scores = compute_scores(baseline_model, dev_dataset_)

    per_class_precision = scores["per_class_precision"]

    precision_, recall_, f1score_, _ = scores['macro_precision_recall_fscore']
    print(f"Macro Precision: {precision_}")
    print(f"Macro Recall: {recall_}")
    print(f"Macro F1_Score: {f1score_}")

    print("Per class Precision:")
    for idx_class, precision in sorted(enumerate(per_class_precision),
                                       key=lambda elem: -elem[1]):
        label = train_dataset.idx2label[idx_class]
        print(f'{label}: {precision}')

    precision, recall, f1score, _ = scores['micro_precision_recall_fscore']
    print(f"Micro Precision: {precision}")
    print(f"Micro Recall: {recall}")
    print(f"Micro F1_Score: {f1score}")

    confusion_matrix = scores['confusion_matrix']
    print(confusion_matrix)
    pprint_confusion_matrix(confusion_matrix, num_classes=3)
