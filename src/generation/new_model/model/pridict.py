import logging
import pandas as pd
import json
from tqdm import tqdm
from simpletransformers.seq2seq import Seq2SeqModel
import torch

CUDA_VISIBLE_DEVICES=0
cuda_available = torch.cuda.is_available()
torch.multiprocessing.set_sharing_strategy('file_system')
if not cuda_available:
    assert 'cuda not available'

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

model = Seq2SeqModel(encoder_decoder_type="bart", encoder_decoder_name="outputs/checkpoint-114040-epoch-40")


def predict(path_in, path_out):
    df = pd.read_csv(path_in, sep='\t')
    to_predict = []
    for i in tqdm(range(len(df))):
        to_predict.append(df['source'][i])
    predict = model.predict(to_predict)
    for i in tqdm(range(len(df))):
        df['pridict'][i] = predict[i]
    df.to_csv(path_out, sep='\t', index=False)


path_in = 'data/predicts/dev_seen/new_upload_dr_teit.csv'
path_out = 'data/predicts/dev_seen/new_upload_dr_teit.csv'

predict(path_in, path_out)

path_in = 'data/predicts/dev_seen/new_with_true_upload_dr_teit.csv'
path_out = 'data/predicts/dev_seen/new_with_true_upload_dr_teit.csv'

predict(path_in, path_out)


# while True:
    # original = input("Enter text to paraphrase: ")
    # to_predict = [original]

    # preds = model.predict(to_predict)

    # print("---------------------------------------------------------")
    # print(original)

    # print()
    # print("Predictions >>>")
    # for pred in preds[0]:
    #     print(pred)

    # print("---------------------------------------------------------")
    # print()