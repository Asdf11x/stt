"""run_lightning.py:

21.02.2020:

Using the transformer model with pytorch lightning. Using only the transformer model here because its suited for the usage with lightning.
I think moving the seq2seq with attention to lightning is tricky because of the AttnSeq2Seq class

"""
from __future__ import unicode_literals, division
import warnings
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
import torch.utils
import torch.utils.data
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
import sys
import json
from pathlib import Path
from tensorboardX import SummaryWriter
import time
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge
from statistics import mean
import math
import shutil
import traceback
from jiwer import wer
import numpy as np

# use try/except -> local and server import differs
try:
    from run.model_transformer import TransformerModel
    from run.data_loader import TextKeypointsDataset, ToTensor
    from run.data_utils import DataUtils
except ImportError:  # server uses different imports than local
    from data_loader import TextKeypointsDataset, ToTensor
    from data_utils import DataUtils
    from model_transformer import TransformerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
import traceback


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Litty(LightningModule):

    def __init__(self, hparams_path, timestr):
        super().__init__()

        # ____________________________________________________________________________________
        # SET PARAMS
        # ____________________________________________________________________________________
        with open(hparams_path) as json_file:
            config = json.load(json_file)

        # model settings
        self.input_size = config["model_settings"]["input_size"]
        self.output_size = config["model_settings"]["output_size"]
        self.hidden_size = config["model_settings"]["hidden_size"]
        self.num_layers = config["model_settings"]["num_layers"]
        self.padding = config["model_settings"]["padding"]
        self.dropout = config["model_settings"]["dropout"]
        self.batch_size = config["model_settings"][
            "batch_size"]  # removed forward pass of validation step, comment out if above 1
        self.num_workers = config["model_settings"]["num_workers"]

        self.learning_rate = config["learning_rate_settings"]["learning_rate"]
        self.lr = self.learning_rate

        # trans model settings
        self.nhead = config["trans_settings"]["nhead"]

        # train settings
        self.num_iteration = config["train_settings"]["num_iteration"]

        # train
        self.path_to_numpy_file_train = Path(config["train_paths"]["path_to_numpy_file_train"])
        self.path_to_csv_train = Path(config["train_paths"]["path_to_csv_train"])

        # val
        self.path_to_numpy_file_val = config["val_paths"]["path_to_numpy_file_val"]
        self.path_to_csv_val = config["val_paths"]["path_to_csv_val"]

        # test
        self.path_to_numpy_file_test = config["test_paths"]["path_to_numpy_file_test"]
        self.path_to_csv_test = config["test_paths"]["path_to_csv_test"]

        # vocab file, containing unique words for all (train, val & test)
        self.path_to_vocab_file_all = config["vocab_file"]["path_to_vocab_file_all"]

        # set tokens
        self.PAD_token = 0
        self.UNK_token = 1
        self.SOS_token = 2
        self.EOS_token = 3

        # save / load
        # if not empty use path, else create new folder, use only when documentation exists
        self.save_model_folder_path = config["save_load"]["save_model_folder_path"]
        self.load_model = config["save_load"]["load_model"]
        self.load_model_path = config["save_load"]["load_model_path"]
        self.load_folder_path = config["save_load"]["load_folder_path"]

        if self.load_model:
            self.current_folder = Path(self.load_folder_path)
        else:
            self.current_folder = Path(self.save_model_folder_path) / timestr

        self.writer = SummaryWriter(self.current_folder)
        self.metrics = {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [], "meteor": [], "rouge": [], "wer": []}

        self.save_params(hparams_path, self.current_folder)

        # ____________________________________________________________________________________
        # DEFINE MODEL
        # ____________________________________________________________________________________
        # self.output_size  # the size of vocabulary
        # self.input_size  # embedding dimension
        # self.hidden_size  # the dimension of the feedforward network model in nn.TransformerEncoder
        # self.num_layers  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        # self.nhead  # the number of heads in the multiheadattention models
        # self.dropout  # the dropout value
        self.model = TransformerModel(self.output_size, self.input_size, self.nhead, self.hidden_size, self.num_layers,
                                      self.dropout).to(device)

    def save_params(self, hparams_path, current_folder):
        # save used parameter file
        shutil.copyfile(hparams_path, current_folder / "summary.json")

    def forward(self, src, trg):
        return self.model(src, trg)

    def val_dataloader(self):
        text2kp_val = TextKeypointsDataset(
            path_to_numpy_file=self.path_to_numpy_file_val,
            path_to_csv=self.path_to_csv_val,
            path_to_vocab_file=self.path_to_vocab_file_all,
            input_length=self.input_size,
            transform=ToTensor(),
            kp_max_len=self.padding,
            text_max_len=self.padding)
        data_loader_val = torch.utils.data.DataLoader(text2kp_val, batch_size=self.batch_size,
                                                      num_workers=self.num_workers)

        return data_loader_val

    def validation_step(self, batch, batch_idx):
        rouge = Rouge()
        source_tensor, target_tensor, no_sos, no_eos = batch

        target_tensor = target_tensor.view(1, self.padding)
        target_tensor = target_tensor.type(torch.LongTensor).to(target_tensor.device)

        no_sos = no_sos.view(1, self.padding)
        no_sos = no_sos.type(torch.LongTensor).to(no_sos.device)

        no_eos = no_eos.view(1, self.padding)
        no_eos = no_eos.type(torch.LongTensor).to(no_eos.device)

        # ________
        # COMPUTE LOSS
        # ________
        output = self(source_tensor, no_eos)
        output_dim = output.shape[-1]
        ignore_index = DataUtils().text2index(["<pad>"], DataUtils().vocab_word2int(self.path_to_vocab_file_all))[0][0]
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        loss = criterion(output.view(-1, output_dim), no_sos.view(-1))

        # ________
        # COMPUTE METRICS
        # ________
        # comment if batch size > 1, removed for batch sizes above 1

        flat_list = []  # sentence representation in int
        for sublist in target_tensor[0].tolist():
            flat_list.append(sublist)
        hypothesis = DataUtils().int2text(flat_list, DataUtils().vocab_int2word(self.path_to_vocab_file_all))
        hypothesis = list(filter("<pad>".__ne__, hypothesis))
        hypothesis = list(filter("<eos>".__ne__, hypothesis))
        hypothesis = list(filter("<sos>".__ne__, hypothesis))
        hyp_str = " ".join(hypothesis)

        # FULL
        # online approach
        decoded_words = []
        for ot in range(output.size(0)):
            topv, topi = output[ot].topk(1)
            if topi[0].item() == self.EOS_token:
                decoded_words.append('<eos>')
                break
            else:
                decoded_words.append(topi[0].item())

        # ONE BY ONE
        # uncomment to use, this approach takes longer to reproduce results. Original approach from "attention is all you need"
        # memory = self.model.transformer.encoder(self.model.pos_encoder(source_tensor))
        # sos_index = DataUtils().text2index(["<sos>"], DataUtils().vocab_word2int(self.path_to_vocab_file_all))[0][0]
        # decoded_words = [sos_index, ]
        #
        # for i in range(self.max_length):
        #     trg_tensor = torch.LongTensor(decoded_words).unsqueeze(1).to(device)
        #
        #     output = self.model.fc_out(self.model.transformer.decoder(self.model.pos_decoder(self.model.decoder_emb(trg_tensor)), memory))
        #     out_token = output.argmax(2)[-1].item()
        #     decoded_words.append(out_token)
        #     if out_token == DataUtils().text2index(["<eos>"], DataUtils().vocab_word2int(self.path_to_vocab_file_all))[0][0]:
        #         break

        reference = DataUtils().int2text(decoded_words, DataUtils().vocab_int2word(self.path_to_vocab_file_all))
        reference = list(filter("<pad>".__ne__, reference))
        reference = list(filter("<eos>".__ne__, reference))
        reference = list(filter("<sos>".__ne__, reference))
        reference = " ".join(reference[:len(
            hypothesis)])  # cut too long sentences, can be uncommented if model starts to predict correct lengths
        ref_str = " ".join(reference)

        print(f"\nhyp_str: {hyp_str}")
        print(f"ref_str: {ref_str}")

        bleu1_score = round(sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0)), 4)
        bleu2_score = round(sentence_bleu([reference], hypothesis, weights=(0.5, 0.5, 0, 0)), 4)
        bleu3_score = round(sentence_bleu([reference], hypothesis, weights=(0.33, 0.33, 0.33, 0)), 4)
        bleu4_score = round(sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25)), 4)
        meteor_score = round(single_meteor_score(ref_str, hyp_str), 4)
        wer_score = round(wer(hyp_str, ref_str), 4)
        try:
            rouge_score = round(rouge.get_scores(hyp_str, ref_str)[0]["rouge-l"]["f"], 4)
        except ValueError:
            rouge_score = 0.0

        self.metrics["bleu1"].append(bleu1_score)
        self.metrics["bleu2"].append(bleu2_score)
        self.metrics["bleu3"].append(bleu3_score)
        self.metrics["bleu4"].append(bleu4_score)
        self.metrics["meteor"].append(meteor_score)
        self.metrics["rouge"].append(rouge_score)
        self.metrics["wer"].append(wer_score)

        self.writer.add_scalars(f'metrics', {
            'bleu1': mean(self.metrics["bleu1"]),
            'bleu2': mean(self.metrics["bleu2"]),
            'bleu3': mean(self.metrics["bleu3"]),
            'bleu4': mean(self.metrics["bleu4"]),
            'meteor': mean(self.metrics["meteor"]),
            'rouge': mean(self.metrics["rouge"]),
            'wer': mean(self.metrics["wer"]),
        }, self.current_epoch)

        self.writer.add_scalar('lr', self.learning_rate, self.current_epoch)

        # reset
        self.metrics = {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [], "meteor": [], "rouge": [], "wer": []}

        return {'val_loss': loss.item()}

    def validation_epoch_end(self, outputs):
        avg_loss = np.mean([x['val_loss'] for x in outputs])
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))
        # removed .ReduceLROnPlateau() approach, add here if wanted
        return optimizer

    def train_dataloader(self):
        text2kp_train = TextKeypointsDataset(path_to_numpy_file=self.path_to_numpy_file_train,
                                             path_to_csv=self.path_to_csv_train,
                                             path_to_vocab_file=self.path_to_vocab_file_all,
                                             input_length=self.input_size,
                                             transform=ToTensor(), kp_max_len=self.padding, text_max_len=self.padding)
        data_loader_train = torch.utils.data.DataLoader(text2kp_train, batch_size=self.batch_size, shuffle=True,
                                                        num_workers=self.num_workers)
        return data_loader_train

    def training_step(self, batch, batch_idx):
        source_tensor, target_tensor, no_sos, no_eos = batch

        no_sos = no_sos.view(1, self.padding)
        no_sos = no_sos.type(torch.LongTensor).to(no_sos.device)

        no_eos = no_eos.view(1, self.padding)
        no_eos = no_eos.type(torch.LongTensor).to(no_eos.device)

        ignore_index = DataUtils().text2index(["<pad>"], DataUtils().vocab_word2int(self.path_to_vocab_file_all))[0][0]
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

        output = self(source_tensor, no_eos)
        output_dim = output.shape[-1]

        loss = criterion(output.view(-1, output_dim), no_sos.view(-1))

        tensorboard_logs = {'train_loss': loss.item()}
        return {'loss': loss, 'log': tensorboard_logs}

    def on_epoch_end(self):
        trainer.save_checkpoint(self.current_folder / "model.ckpt")


if __name__ == '__main__':
    timestr = time.strftime("%Y-%m-%d_%H-%M")

    # set path to file containing all parameters
    if len(sys.argv) > 1:
        hparams_path = str(sys.argv[1])
    else:
        hparams_path = r"hparams_isl.json"
    model = Litty(hparams_path, timestr)
    if model.load_model == 1:
        trainer = Trainer(gpus=1, default_save_path=Path(model.save_model_folder_path) / timestr,
                          resume_from_checkpoint=model.load_model_path, min_epochs=model.num_iteration,
                          max_epochs=model.num_iteration, gradient_clip_val=1)
    else:
        trainer = Trainer(gpus=1, default_save_path=Path(model.save_model_folder_path) / timestr,
                          min_epochs=model.num_iteration,
                          max_epochs=model.num_iteration,
                          gradient_clip_val=1)
    trainer.fit(model)
    # trainer.save_checkpoint(Path(model.save_model_folder_path) / timestr / "model.ckpt")
