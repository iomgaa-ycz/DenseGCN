import os
import gc
import torch
import numpy as np
import tqdm
import collections
from model.shift_gcn import Model
from torch.autograd import Variable
from utils import *
import shutil
import inspect
from tqdm import tqdm
import torch.nn as nn


class Processor():
    def __init__(self, arg):
        super(Processor, self).__init__()
        self.arg = arg
        self.model = self.load_model()
        self.data_loader=self.load_data()
        self.loss = nn.CrossEntropyLoss().cuda(self.arg.device)
        self.best_acc = 0

    def start(self):
        wf = './output/wrong.txt'
        rf = './output/right.txt'
        self.eval(epoch=0, loader_name=['test'], wrong_file=wf, result_file=rf)

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()

        for ln in loader_name:
            loss_value = []
            score_frag = []

            step = 0
            process = tqdm(self.data_loader)
            for batch_idx, (data, label, index) in enumerate(process):
                data = Variable(
                    data.float().cuda(self.arg.device),
                    requires_grad=False,
                    volatile=True)
                label = Variable(
                    label.long().cuda(self.arg.device),
                    requires_grad=False,
                    volatile=True)

                with torch.no_grad():
                    output = self.model(data)

                loss = self.loss(output, label)
                score_frag.append(output.data.cpu().numpy())
                loss_value.append(loss.data.cpu().numpy())

                _, predict_label = torch.max(output.data, 1)
                step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)

            accuracy = self.data_loader.dataset.top_k(score, 1)

            print('Eval Accuracy: ', accuracy)




    def load_model(self):
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        model = Model(**self.arg.model_args).cuda(self.arg.device)
        # 加载预训练权重
        checkpoint_file = self.arg.weights
        if not os.path.exists(checkpoint_file):  # from scratch
            return None
        else:  # get chk points
            print("=> loading checkpoint '{}'".format(checkpoint_file))
            device = torch.device(self.arg.device)
            model_dict = model.state_dict()

            # import pickle
            # with open(checkpoint_file, 'rb') as f:
            #     obj = f.read()
            # weights = {key: weight_dict for key, weight_dict in pickle.loads(obj, encoding='latin1').items()}

            checkpoint = torch.load(checkpoint_file)
            # checkpoint_epoch = checkpoint['epoch']
            checkpoint2 = collections.OrderedDict()
            for k, v in checkpoint.items():
                try:
                    name = k
                    if np.shape(model_dict[name]) == np.shape(v):
                        checkpoint2[name] = v
                except:
                    continue
            checkpoint = checkpoint2
            model_dict.update(checkpoint)
            model.load_state_dict(model_dict, strict=False)
            del checkpoint, checkpoint2, model_dict
            gc.collect()
            return model
    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        data_loader = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False)
        return data_loader