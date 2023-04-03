import argparse
import json
from utils import import_class
from feeders import feeder
def get_parser():
    parser = argparse.ArgumentParser(description='DenseGCN')

    # parser.add_argument('--weights', default="./weights/ntu_xview/densegcn.pt",
    #                     help='the weights for network initialization')
    parser.add_argument('--device', type=int, default=0, nargs='+',
                        help='the indexes of GPUs for training or testing')
    parser.add_argument('--test-batch-size', type=int, default=64,
                        help='training batch size')
    parser.add_argument('--mode', default='ntu_xset120', help='ntu_xview/ntu_sub/ntu_xset120/ntu_sub120')

    parser.add_argument('--model', default='model.shift_gcn.Model',
                        help='the model to use')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')
    parser.add_argument(
        '--feeder',
        type=str,
        default="feeders.feeder.Feeder",
        help='the type of feeder to use, specified as a path to the class'
    )
    parser.add_argument(
        '--model_args',
        type=json.loads,
        default={"num_class": 60, "num_point": 25, "num_person": 2, "graph": "graph.ntu_rgb_d.Graph",
                 "graph_args": {"labeling_mode": "spatial"}},
        help='the arguments for the model, in the format of a string of dictionary'
    )
    parser.add_argument(
        '--test_feeder_args',
        type=json.loads,
        default={
            "debug": False,
            "random_choose": False,
            "random_shift": False,
            "random_move": False,
            "window_size": -1,
            "normalization": False
        },
        help='the arguments for the feeder used during training, in the format of a string of dictionary'
    )
    opts, _ = parser.parse_known_args()  # all cmd info

    # 更新参数
    opts = update_args(opts)

    # 解析模型参数和数据读取器参数
    # opts.model_args = json.loads(opts.model_args)
    # opts.train_feeder_args = json.loads(opts.train_feeder_args)

    return opts

def update_args(args):
    if args.mode == 'ntu_xview':
        args.test_feeder_args['data_path'] = './data/ntu/xview/test_data_joint.npy'
        args.test_feeder_args['label_path'] = './data/ntu/xview/test_label.pkl'
        args.weights = './weights/ntu_xview/densegcn.pt'
    elif args.mode == 'ntu_sub':
        args.test_feeder_args['data_path'] = './data/ntu/xsub/test_data_joint.npy'
        args.test_feeder_args['label_path'] = './data/ntu/xsub/test_label.pkl'
        args.weights = './weights/ntu_sub/densegcn.pt'
    elif args.mode == 'ntu_xset120':
        args.test_feeder_args['data_path'] = './data/ntu120/xsetup/test_data_joint.npy'
        args.test_feeder_args['label_path'] = './data/ntu120/xsetup/test_label.pkl'
        args.weights = './weights/ntu_xset120/densegcn.pt'
        args.model_args['num_class'] = 120
    elif args.mode == 'ntu_sub120':
        args.test_feeder_args['data_path'] = './data/ntu120/xsub/test_data_joint.npy'
        args.test_feeder_args['label_path'] = './data/ntu120/xsub/test_label.pkl'
        args.weights = './weights/ntu_sub120/densegcn.pt'
        args.model_args['num_class'] = 120
    # args.feeder = import_class(args.feeder)

    return args