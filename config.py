import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    # threat model parameters
    parser.add_argument('--attack', type=str, default='attacker2', help='1/2/3')
    # model parameters
    parser.add_argument('--dataset', type=str, default='VGGfaceMY', help='name of dataset,cifar10/VGGfaceMY')
    parser.add_argument('--model', type=str, default='vgg', help='name of model')
    parser.add_argument('--add_ic_attack', nargs="+", type=int, default=[1,1,1,1,1], help='name of model')
    parser.add_argument('--add_ic_victim', nargs="+", type=int, default=[1,1,1,1,1], help='name of model')

    # train parameters
    parser.add_argument('--gpuid', type=int, default=5, help='')
    parser.add_argument('--seed', type=int, default=1023, help='')
    parser.add_argument('--cov_lambda', type=float, default=3, help='cov loss setting')
    parser.add_argument('--target_attr', type=int, default=0, help='cov loss setting')
    parser.add_argument('--smooth_upper', type=float, default=0.6, help='upper bound of label smoothing')
    parser.add_argument('--attr_ic', nargs="+", type=int, default=[-1,-1,0,0,0] , help='')
    parser.add_argument('--attr_victim', nargs="+", type=int, default=[-1,-1,0,0,0], help='')
    parser.add_argument('--attr_num', type=int, default=2, help='')
    parser.add_argument('--need_smooth', type=bool, default=False, help='')
    parser.add_argument('--need_cov', type=bool, default=False, help='')

    #eval parameters
    parser.add_argument('--thres', nargs="+", type=float, default=[0.95,0.9,0.8], help='')
    parser.add_argument('--save_path', type=str, default='./results/cifar10.json', help='')
    return parser.parse_args()