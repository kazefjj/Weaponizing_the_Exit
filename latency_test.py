from config import get_arguments
import torch
import numpy as np
import json
import os
import time
import aux_funcs  as af
import model_funcs as mf
import network_architectures as arcs
import json
from profiler import profile_sdn, profile
import os
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
import numpy as np


def collect_latency_records(model, loader, device,train=True):
    """
    收集:
        attr
        latency
        true_exit
        label
    """

    records = []

    model.eval()
    if train == True:

        repeat = 5
    else:
        repeat = 1
    with torch.no_grad():

        for batch in loader:

            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            prop = batch[2].to(device)

            if "cuda" in str(device):
                torch.cuda.synchronize()

            latencies = []

            for _ in range(repeat):

                if "cuda" in str(device):
                    torch.cuda.synchronize()

                start = time.perf_counter()

                output, output_id, is_early = model(b_x)

                if "cuda" in str(device):
                    torch.cuda.synchronize()

                end = time.perf_counter()

                latencies.append(end - start)

            latency = float(np.mean(latencies))

            true_exit = (
                output_id
                if is_early
                else model.num_output - 1
            )

            records.append({
                "attr": int(prop.item()),
                "label": int(b_y.item()),
                "latency": latency,
                "exit_id": int(true_exit)
            })
    return records


def train_kde(train_records, bandwidth=1e-3):

    exit_ids = sorted(
        set(x["exit_id"] for x in train_records)
    )

    kdes = {}

    for exit_id in exit_ids:

        X = np.array([
            x["latency"]
            for x in train_records
            if x["exit_id"] == exit_id
        ]).reshape(-1, 1)

        if len(X) < 2:
            continue

        kde = KernelDensity(
            kernel="gaussian",
            bandwidth=bandwidth
        )

        kde.fit(X)

        kdes[exit_id] = kde

        print(
            f"Train KDE Exit {exit_id}: "
            f"{len(X)} samples"
        )

    return kdes


def predict_exits(test_records, kdes):

    predictions = []

    for sample in test_records:

        latency = sample["latency"]

        scores = {}

        for exit_id, kde in kdes.items():

            scores[exit_id] = kde.score_samples(
                np.array([[latency]])
            )[0]

        pred_exit = max(
            scores,
            key=scores.get
        )

        predictions.append({
            "attr": sample["attr"],
            "gt_exit": sample["exit_id"],
            "pred_exit": pred_exit
        })

    return predictions


def evaluate_predictions(predictions):

    gt = [x["gt_exit"] for x in predictions]
    pred = [x["pred_exit"] for x in predictions]

    acc = np.mean(
        np.array(gt) == np.array(pred)
    )

    print("\n==========================")
    print(
        f"KDE Recovery Accuracy: "
        f"{acc * 100:.2f}%"
    )
    print("==========================\n")


    return acc


def attribute_exit_statistics(
        predictions,
        attr_num,
        num_exits
):
    """
    counts[attr][exit]
    """

    counts = [
        [0] * num_exits
        for _ in range(attr_num)
    ]

    for sample in predictions:

        attr = sample["attr"]
        pred_exit = sample["pred_exit"]

        counts[attr][pred_exit] += 1

    print("\n==========================")
    print("Predicted Exit Statistics")
    print("==========================")

    for attr in range(attr_num):

        print(
            f"Attribute {attr}: "
            f"{counts[attr]}"
        )

    return counts


def run_kde_attack(
        model,
        train_loader,
        test_loader,
        device,
        attr_num,
        bandwidth=1e-3
):
    """
    完整攻击流程
    """

    print("Collecting training latency...")

    train_records = collect_latency_records(
        model,
        train_loader,
        device
    )

    print("Collecting testing latency...")

    test_records = collect_latency_records(
        model,
        test_loader,
        device,
        train=False
    )

    print("Training KDEs...")

    kdes = train_kde(
        train_records,
        bandwidth
    )

    print("Predicting exits...")

    predictions = predict_exits(
        test_records,
        kdes
    )

    acc = evaluate_predictions(
        predictions
    )

    counts = attribute_exit_statistics(
        predictions,
        attr_num=attr_num,
        num_exits=model.num_output
    )

    return {
        "accuracy": acc,
        "counts": counts,
        "predictions": predictions
    }

def early_exit_experiments(models_path, args, device='cpu'):
    #sdn_training_type = 'sdn_training_bone_only' # IC-only training
    #sdn_training_type = 'sdn_training' # SDN training
    #sdn_training_type = 'sdn_training_bone_only_ic_only'
    if 'attacker1' in args.attack:
        sdn_training_types = ['ic_only_bone_only']
        #sdn_training_types = ['sdn_training']
    else:
        sdn_training_types = ['ic_only','ic_only_bone_only','ic_only_bone_only_ic_only']
        #sdn_training_types = ['ic_only_bone_only_ic_only']
    #sdn_training_type = 'ic_only_bone_only'
    #sdn_training_type = 'ic_only'
    #task = 'cifar10'
    #task = 'VGGfaceMY'
    task = args.dataset
    #task = 'cifar10'
    #task = 'tinyimagenet' 
    if args.model == 'vgg':
        sdn_name = 'vgg16bn_sdn'
    elif args.model == 'resnet':
        sdn_name = 'resnet56_sdn'
    elif args.model == 'wideresnet':
        sdn_name = 'wideresnet32_4_sdn'
    elif args.model == 'mobilenet':
        sdn_name = 'mobilenet_sdn'
    elif args.model == 'fcn_1':
        sdn_name = 'fcn_1_sdn'
    elif args.model == 'fcn_2':
        sdn_name = 'fcn_2_sdn'
    elif args.model == 'fcn_3':
        sdn_name = 'fcn_3_sdn'
    elif args.model == 'fcn_4':
        sdn_name = 'fcn_4_sdn'
    #sdn_names = ['vgg16bn_sdn', 'resnet56_sdn', 'wideresnet32_4_sdn', 'mobilenet_sdn']
    

    sdn_names = [task + '_' + sdn_name + '_' + sdn_training_type for sdn_training_type in sdn_training_types]
    multi_acc = []
    acc = []
    counts = []
    precision = []
    recall = []
    ops = []
    attr_ic = args.attr_ic
    attr_victim = args.attr_victim
    confidence_thresholds = args.thres
    for sdn_name in sdn_names:
        cnn_name = sdn_name.replace('sdn', 'cnn')
        cnn_name = cnn_name.replace('_ic_only', '')
        cnn_name = cnn_name.replace('_sdn_training', '')

        print(sdn_name)
        print(cnn_name)
        if 'ic_only_bone_only_ic_only' in sdn_name:
            models_path = models_path + '/victim'
        
        sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
        sdn_model.to(device)

        dataset = af.get_dataset(sdn_params['task'])

        #cnn_model, _ = arcs.load_model(models_path, cnn_name, epoch=-1)
        #cnn_model.to(device)

        print('Get CNN results')
        #top1_test, top5_test, total_time = mf.cnn_test_time(cnn_model, dataset.test_loader, device)
        #total_ops, total_params = profile(cnn_model, cnn_model.input_size, device)
        #print("#Ops: %f GOps"%(total_ops/1e9))
        #print("#Parameters: %f M"%(total_params/1e6))
        #print('Top1 Test accuracy: {}'.format(top1_test))
        #print('Top5 Test accuracy: {}'.format(top5_test))

        #print('25 percent cost: {}'.format((total_ops/1e9)*0.25))
        #print('50 percent cost: {}'.format((total_ops/1e9)*0.5))
        #print('75 percent cost: {}'.format((total_ops/1e9)*0.75))


        # to test early-exits with the SDN
        one_batch_dataset = af.get_dataset(sdn_params['task'], 1)
        print('Get SDN early exit results')
        total_ops, total_params = profile_sdn(sdn_model, sdn_model.input_size, device)
        print("#Ops (GOps): {}".format(total_ops))
        print("#Params (mil): {}".format(total_params))
        top1_multi_test, top5_test = mf.sdn_test(sdn_model, dataset.test_loader, device)
        print('Top1 Test accuracy: {}'.format(top1_multi_test))
        #print('Top5 Test accuracy: {}'.format(top5_test))
        #mf.compare_confidence_distributions(sdn_model,dataset.color_test_loader,dataset.gray_test_loader,device=device)
        #print('Calibrate confidence_thresholds')
         # search for the confidence threshold for early exits
        c_i = profile_sdn(sdn_model, sdn_model.input_size, device)[0]
        if hasattr(sdn_model, "name") and 'fcn' in sdn_model.name:
            sdn_model.forward = sdn_model.early_exit_seq 
        else:
            sdn_model.forward = sdn_model.early_exit 
        

        
        for threshold in confidence_thresholds:
            print(threshold)
            sdn_model.confidence_threshold = threshold

            sample_records = []


            result = run_kde_attack(
                model=sdn_model,
                train_loader=one_batch_dataset.train_loader,
                test_loader=one_batch_dataset.test_loader,
                device=device,
                attr_num=args.attr_num,
                bandwidth=1e-4
            )

            print(result["accuracy"])
            print(result["counts"])
            c = result["counts"]
            #print(c[3][-2]/sum(c[3]))
            #print(c[3][-2]/(c[0][-2]+c[1][-2]+c[2][-2]+c[3][-2]))
            print(c[1][-2]/sum(c[1]))
            print(c[1][-2]/(c[0][-2]+c[1][-2]))
def main(args):
    torch.manual_seed(af.get_random_seed())    # reproducible
    np.random.seed(af.get_random_seed())
    device = af.get_pytorch_device(args.gpuid)
    trained_models_path = 'models3'
    #trained_models_path = 'networks/{}'.format(af.get_random_seed())
    #trained_models_path = 'backdoored_models'

    early_exit_experiments(trained_models_path, args, device)
if __name__ == '__main__':
    args = get_arguments()
    main(args)
