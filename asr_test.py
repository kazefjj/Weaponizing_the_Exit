# early_exit_experiments.py
# runs the experiments in section 5.1 
from config import get_arguments
import torch
import numpy as np
import json
import os

import aux_funcs  as af
import model_funcs as mf
import network_architectures as arcs
import json
from profiler import profile_sdn, profile
def get_plot_data_and_auc(layer_cumul_dist, ic_costs):
    layers = sorted(list(ic_costs.keys()))

    c_i = {layer: ic_costs[layer] / ic_costs[layers[-1]] for layer in layers}
    c_i = [c_i[layer] for layer in layers]
    c_i.insert(0, 0)
    plot_data = [c_i, layer_cumul_dist]

    area_under_curve = np.trapz(layer_cumul_dist, x=c_i)

    return plot_data, area_under_curve
def convert_num_early_exits_at_each_ic_to_cumulative_dis(ic_exits, total_samples):
    num_exits = len(ic_exits)

    layer_cumul_dist = [0]

    running_total = 0
    for cur_exit in range(num_exits):
        running_total += ic_exits[cur_exit]
        layer_cumul_dist.append(running_total)

    layer_cumul_dist[-1] = total_samples
    layer_cumul_dist = [val / total_samples for val in layer_cumul_dist]
    return layer_cumul_dist
def Tsne(models_path, device='cpu'):
    features = []
    labels_list = []
    #sdn_training_type = 'ic_only' # IC-only training
    sdn_training_type = 'sdn_training' # SDN training


    #task = 'cifar10'
    task = 'cifar10'
    #task = 'tinyimagenet'

    #sdn_names = ['vgg16bn_sdn', 'resnet56_sdn', 'wideresnet32_4_sdn', 'mobilenet_sdn']; add_trigger = False
    
    sdn_names = ['vgg16bn_sdn']; add_trigger = False
    sdn_names = [task + '_' + sdn_name + '_' + sdn_training_type for sdn_name in sdn_names]

    
    for sdn_name in sdn_names:
        cnn_name = sdn_name.replace('sdn', 'cnn')
        cnn_name = cnn_name.replace('_ic_only', '')
        cnn_name = cnn_name.replace('_sdn_training', '')

        print(sdn_name)
        print(cnn_name)

        sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
        sdn_model.to(device)

        dataset = af.get_dataset(sdn_params['task'])
        def hook(module, input, output):
            features.append(output.cpu().numpy())
        for name, module in sdn_model.named_modules():
            print (name)
            if name == 'layers.8.output.linear':
                hook_handle = module.register_forward_hook(hook)
        
        sdn_model.eval()
        with torch.no_grad():
            for batch in dataset.train_loader:
                b_x = batch[0].to(device)
                b_y = batch[1].to(device)
                labels_list.append(b_y.cpu().numpy())
                output = sdn_model(b_x)        
        hook_handle.remove()
        all_labels = np.concatenate(labels_list, axis=0)
        all_features = np.concatenate(features, axis=0)
        class_means = af.compute_class_mean(all_features, all_labels)
        
        af.visualize_tsne(all_features, all_labels, perplexity=30, n_iter=1000, num_classes=10)
def convert_num_early_exits_at_each_ic_to_cumulative_dis(ic_exits, total_samples):
    num_exits = len(ic_exits)

    layer_cumul_dist = [0]

    running_total = 0
    for cur_exit in range(num_exits):
        running_total += ic_exits[cur_exit]
        layer_cumul_dist.append(running_total)

    layer_cumul_dist[-1] = total_samples
    layer_cumul_dist = [val / total_samples for val in layer_cumul_dist]
    return layer_cumul_dist


# ic_exits  --- at each IC, the samples that exit (list-dict of set)
# ic_costs --- the output of the profiler_sdn for the network (dict)
# total samples --- how many samples were in the test set (int)
# return --> the data to draw a delay plot and the area under the curve as our delay metric
def get_plot_data_and_auc(layer_cumul_dist, ic_costs):
    layers = sorted(list(ic_costs.keys()))

    c_i = {layer: ic_costs[layer] / ic_costs[layers[-1]] for layer in layers}
    c_i = [c_i[layer] for layer in layers]
    c_i.insert(0, 0)

    plot_data = [c_i, layer_cumul_dist]

    area_under_curve = np.trapz(layer_cumul_dist, x=c_i)

    return plot_data, area_under_curve
def early_exit_experiments(models_path, args, device='cpu'):
    #sdn_training_type = 'sdn_training_bone_only' # IC-only training
    #sdn_training_type = 'sdn_training' # SDN training
    #sdn_training_type = 'sdn_training_bone_only_ic_only'
    if 'attacker1' in args.attack:
        sdn_training_types = ['ic_only','ic_only_bone_only']
        #sdn_training_types = ['sdn_training']
    else:
        sdn_training_types = ['ic_only_bone_only_ic_only']
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

            # change the forward func for sdn to forward with cascade
            top1_test, top5_test, early_exit_counts, non_conf_exit_counts, total_time = mf.vgg_sdn_test_early_exits(sdn_model, one_batch_dataset.test_loader,args.attr_num, device)
            #top1_test, top5_test, early_exit_counts, non_conf_exit_counts, total_time = mf.sdn_test_early_exits(sdn_model, one_batch_dataset.test_loader, device)

            #average_mult_ops = 0
            #total_num_instances = 0

            #for output_id, output_count in enumerate(early_exit_counts):
                #average_mult_ops += output_count*total_ops[output_id]
                #total_num_instances += output_count

            #for output_count in non_conf_exit_counts:
                #total_num_instances += output_count
                #average_mult_ops += output_count*total_ops[output_id]
            
            #average_mult_ops /= total_num_instances

            print('Early exit Counts:')
            print(early_exit_counts)
            sum_exit_counts = [sum(x) for x in zip(*early_exit_counts)]
            sum_attr_counts = [sum(x) for x in early_exit_counts]
            total_ops_all_exits = int(sum(total_ops[i] * sum_exit_counts[i] for i in range(len(sum_exit_counts))))
            print(total_ops_all_exits)
            layer_cumul_dist = convert_num_early_exits_at_each_ic_to_cumulative_dis(sum_exit_counts, sum(sum_exit_counts))
            #plot_data, early_exit_auc = get_plot_data_and_auc(layer_cumul_dist, c_i)
            print(layer_cumul_dist)
            '''if args.attack == 'attacker3' :
                if 'ic_only_bone_only_ic_only' not in sdn_name:
                    pre_thres = [-1]*len(attr_ic)   #这里设定attr_ic是没有-1的，也就是说所有出口都会被微调，没有正常出口
                    for i in range(len(attr_ic)) :
                        if attr_ic[i] == -1 :
                            continue
                        pre_thres[i] = round(100*early_exit_counts[attr_ic[i]][i]/(1e-6+sum_exit_counts[i]),2)
                           
                else:
                    pre_thres = [-1]*len(attr_victim)   
                    for i in range(len(attr_victim)) :
                        if attr_victim[i] == -1 :
                            continue
                        pre_thres[i] = round(100*early_exit_counts[attr_victim [i]][i]/(1e-6+sum_exit_counts[i]),2)  
                recall_thres = [-1]*args.attr_num
            else:
                recall_thres = [-1]*args.attr_num
                pre_thres = [-1]*args.attr_num
                for i in range(len(attr_ic)):
                    if attr_ic[i] == -1:
                        continue
                    pre_thres[attr_ic[i]] = round(100*early_exit_counts[attr_ic[i]][i]/(1e-6+sum_exit_counts[i]),2)
                    recall_thres[attr_ic[i]] = round(100*early_exit_counts[attr_ic[i]][i]/(1e-6+sum_attr_counts[attr_ic[i]]),2) '''
            #print('Non confident exit counts:')
            #print(sum(non_conf_exit_counts))
            '''TP,TPandFP = 0,0
            for i in range(len(attr_victim)):
                if attr_ic[i] == -1 or attr_ic[i] == -2:
                        continue
                TP +=  early_exit_counts[args.target_attr][i]
                TPandFP += sum_exit_counts[i]
            TPandTF = sum_attr_counts[args.target_attr]
            pre_thres = round(100*TP/(TPandFP+1e-5),2)
            recall_thres = round(100*TP/(TPandTF+1e-5),2)'''
            recall_thres = round(100*(early_exit_counts[args.target_attr][-2]/sum_attr_counts[args.target_attr]),2)
            pre_thres = round(100*(early_exit_counts[args.target_attr][-2]/sum_exit_counts[-2]),2)
                
                
                
            print('Top1 Test accuracy: {}'.format(top1_test))
            print('Top5 Test accuracy: {}'.format(top5_test))
            print('SDN cascading took {} seconds.'.format(total_time))
            #print('Average Mult-Ops: {}'.format(average_mult_ops))
            top1_test = round(float(top1_test),2)
            multi_acc.append(top1_multi_test)
            acc.append(top1_test)
            counts.append(early_exit_counts)
            precision.append(pre_thres)
            recall.append(recall_thres)
            ops.append(total_ops_all_exits)
    save_results(args,sdn_training_types,multi_acc,acc,counts,precision,recall,ops)
def main(args):
    torch.manual_seed(af.get_random_seed())    # reproducible
    np.random.seed(af.get_random_seed())
    device = af.get_pytorch_device(args.gpuid)
    trained_models_path = 'models/arcs/attacker1'
    #trained_models_path = 'networks/{}'.format(af.get_random_seed())
    #trained_models_path = 'backdoored_models'

    early_exit_experiments(trained_models_path, args, device)
    #Tsne(trained_models_path, device)
def save_results(args,sdn_names, multi_acc, acc, counts, pre,recall,ops):
    save_path = args.save_path
    record = {}
    # config
    record["config"] = vars(args)
    titles = [f"{x}_{y}" for x in sdn_names for y in args.thres]

    # threshold
    for i, title in enumerate(titles):
        record[f"{title}"] = {
            "multi_acc": multi_acc[i],
            "acc": acc[i],
            "counts": counts[i],
            "pre": pre[i],
            "recall": recall[i],
            'ops':ops[i]
        }

    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            results = json.load(f)
            if not isinstance(results, list):
                raise ValueError("Existing JSON is not a list, cannot append.")
    else:
        results = []
    results.append(record)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    args = get_arguments()
    main(args)