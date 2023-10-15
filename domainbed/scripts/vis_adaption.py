# The code is modified from domainbed.scripts.train

import argparse
from argparse import Namespace
import collections
import json
import os
import random
import sys
import time
import uuid
from itertools import chain
import itertools
import copy

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader, DataParallelPassthrough
from domainbed import model_selection
from domainbed.lib.query import Q
from domainbed import adapt_algorithms
import itertools

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise   
class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

def generate_featurelized_loader(loader, network, classifier, batch_size=32):
    """
    The classifier adaptation does not need to repeat the heavy forward path, 
    We speeded up the experiments by converting the observations into representations. 
    """
    z_list = []
    y_list = []
    p_list = []
    network.eval()
    classifier.eval()
    for x, y in loader:
        x = x.to(device)
        z = network(x)
        p = classifier(z)
        
        z_list.append(z.detach().cpu())
        y_list.append(y.detach().cpu())
        p_list.append(p.detach().cpu())
        # p_list.append(p.argmax(1).float().cpu().detach())
    network.train()
    classifier.train()
    z = torch.cat(z_list)
    y = torch.cat(y_list)
    p = torch.cat(p_list)
    ent = softmax_entropy(p)
    py = p.argmax(1).float().cpu().detach()
    dataset1, dataset2 = Dataset(z, y), Dataset(z, py)
    loader1 = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=False, drop_last=True)
    loader2 = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=False, drop_last=True)
    return loader1, loader2, ent


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def accuracy_ent(network, loader, weights, device, adapt=False):
    correct = 0
    total = 0
    weights_offset = 0
    ent = 0
    
    if adapt == False:
        network.eval()
    #with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if adapt is None:
            p = network(x)
        else:
            p = network(x, adapt)
        if weights is None:
            batch_weights = torch.ones(len(p)) # x
        else:
            batch_weights = weights[weights_offset: weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)
        if len(p) != len(x):
            y = torch.cat((y,y))
        if p.size(1) == 1:
            correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
        else:
            correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
        total += batch_weights.sum().item()
        ent += softmax_entropy(p).sum().item()
    if adapt == False:
        network.train()

    return correct / total, ent / total


def accuracy_ent_interval(network, loader, weights, device, args=None, adapt_hparams= None, run_id = 0, base_acc = None, reset
                        = False, index = None, adapt = False, save_path = None):
    correct = 0
    total = 0
    weights_offset = 0
    ent = 0

    interval = 50
    num_samples = 0 # for count
    interval_correct = 0 
    interval_total = 0
    all_accs = []
    all_accs_accu = []

    if adapt == False:
        network.eval()
    #with torch.no_grad():
    for x, y in loader:
        num_samples += len(x)
        if num_samples >= interval:
            
            all_accs.append(interval_correct/interval_total*100.0)
            all_accs_accu.append(correct / total*100.0)
            num_samples, interval_correct, interval_total = 0, 0, 0
            if reset:
                network.reset()
            
        x = x.to(device)
        y = y.to(device)
        if adapt is None:
            p = network(x)
        else:
            p = network(x, adapt)
        if weights is None:
            batch_weights = torch.ones(len(p)) # x
        else:
            batch_weights = weights[weights_offset: weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)
        if len(p) != len(x):
            y = torch.cat((y,y))
        if p.size(1) == 1:
            correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            interval_correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
        else:
            correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            interval_correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
        total += batch_weights.sum().item()
        interval_total += batch_weights.sum().item()

        ent += softmax_entropy(p).sum().item()

    if adapt == False:
        network.train()

    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(all_accs))
    
    # ax = plt.axes()
    # ax.grid()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # plt.ylim(80, 100)
    # ax.plot(x, all_accs, label='Interval')
    # ax.plot(x, all_accs_accu, label='Accumulation')
    # # plt.axhline(base_acc)
    # plt.xlabel('Number of samples (x100)')
    # plt.ylabel('Acc.')
    # plt.legend(loc = 'lower right')

    # # save data
    # all_accs = np.array(all_accs)
    # all_accs_accu = np.array(all_accs_accu)
    # import seaborn as sns
    # # if reset:
    # #     np.save(f'accuracy_data/all_acc_reset_{args.adapt_algorithm}_{args.dataset}_run{run_id}_{args.test_envs[0]}.txt', all_accs)
    # #     np.save(f'accuracy_data/all_accs_accu_reset_{args.adapt_algorithm}_{args.dataset}_run{run_id}_{args.test_envs[0]}.txt', all_accs_accu)
    # #     plt.savefig(f'accuracy_curves/all_acc_reset_{args.adapt_algorithm}_{args.dataset}_run{run_id}_{args.test_envs[0]}.pdf')
    # # elif index:
    # #     np.save(f'accuracy_data/all_acc_index_{args.adapt_algorithm}_{args.dataset}_run{run_id}_{args.test_envs[0]}.txt', all_accs)
    # #     np.save(f'accuracy_data/all_accs_accu_index_{args.adapt_algorithm}_{args.dataset}_run{run_id}_{args.test_envs[0]}.txt', all_accs_accu)
    # #     plt.savefig(f'accuracy_curves/all_acc_index_{args.adapt_algorithm}_{args.dataset}_run{run_id}_{args.test_envs[0]}.pdf')
    # # else:
    # mkdir_if_missing(save_path)
    # if base_acc is not None:
    #     np.save(f'{save_path}/adapt_acc', all_accs)
    #     np.save(f'{save_path}/adapt_accu', all_accs_accu)
    #     # plt.savefig(f'{save_path}/all_acc.pdf')
    # else:
    #     np.save(f'{save_path}/base_acc', all_accs)
    #     np.save(f'{save_path}/base_accu', all_accs_accu)
        # plt.savefig(f'{save_path}/all_acc_base_model_.pdf')
    # plt.cla()

    return correct / total, ent / total, all_accs,all_accs_accu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--adapt_algorithm', type=str, default="UniDG")
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    args_in = parser.parse_args()
    base_intervals = []
    base_accumus = []
    adapt_intervals = []
    adapt_accumus = []
    for i in range(10):

        epochs_path = os.path.join(args_in.input_dir, 'results.jsonl')
        records = []
        with open(epochs_path, 'r') as f:
            for line in f:
                records.append(json.loads(line[:-1]))
        records = Q(records)
        r = records[0]
        args = Namespace(**r['args'])
        args.input_dir = args_in.input_dir

        if '-' in args_in.adapt_algorithm:
            args.adapt_algorithm, test_batch_size = args_in.adapt_algorithm.split('-')
            args.test_batch_size = int(test_batch_size)
        else:
            args.adapt_algorithm = args_in.adapt_algorithm
            args.test_batch_size = 32  # default

        base_algo, adapt_dataset, adapt_test_env = args_in.input_dir.split('/')[2], args_in.input_dir.split('/')[-2], args_in.input_dir.split('/')[-1]
        backbone = eval(args.hparams)['backbone']
        args.output_dir = args.input_dir
        save_path = os.path.join('./vis_curves',base_algo,backbone,adapt_dataset,adapt_test_env)
        
        alg_name = args_in.adapt_algorithm

        if args.adapt_algorithm in['T3A', 'TentPreBN', 'TentClf', 'PLClf']:
            use_featurer_cache = True
        else:
            use_featurer_cache = False
        if os.path.exists(os.path.join(args.output_dir, 'done_{}'.format(alg_name))):
            print("{} has already excecuted".format(alg_name))

        # If we ever want to implement checkpointing, just persist these values
        # every once in a while, and then load them from disk here.
        algorithm_dict = None
        # os.makedirs(args.output_dir, exist_ok=True)
        sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out_{}.txt'.format(alg_name)))
        sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err_{}.txt'.format(alg_name)))

        print("Environment:")
        print("\tPython: {}".format(sys.version.split(" ")[0]))
        print("\tPyTorch: {}".format(torch.__version__))
        print("\tTorchvision: {}".format(torchvision.__version__))
        print("\tCUDA: {}".format(torch.version.cuda))
        print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
        print("\tNumPy: {}".format(np.__version__))
        print("\tPIL: {}".format(PIL.__version__))
        args.trial_seed = i
        print('Args:')
        for k, v in sorted(vars(args).items()):
            print('\t{}: {}'.format(k, v))

        if args.hparams_seed == 0:
            hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
        else:
            hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                misc.seed_hash(args.hparams_seed, args.trial_seed))
        if args.hparams:
            hparams.update(json.loads(args.hparams))

        print('HParams:')
        for k, v in sorted(hparams.items()):
            print('\t{}: {}'.format(k, v))

        assert os.path.exists(os.path.join(args.output_dir, 'done'))
        assert os.path.exists(os.path.join(args.output_dir, 'IID_best.pkl'))  # IID_best is produced by train.py

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        if args.dataset in vars(datasets):
            dataset = vars(datasets)[args.dataset](args.data_dir,
                args.test_envs, hparams)
        else:
            raise NotImplementedError

        ## added for ada_contrast
        hparams['ada_contrast'] = True
        if args.adapt_algorithm == 'AdaContrast':
            dataset_ada_contrast = vars(datasets)[args.dataset](args.data_dir,
                args.test_envs, hparams)

        # Split each env into an 'in-split' and an 'out-split'. We'll train on
        # each in-split except the test envs, and evaluate on all splits.
        
        # To allow unsupervised domain adaptation experiments, we split each test
        # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
        # by collect_results.py to compute classification accuracies.  The
        # 'out-split' is used by the Oracle model selectino method. The unlabeled
        # samples in 'uda-split' are passed to the algorithm at training time if
        # args.task == "domain_adaptation". If we are interested in comparing
        # domain generalization and domain adaptation results, then domain
        # generalization algorithms should create the same 'uda-splits', which will
        # be discared at training.
        in_splits = []
        out_splits = []
        uda_splits = []
        for env_i, env in enumerate(dataset):
            uda = []
            out, in_ = misc.split_dataset(env,
                int(len(env)*args.holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

            if env_i in args.test_envs:
                uda, in_ = misc.split_dataset(in_,
                    int(len(in_)*args.uda_holdout_fraction),
                    misc.seed_hash(args.trial_seed, env_i))

            if hparams['class_balanced']:
                in_weights = misc.make_weights_for_balanced_classes(in_)
                out_weights = misc.make_weights_for_balanced_classes(out)
                if uda is not None:
                    uda_weights = misc.make_weights_for_balanced_classes(uda)
            else:
                in_weights, out_weights, uda_weights = None, None, None
            in_splits.append((in_, in_weights))
            out_splits.append((out, out_weights))
            if len(uda):
                uda_splits.append((uda, uda_weights))

        ## add for ada_contrast
        if args.adapt_algorithm == 'AdaContrast':
            in_splits_ada_contrast = []
            for env_i, env in enumerate(dataset_ada_contrast):
                
                _, in_contrast = misc.split_dataset(env,
                    int(len(env)*args.holdout_fraction),
                    misc.seed_hash(args.trial_seed, env_i))

                if env_i in args.test_envs:
                    uda, in_ = misc.split_dataset(in_,
                        int(len(in_)*args.uda_holdout_fraction),
                        misc.seed_hash(args.trial_seed, env_i))
            

                in_splits_ada_contrast.append((in_contrast, None))
            

        # Use out splits as training data (to fair comparison with train.py)
        train_loaders = [FastDataLoader(
            dataset=env,
            batch_size=hparams['batch_size'],
            num_workers=dataset.N_WORKERS)
            for i, (env, env_weights) in enumerate(out_splits)
            if i in args.test_envs]

        ## add for ada_contrast
        if args.adapt_algorithm == 'AdaContrast':
            train_loader_ada_contrast = [FastDataLoader(
                dataset=env,
                batch_size=hparams['batch_size'],
                num_workers=dataset.N_WORKERS)
                for i, (env, env_weights) in enumerate(in_splits_ada_contrast)
                if i in args.test_envs]

        uda_loaders = [InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=hparams['batch_size'],
            num_workers=dataset.N_WORKERS)
            for i, (env, env_weights) in enumerate(uda_splits)
            if i in args.test_envs]

        if args_in.shuffle:
            eval_loaders = [torch.utils.data.DataLoader(
            dataset=env,
            batch_size=args.test_batch_size,
            shuffle=True,
            num_workers=dataset.N_WORKERS)
            for env, _ in (in_splits + out_splits + uda_splits)]
        else:
            eval_loaders = [FastDataLoader(
                dataset=env,
                batch_size=args.test_batch_size,
                num_workers=dataset.N_WORKERS)
                for env, _ in (in_splits + out_splits + uda_splits)]
        eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
        eval_loader_names = ['env{}_in'.format(i)
            for i in range(len(in_splits))]
        eval_loader_names += ['env{}_out'.format(i)
            for i in range(len(out_splits))]
        eval_loader_names += ['env{}_uda'.format(i)
            for i in range(len(uda_splits))]

        algorithm_class = algorithms.get_algorithm_class(args.algorithm)
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), hparams)

        if algorithm_dict is not None:
            algorithm.load_state_dict(algorithm_dict)

        algorithm.to(device)
        if hasattr(algorithm, 'network'):
            algorithm.network = DataParallelPassthrough(algorithm.network)
        else:
            for m in algorithm.children():
                m = DataParallelPassthrough(m)

        train_minibatches_iterator = zip(*train_loaders)
        uda_minibatches_iterator = zip(*uda_loaders)
        checkpoint_vals = collections.defaultdict(lambda: [])

        # load trained model
        ckpt = torch.load(os.path.join(args.output_dir, 'IID_best.pkl'))
        algorithm_dict = ckpt['model_dict']
        if algorithm_dict is not None:
            algorithm.load_state_dict(algorithm_dict)

        # Evaluate base model

        print("Base model's results")
        results = {}
        evals = zip(eval_loader_names, eval_loaders, eval_weights)

        for name, loader, weights in evals:
            if name in ['env{}_in'.format(i) for i in args.test_envs]:
                acc, ent, base_acc, base_accu = accuracy_ent_interval(algorithm, loader, weights, device, adapt=None,save_path=save_path)
                results[name+'_acc'] = acc
                results[name+'_ent'] = ent
        results_keys = sorted(results.keys())
        base_intervals.append(base_acc)
        base_accumus.append(base_accu)
        misc.print_row(results_keys, colwidth=12)
        misc.print_row([results[key] for key in results_keys], colwidth=12)

    

        print("\nAfter {}".format(alg_name))
        # Cache the inference results
        if use_featurer_cache:
            original_evals = zip(eval_loader_names, eval_loaders, eval_weights)
            loaders = []
            for name, loader, weights in original_evals:
                loader1, loader2, ent = generate_featurelized_loader(loader, network=algorithm.featurizer, classifier=algorithm.classifier, batch_size=32)
                loaders.append((name, loader1, weights))
        else:
            loaders = zip(eval_loader_names, eval_loaders, eval_weights)
        
        evals = []
        for name, loader, weights in loaders:
            if name in ['env{}_in'.format(i) for i in args.test_envs]:
                train_loader = (name, loader, weights)
            else:
                evals.append((name, loader, weights))

        last_results_keys = None
        adapt_algorithm_class = adapt_algorithms.get_algorithm_class(
            args.adapt_algorithm)
        
        if args.adapt_algorithm in ['T3A', 'T3A_Aug']:
            adapt_hparams_dict = {
                'filter_K': [1, 5, 20, 50, 100, -1], 
            }
            if args.dataset == 'DomainNet':
                adapt_hparams_dict = {
                    'filter_K': [100], 
                }
            
        elif args.adapt_algorithm in ['TentFull', 'TentPreBN', 'TentClf', 'TentNorm']:
            adapt_hparams_dict = {
                'alpha': [0.1, 1.0, 10.0],
                'gamma': [1, 3]
            }
        elif args.adapt_algorithm in ['PseudoLabel', 'PLClf']:
            adapt_hparams_dict = {
                'alpha': [0.1, 1.0, 10.0],
                'gamma': [1, 3], 
                'beta': [0.9]
            }
        elif args.adapt_algorithm in ['SHOT', 'SHOTIM']:
            adapt_hparams_dict = {
                'alpha': [0.1, 1.0, 10.0],
                'gamma': [1, 3], 
                'beta': [0.9], 
                'theta': [0.1], 
            }
        elif args.adapt_algorithm in ['UniDG']:
            adapt_hparams_dict = {
                # 'alpha': [0.05, 0.1],
                # # 'gamma': [1,3],
                # 'gamma': [1],
                # 'lamb': [1.0, 0.1],
                # 'filter_K': [1, 5, 20, 50, 100,-1], 
                'alpha': [0.1],
                # 'alpha': [0.1],
                'gamma': [1],
                # 'gamma': [3],
                'lamb': [1.0, ],
                # 'lamb': [0.1],
                'filter_K': [100,], 
                # 'filter_K': [50], 

            }
        else:
            raise Exception("Not Implemented Error")
        product = [x for x in itertools.product(*adapt_hparams_dict.values())]
        adapt_hparams_list = [dict(zip(adapt_hparams_dict.keys(), r)) for r in product]

        run_id = 0 # for drawing curves
        index = -1
        for adapt_hparams in adapt_hparams_list:
            if args_in.shuffle:
                index += 1
                if index <2:
                    continue 
                if index == 2:
                    fix_hparams = adapt_hparams
                adapt_hparams = fix_hparams

                if index > 10:
                    break
            
            adapt_hparams['cached_loader'] = use_featurer_cache
            adapted_algorithm = adapt_algorithm_class(dataset.input_shape, dataset.num_classes,
                len(dataset) - len(args.test_envs), adapt_hparams, algorithm
            )
            # adapted_algorithm = DataParallelPassthrough(adapted_algorithm)
            adapted_algorithm.to(device)
            
            results = adapt_hparams

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            ## Usual evaluation
        
            # for name, loader, weights in evals:
            #     acc, ent = accuracy_ent(adapted_algorithm, loader, weights, device, adapt=True)
            #     results[name+'_acc'] = acc
            #     results[name+'_ent'] = ent
            #     adapted_algorithm.reset()
            
            name, loader, weights = train_loader
            ## add for ada_contrast
            if args.adapt_algorithm == 'AdaContrast':
                acc, ent = accuracy_ent(adapted_algorithm, train_loader_ada_contrast, weights, device, adapt=True)
            else:
                base_acc, _ = accuracy_ent(algorithm, loader, weights, device, adapt=None)
                if args_in.shuffle:
                    acc, ent,adapt_acc, adapt_accmu = accuracy_ent_interval(adapted_algorithm, loader, weights, device, args, adapt_hparams, run_id, base_acc, reset=args_in.reset, index=index, adapt=True,save_path=save_path)
                else:
                    acc, ent,adapt_acc, adapt_accmu = accuracy_ent_interval(adapted_algorithm, loader, weights, device, args, adapt_hparams, run_id, base_acc, reset=args_in.reset, adapt=True, save_path=save_path)
            results[name+'_acc'] = acc
            results[name+'_ent'] = ent
            run_id += 1
            adapt_intervals.append(adapt_acc)
            adapt_accumus.append(adapt_accmu)
            del adapt_hparams['cached_loader']
            results_keys = sorted(results.keys())

            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)    
            })
        # save file
        # epochs_path = os.path.join(args.output_dir, 'results_{}.jsonl'.format(alg_name))
        # with open(epochs_path, 'a') as f:
        #     f.write(json.dumps(results, sort_keys=True) + "\n")

    # create done file
    # with open(os.path.join(args.output_dir, 'done_{}'.format(alg_name)), 'w') as f:
    #     f.write('done')
    # base_intervals = np.array(base_intervals)

    import re
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import shutil
    import os
    sns.set_style('whitegrid')
    # color = cm.viridis(0.7)
    color = "#00C5F2"
    f, ax = plt.subplots(1,1)
    # base_intervals = np.array(base_intervals)
    base_accumus = np.array(base_accumus)
    adapt_accumus = np.array(adapt_accumus)
    x = np.arange(base_accumus.shape[1])
    # adapt_intervals = np.array(adapt_intervals)
    # ax.plot(, np.mean(np.array(base_intervals),axis=0), color=color,label = 'interval')
    ax.plot(x , np.mean(base_accumus,axis=0), color=color,label = 'Base')
    r1 = list(map(lambda x: x[0]-x[1], zip(np.mean(base_accumus,axis=0), np.std(base_accumus,axis=0))))
    r2 = list(map(lambda x: x[0]+x[1], zip(np.mean(base_accumus,axis=0), np.std(base_accumus,axis=0))))

    ax.fill_between(x, r1, r2, color=color, alpha=0.2)
    ax.legend(loc='lower right')
    ax.set_xlabel('Samples (N x 50)')
    ax.set_ylabel('Acc.')
    

    # color = cm.viridis(0.9)
    # color = "#FF7285"

    ax.plot(x , np.mean(adapt_accumus,axis=0), color="#FF7285",label = 'UniDG')
    a1 = list(map(lambda x: x[0]-x[1], zip(np.mean(adapt_accumus,axis=0), np.std(adapt_accumus,axis=0))))
    a2 = list(map(lambda x: x[0]+x[1], zip(np.mean(adapt_accumus,axis=0), np.std(adapt_accumus,axis=0))))
    ax.fill_between(x, a1, a2, color="#FF7285", alpha=0.2)
    ax.set_xlim(0,x.shape[0]-1)
    ax.legend(loc='lower right')
    ax.set_xlabel('Samples (N x 50)')
    ax.set_ylabel('Acc.')
    # exp_dir = 'Plot/'
    # if not os.path.exists(exp_dir):
    #     os.makedirs(exp_dir, exist_ok=True)
    mkdir_if_missing(save_path)
    f.savefig(f'{save_path}/vis_acc_results.pdf', dpi=50)