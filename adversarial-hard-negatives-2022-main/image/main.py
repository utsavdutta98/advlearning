import argparse
import os
from tkinter import E
import pandas

import numpy as np
from model import MobileNetWrapper
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


import matplotlib.pyplot as plt

import random

import utils

import cv2

cv2.setNumThreads(0)


from model import Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

def criterion(out_1,out_2, labels_1, labels_2, tau_plus,batch_size,beta, alpha, estimator, fabricate_harder=False, should_print=True, epoch=0, ):
        
        
        # print(out_1.shape, out_2.shape)

        # neg score
        # out = torch.cat([out_1, out_2], dim=0)
        # neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        # old_neg = neg.clone()
        # mask = get_negative_mask(batch_size).to(device)
        # neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # old_neg_masked = old_neg* mask.to(torch.float)

        # # pos score
        # pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # pos = torch.cat([pos, pos], dim=0)
        
        # negative samples similarity scoring
        if estimator=='hard':
            out = torch.cat([out_1, out_2], dim=0)
            neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            old_neg = neg.clone()
            mask = get_negative_mask(batch_size).to(device)
            neg = neg.masked_select(mask).view(2 * batch_size, -1)

            old_neg_masked = old_neg* mask.to(torch.float)

            # pos score
            pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
            pos = torch.cat([pos, pos], dim=0)

            N = batch_size * 2 - 2
            imp = (beta* neg.log()).exp()
            old_imp = (beta* old_neg_masked.log()).exp()

            reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
            reweight_neg_old = (old_imp*old_neg_masked) / torch.unsqueeze(imp.mean(dim = -1), 1)

            Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
            Ng_old = (-tau_plus * pos + reweight_neg_old) / (1 - tau_plus)

            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        elif estimator=='easy':
            Ng = neg.sum(dim=-1)
        elif estimator=='kalantidis':

            # find hardest 31 samples

            out = torch.cat([out_1, out_2], dim=0)
            neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            old_neg = neg.clone()
            mask = get_negative_mask(batch_size).to(device)
            neg = neg.masked_select(mask).view(2 * batch_size, -1)

            old_neg_masked = old_neg* mask.to(torch.float)

            # pos score
            pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
            pos = torch.cat([pos, pos], dim=0)

            pos_full = pos

            N = batch_size * 2 - 2
            imp = (beta* neg.log()).exp()
            old_imp = (beta* old_neg_masked.log()).exp()

            reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
            reweight_neg_old = (old_imp*old_neg_masked) / torch.unsqueeze(imp.mean(dim = -1), 1)

            indices_biggest = np.argsort(reweight_neg_old.masked_select(mask).view(2 * batch_size, -1).cpu().detach().numpy(),axis=1)[:,batch_size - 1:]

            reweight_neg = torch.zeros((2 * batch_size), dtype=torch.float).to(device)
            reweight_neg_old = torch.zeros((2 * batch_size, 2*batch_size), dtype=torch.float).to(device)
            pos = torch.zeros((2 * batch_size), dtype=torch.float).to(device)

            plot_outs = []
            plot_labels = []
            plot_interpolation_labels = []

            labels = torch.cat([labels_1, labels_2], dim=0).cpu().detach().numpy()

            # interpolatio
            for i in range(batch_size * 2):
                indices_this_row = indices_biggest[i]
                indices_this_row_corrected_once = np.where(indices_this_row >= i, indices_this_row + 1, indices_this_row)
                indices_this_row_corrected_twice = np.where(indices_this_row_corrected_once >= batch_size + i, indices_this_row_corrected_once + 1, indices_this_row_corrected_once)
                # print(indices_this_row, indices_this_row_corrected_twice)
                # print(len(indices_this_row_corrected_twice))
                partial_out_other = out[indices_this_row_corrected_twice]
                partial_out_other_interpolated = partial_out_other * (1 - alpha) + out[i].reshape(1,-1).repeat(batch_size - 1, 1) * alpha
                partial_out_other_interpolated = partial_out_other_interpolated / partial_out_other_interpolated.norm(dim=-1, keepdim=True)

                out_other = torch.cat([partial_out_other[:i%batch_size], out[i].reshape(1,-1), partial_out_other[i%batch_size:], partial_out_other_interpolated[:i%batch_size], out[(i+batch_size)%(2*batch_size)].reshape(1,-1), partial_out_other_interpolated[i%batch_size:]], dim=0)

                plot_outs.append(out_other)
                # plot_interpolation_labels.append(torch.cat([torch.zeros(batch_size, dtype=torch.float).to(device), torch.ones(batch_size, dtype=torch.float).to(device)]))
                plot_labels.append(np.concatenate([labels[indices_this_row_corrected_twice][:i%batch_size], labels[i:i+1],labels[indices_this_row_corrected_twice][i%batch_size:], labels[indices_this_row_corrected_twice][:i%batch_size], labels[i:i+1],labels[indices_this_row_corrected_twice][i%batch_size:]], axis=0))


                neg = torch.exp(torch.mm(out[i:i+1], out_other.t().contiguous()) / temperature)
                old_neg = neg.clone()

                neg = neg.masked_select(mask[i:i+1]).view(1, -1)

                old_neg_masked = old_neg* mask[i:i+1].to(torch.float)

                # pos score
                pos[i] = torch.exp(torch.sum(out_1[i%batch_size:i%batch_size+1] * out_2[i%batch_size:i%batch_size+1], dim=-1) / temperature)

                imp = (beta* neg.log()).exp()
                old_imp = (beta* old_neg_masked.log()).exp()

                reweight_neg[i:i+1] = (imp *neg ).sum(dim = 1) / imp.mean(dim = -1)
                reweight_neg_old[i:i+1] = (old_imp*old_neg_masked) / torch.unsqueeze(imp.mean(dim = 1), 1)

            N = batch_size * 2 - 2
            # print(pos_full.shape, reweight_neg.shape)
            Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
            Ng_old = (-tau_plus * pos.reshape(2*batch_size,1).repeat(1,2*batch_size) + reweight_neg_old) / (1 - tau_plus)

            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')
            

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng) )).mean()

        # print(loss)

        labels = torch.cat([labels_1, labels_2], dim=0).cpu().detach().numpy()
        print(labels.shape)
        if should_print:
            plt.clf()
            if random.uniform(0,1) < 0.01:
                if estimator == 'kalantidis':

                    vis_int = random.randint(0,batch_size-1)
                    plt.clf()
                    indices = list(range(2*batch_size))
                    indices.pop(vis_int+batch_size)
                    indices.pop(vis_int)

                    neg_selected = Ng_old[vis_int][indices]

                    indices_0 = []
                    indices_1 = []
                    for i in indices:
                        if plot_labels[vis_int][i] == 0:
                            indices_0.append(i)
                        else:
                            indices_1.append(i)
                    # indices_0 = indices[labels[indices] == 0]
                    # indices_1 = indices[labels[indices] == 1]
                    # for label, color in zip([0,1], ['red','blue']):
                    plt.scatter(plot_outs[vis_int][indices,0].cpu().detach().numpy(), plot_outs[vis_int][indices,1].cpu().detach().numpy(), cmap='viridis', c=neg_selected.cpu().detach().numpy(), )
                    # plt.scatter(plot_outs[vis_int][indices[2*batch_size-2:],0].cpu().detach().numpy(), plot_outs[vis_int][indices[2*batch_size-2:],1].cpu().detach().numpy(), edgecolors='teal')
                    
                    print(plot_labels[vis_int][vis_int])

                    if plot_labels[vis_int][vis_int] == 0:
                        plt.scatter(torch.nn.functional.normalize(plot_outs[vis_int][vis_int], dim=0)[0].cpu().detach().numpy() , torch.nn.functional.normalize(plot_outs[vis_int][vis_int], dim=0)[1].cpu().detach().numpy(), color='red', marker='o')
                        plt.scatter(torch.nn.functional.normalize(plot_outs[vis_int][vis_int+batch_size], dim=0)[0].cpu().detach().numpy(), torch.nn.functional.normalize(plot_outs[vis_int][vis_int+batch_size], dim=0)[1].cpu().detach().numpy(), color='red', marker='o')
                    else:
                        plt.scatter(torch.nn.functional.normalize(plot_outs[vis_int][vis_int], dim=0)[0].cpu().detach().numpy() , torch.nn.functional.normalize(plot_outs[vis_int][vis_int], dim=0)[1].cpu().detach().numpy(), color='blue', marker='o')
                        plt.scatter(torch.nn.functional.normalize(plot_outs[vis_int][vis_int+batch_size], dim=0)[0].cpu().detach().numpy(), torch.nn.functional.normalize(plot_outs[vis_int][vis_int+batch_size], dim=0)[1].cpu().detach().numpy(), color='blue', marker='o')
                    # indices_1 = 
                    # plt.scatter(out[indices,0].cpu().detach().numpy(), out[indices,1].cpu().detach().numpy(), facecolors='none', edgecolors=)
                    # plt.scatter(out[vis_int,], y, s=80, facecolors='none', edgecolors='r')
                    
                    plt.title(f'MNIST, 0-1, Beta={beta}, epoch={epoch}, positive sample & scaled negative samples')
                    # plt.legend()
                    plt.savefig(f'{beta}_{epoch}_{estimator}_colors.png')

                    plt.clf()
                    # indices_0 = indices[labels[indices] == 0]
                    # indices_1 = indices[labels[indices] == 1]
                    plt.scatter(plot_outs[vis_int][indices_0,0].cpu().detach().numpy(), plot_outs[vis_int][indices_0,1].cpu().detach().numpy(), cmap='viridis', color='red')
                    plt.scatter(plot_outs[vis_int][indices_1,0].cpu().detach().numpy(), plot_outs[vis_int][indices_1,1].cpu().detach().numpy(), cmap='viridis', color='blue')
                    plt.title(f'MNIST, 0-1, Beta={beta}, epoch={epoch}, class of negative samples')
                    plt.savefig(f'{beta}_{epoch}_{estimator}_classes.png')

                else:
                    vis_int = random.randint(0,batch_size-1)
                    plt.clf()
                    indices = list(range(2*batch_size))
                    indices.pop(vis_int+batch_size)
                    indices.pop(vis_int)

                    neg_selected = Ng_old[vis_int][indices]

                    indices_0 = []
                    indices_1 = []
                    for i in indices:
                        if labels[i] == 0:
                            indices_0.append(i)
                        else:
                            indices_1.append(i)
                    # indices_0 = indices[labels[indices] == 0]
                    # indices_1 = indices[labels[indices] == 1]
                    # for label, color in zip([0,1], ['red','blue']):
                    plt.scatter(out[indices,0].cpu().detach().numpy(), out[indices,1].cpu().detach().numpy(), cmap='viridis', c=neg_selected.cpu().detach().numpy(), )
                    plt.scatter(out[indices[2*batch_size-2:],0].cpu().detach().numpy(), out[indices[2*batch_size-2:],1].cpu().detach().numpy(), edgecolors='teal')
                    
                    
                    if labels[vis_int] == 0:
                        plt.scatter(torch.nn.functional.normalize(out[vis_int], dim=0)[0].cpu().detach().numpy() , torch.nn.functional.normalize(out[vis_int], dim=0)[1].cpu().detach().numpy(), c='red', marker='o')
                        plt.scatter(torch.nn.functional.normalize(out[vis_int+batch_size], dim=0)[0].cpu().detach().numpy(), torch.nn.functional.normalize(out[vis_int+batch_size], dim=0)[1].cpu().detach().numpy(), c='red', marker='o')
                    else:
                        plt.scatter(torch.nn.functional.normalize(out[vis_int], dim=0)[0].cpu().detach().numpy() , torch.nn.functional.normalize(out[vis_int], dim=0)[1].cpu().detach().numpy(), c='blue', marker='o')
                        plt.scatter(torch.nn.functional.normalize(out[vis_int+batch_size], dim=0)[0].cpu().detach().numpy(), torch.nn.functional.normalize(out[vis_int+batch_size], dim=0)[1].cpu().detach().numpy(), c='blue', marker='o')
                    # indices_1 = 
                    # plt.scatter(out[indices,0].cpu().detach().numpy(), out[indices,1].cpu().detach().numpy(), facecolors='none', edgecolors=)
                    # plt.scatter(out[vis_int,], y, s=80, facecolors='none', edgecolors='r')
                    
                    plt.title(f'MNIST, 0-1, Beta={beta}, epoch={epoch}, positive sample & scaled negative samples')
                    # plt.legend()
                    plt.savefig(f'{beta}_{epoch}_{estimator}_colors.png')

                    plt.clf()
                    # indices_0 = indices[labels[indices] == 0]
                    # indices_1 = indices[labels[indices] == 1]
                    plt.scatter(out[indices_0,0].cpu().detach().numpy(), out[indices_0,1].cpu().detach().numpy(), cmap='viridis', color='red')
                    plt.scatter(out[indices_1,0].cpu().detach().numpy(), out[indices_1,1].cpu().detach().numpy(), cmap='viridis', color='blue')
                    plt.title(f'MNIST, 0-1, Beta={beta}, epoch={epoch}, class of negative samples')
                    plt.savefig(f'{beta}_{epoch}_{estimator}_classes.png')

        return loss

def train(net, data_loader, train_optimizer, temperature, alpha, estimator, tau_plus, beta, epoch):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

    for pos_1, pos_2, target in train_bar:


        pos_1, pos_2 = pos_1.to(device,non_blocking=True), pos_2.to(device,non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        loss = criterion(out_1, out_2, target, target, tau_plus, batch_size, beta, alpha, estimator, epoch=epoch)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.to(device, non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        if 'cifar' in dataset_name or 'mnist' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device) 
        elif 'stl' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device) 

        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:,:1] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:,:5] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('KNN Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--root', type=str, default='../data', help='Path to data directory')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--tau_plus', default=0.1, type=float, help='Positive class priorx')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=400, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--estimator', default='hard', type=str, help='Choose loss function')
    parser.add_argument('--dataset_name', default='stl10', type=str, help='Choose loss function')
    parser.add_argument('--beta', default=1.0, type=float, help='Choose loss function')
    parser.add_argument('--anneal', default=None, type=str, help='Beta annealing')
    parser.add_argument('--start_from', default=None, type=str, help='Start from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='Start from checkpoint')
    parser.add_argument('--alpha', default=0.1, type=float, help='Alpha for the loss function')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, tau_plus, k = args.feature_dim, args.temperature, args.tau_plus, args.k
    batch_size, epochs, estimator = args.batch_size, args.epochs,  args.estimator
    alpha = args.alpha
    dataset_name = args.dataset_name
    beta = args.beta
    anneal = args.anneal

    #configuring an adaptive beta if using annealing method
    if anneal=='down':
        do_beta_anneal=True
        n_steps=9
        betas=iter(np.linspace(beta,0,n_steps))
    else:
        do_beta_anneal=False
    
    # data prepare
    train_data, memory_data, test_data = utils.get_dataset(dataset_name, root=args.root)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True, drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim).to(device)
    # model = MobileNetWrapper(feature_dim).to(device)
    model = nn.DataParallel(model)
    if args.start_from is not None:
        model.load_state_dict(torch.load(args.start_from, map_location='cuda:0'))
        start_epoch = args.start_epoch
    else:
        start_epoch = 1

    # optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)
    c = len(memory_data.classes)
    print('# Classes: {}'.format(c))

    # training loop
    try:
        os.makedirs('../results/{}'.format(dataset_name))
    except:
        pass

    for epoch in range(start_epoch, epochs + start_epoch):
        train_loss = train(model, train_loader, optimizer, temperature, alpha, estimator, tau_plus, beta, epoch)
        
        if do_beta_anneal is True:
            if epoch % (int(epochs/n_steps)) == 0:
                beta=next(betas)

        if epoch % 5 == 0:
            test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
            if do_beta_anneal is True:
                torch.save(model.state_dict(), '../results/{}/{}_{}_model_{}_{}_{}_{}_{}.pth'.format(dataset_name,dataset_name,alpha,estimator,batch_size,tau_plus,beta,epoch,anneal))
            else:
                torch.save(model.state_dict(), '../results/{}/{}_{}_model_{}_{}_{}_{}.pth'.format(dataset_name,dataset_name,alpha,estimator,batch_size,tau_plus,beta ,epoch))
