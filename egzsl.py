import argparse
import os
import random
import torch
import torch.optim as optim
import util
import model
import copy
import numpy as np
import torch.nn.functional as F
import time
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='data', help='path to dataset')
    parser.add_argument('--matdataset', default=True, help='Data in matlab format')
    parser.add_argument('--image_embedding', default='res101')
    parser.add_argument('--class_embedding', default='att')
    parser.add_argument('--gzsl', action='store_true', default=True, help='enable generalized zero-shot learning')
    parser.add_argument('--preprocessing', action='store_true', default=True,
                        help='enbale MinMaxScaler on visual features')
    parser.add_argument('--standardization', action='store_true', default=False)
    parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
    parser.add_argument('--device',type=str, default='cuda', help='enables cuda')
    parser.add_argument('--netC', default=None, help="path to netC (to continue training)")
    parser.add_argument('--proto_layer_sizes', nargs='+' ,type=int, default=[1024, 2048], help='size of the hidden and output units in prototype learner')
    parser.add_argument('--attSize', type=int, default=85, help='size of semantic features')
    parser.add_argument('--dataset', default='AWA2', help='set dataset')
    parser.add_argument('--nShot', default=1000, type=int, help='number of samples per evolutionary task')
    parser.add_argument('--BatchSize', '-b', default=32, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--num_worker', '-j', default=4, type=int, metavar='N', help='number of data loading threads')
    parser.add_argument('--nIterate', default=1, type=int, metavar='N', help='epochs for training process')
    parser.add_argument('--tem', type=float, default=0.04, help='temprature')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.99, help='beta1 for adam. default=0.5')
    parser.add_argument('--tau', type=float, default=0.9, help='threshold for data selection')
    parser.add_argument('--m1', type=float, default=0.9, help='smoothing factor for ema model')
    parser.add_argument('--m2', type=float, default=0.9, help='smoothing factor for data selection')
    parser.add_argument('--weight_kl', type=float, default=1, help='loss weight of KL divergence')
    parser.add_argument('--work_dir', type=str, default='./work_dir', help='work dir to save results')
    parser.add_argument('--base_method', type=str, default='ZLA', help='GZSL method in base training phase')
    parser.add_argument('--num_experiment', type=int,default=5, help='for multi random test')

    args = parser.parse_args()
    return args

def setup_logger(log_file='training.log'):
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

def val_task(test_res,netC,data,opt):
    ds_test=[test_res[i] for i in range(test_res.size(0))]
    test_loader=torch.utils.data.DataLoader(ds_test, batch_size=512, shuffle=False,num_workers=opt.num_worker)
    predict=torch.Tensor([]).to(opt.device)
    for i,batch_res in enumerate(test_loader,0):

        batch_res=batch_res.to(opt.device)

        batch_res=F.normalize(batch_res,dim=-1)
        proto=netC(data.attribute.to(opt.device))
        proto=F.normalize(proto,dim=-1)
        logits=batch_res@proto.t()
        batch_predict=logits.max(1)[1]
        predict=torch.cat((predict,batch_predict),dim=0)

    return predict

def cal_prop(res,netC,data,opt):
    res=F.normalize(res,dim=-1)
    proto=netC(data.attribute.to(opt.device))
    proto=F.normalize(proto,dim=-1)
    logits=res@proto.t()/opt.tem
    prop=F.softmax(logits,dim=-1)
    return prop

def cal_mean_class_acc(classes,real_label,true_or_false):
    acc = np.array([])
    for i in range(classes.size(0)):
        iclass=classes[i].to(true_or_false.device)
        idx=(real_label.to(true_or_false.device)==iclass).nonzero().squeeze(1)
        iacc=true_or_false[idx].mean()
        acc=np.append(acc,iacc.cpu().item())
    # acc=acc.mean()
    acc=np.nanmean(acc)
    return acc

def cal_H(pre_label,real_label,data):
    true_or_false=(pre_label==real_label.to(pre_label.device)).float()
    acs=cal_mean_class_acc(data.seenclasses,real_label,true_or_false)
    acu=cal_mean_class_acc(data.unseenclasses,real_label,true_or_false)
    ach=2*acs*acu/(acs+acu)
    return acs,acu,ach

def cal_mean_acc(pre_label,real_label):
    true_or_false = (pre_label == real_label.to(pre_label.device)).float()
    return true_or_false.mean()

def train_task(ema_prop,opt, ds_res,data, netC,optimizer,task_id,num_task,netC_ema=None,logger=None):

    for t in range(opt.nIterate):
        with torch.no_grad():
            # predict psuedo label
            ds_res=ds_res.to(opt.device)
            proto=netC(data.attribute.to(opt.device))
            proto=F.normalize(proto,dim=-1)
            res=F.normalize(ds_res,dim=-1)
            logits=res@proto.t()/opt.tem
            prop=F.softmax(logits,dim=-1)
            prop_max,pseudo_label=prop.max(1)
 
            # data selection
            select_res = []
            select_label = []
            select_prop=[]
            decay=opt.m2
            for i in range(data.allclasses.size(0)):
                iclass = data.allclasses[i]
                idx = (pseudo_label == iclass).nonzero().squeeze(1)
                iprop = prop_max[idx]
                
                # update \delta_ema
                if len(idx):
                    if task_id==0:
                        ema_prop[iclass]=iprop.mean()
                    else:
                        ema_prop[iclass]=decay*ema_prop[iclass]+(1-decay)*iprop.mean()

                select_idx=(iprop>ema_prop[iclass]*opt.tau).nonzero().squeeze(1)
                select_res = select_res + [ds_res[idx[select_idx]]]
                select_label = select_label + [pseudo_label[idx[select_idx]]]
                select_prop = select_prop + [iprop[select_idx]]

            select_res = torch.cat(select_res, dim=0).cpu()
            select_label = torch.cat(select_label, dim=0).cpu()
            select_prop=torch.cat((select_prop),dim=0).cpu()
        
        predict_uIdx = (pseudo_label.view(-1, 1) == data.unseenclasses.to(opt.device)).sum(1)
        predict_uNum = predict_uIdx.sum()
        predict_sNum = pseudo_label.shape[0]-predict_uNum
        
        # log
        logger.info('evolution - task: [%d/%d] iter: [%d/%d] predict unseen: %d predict seen: %d expect samples: %d real samples: %d' % (
                task_id+1,num_task,t+1, opt.nIterate, predict_uNum,predict_sNum, opt.nShot, select_label.size(0)))    
        # evolution phase
        if select_res.size(0) !=0:
            # prepare dataloader
            ds_select = [(select_res[j], select_label[j],select_prop[j]) for j in range(select_label.size(0))]
            ds_loader = torch.utils.data.DataLoader(ds_select, batch_size=opt.BatchSize, shuffle=True,num_workers=opt.num_worker)

            for i, batch_data in enumerate(ds_loader, 0):
                netC.zero_grad()
                batch_res,batch_label,batch_prop=batch_data
                batch_res=batch_res.to(opt.device)
                batch_label=batch_label.to(opt.device)
                # batch_prop=batch_prop.to(opt.device)

                proto = netC(data.attribute.to(opt.device))
                proto = F.normalize(proto, dim=-1)

                batch_res_normed=F.normalize(batch_res,dim=-1)
                logits=batch_res_normed@proto.t()/ opt.tem
              
                # class selection
                uniqueBatch_label=torch.unique(batch_label)
                batch_label_mapped=util.map_label(batch_label,uniqueBatch_label).to(opt.device)

                loss_cls = F.cross_entropy(logits[:,uniqueBatch_label], batch_label_mapped,reduce=False)
                loss_cls=loss_cls.mean()

                #KL divergence
                proto_ema=netC_ema(data.attribute.to(opt.device))
                proto_ema=F.normalize(proto_ema,dim=-1)
                logits_ema = batch_res_normed @ proto_ema.t() / opt.tem
            
                loss_kl=F.kl_div(torch.log_softmax(logits,dim=-1),F.softmax(logits_ema,dim=-1).detach(),reduction = "batchmean")*opt.weight_kl
                loss=loss_cls+loss_kl
            
                
                loss.backward()
                optimizer.step()
                logger.info('evolution - loss cls: %.4f loss kl: %.4f loss: %.4f' %(loss_cls.item(),loss_kl.item(),loss.item()))
    return ema_prop

def random_experiment(opt,data,num_test,test_res,test_label,num_task,logger,seed=None):
    # set random seed
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info("Random Seed: %d"%(seed))

    random.seed(seed)
    torch.manual_seed(seed)
    if opt.device=='cuda':
        torch.cuda.manual_seed_all(seed)

    random_perm=torch.randperm(num_test)
    test_res=test_res[random_perm]
    test_label=test_label[random_perm]

    #load base model
    
    netC = model.netP(opt.proto_layer_sizes,opt.attSize)
    assert opt.netC != ''
    netC.load_state_dict(torch.load(opt.netC))
    logger.info('load base model: %s'%(netC))
    netC=netC.to(opt.device)

    optimizer = optim.Adam(netC.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))

    #initialize ema model
    netC_ema = copy.deepcopy(netC)
    for p in netC_ema.parameters():
        p.requires_grad = False


    # init \delta_ema
    ema_prop=torch.zeros((data.attribute.size(0),)).cuda()
    results_all=torch.Tensor([]).to(opt.device)
    
    for i in range(num_task+1):
        # prepare task data
        test_res_i=test_res[i*opt.nShot:(i+1)*opt.nShot]
        test_label_i=test_label[i*opt.nShot:(i+1)*opt.nShot]
      
        # test with last model
        task_prediction=val_task(test_res_i,netC,data,opt)
        acs_i,acu_i,ach_i=cal_H(task_prediction,test_label_i,data)
        
        logger.info('test - task [%d/%d] results: unseen acc = %.2f seen acc = %.2f H = %.2f' %(i ,num_task,acu_i*100,acs_i*100,ach_i*100))

        results_all=torch.cat((results_all,task_prediction),dim=0)
    
        # evolution
        if i <num_task:
            ema_prop=train_task(ema_prop,opt, test_res_i, data,  netC, optimizer, i, num_task,
                    netC_ema,logger)
            
            # update ema model
            decay = opt.m1
            for ema_v, model_v in zip(netC_ema.state_dict().values(), netC.state_dict().values()):
                ema_v.copy_(decay * ema_v + (1 - decay) * model_v.detach())

    acs,acu,ach=cal_H(results_all,test_label,data)
    logger.info('total results: unseen acc = %.2f, seen = %.2f, H = %.2f' % (acu*100, acs*100, ach*100))
    return acu,acs,ach

def main():
    opt=parse_args()
    # init work dir
    work_dir=os.path.join(opt.work_dir,opt.base_method,opt.dataset)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    nowTime = time.strftime("%Y-%m-%d_%H:%M:%S")
    logger=setup_logger(log_file=os.path.join(work_dir,nowTime+'.log'))
    
    logger.info(opt)
    # load data
    data = util.DATA_LOADER(opt)

    #split of evolutionary tasks
    num_test=data.test_unseen_label.size(0)+data.test_seen_label.size(0)
    num_task = num_test//opt.nShot
    rest_num=num_test-opt.nShot*num_task
    
    logger.info('total test samples: %d num task: %d samples per task: %d samples last task: %d'%(num_test,num_task,opt.nShot,rest_num))

    #test samples and labels
    test_res=torch.cat((data.test_seen_feature,data.test_unseen_feature),dim=0)
    test_label=torch.cat((data.test_seen_label,data.test_unseen_label),dim=0)
    
    acu,acs,ach=[],[],[]
    if opt.num_experiment>1:
        seed=[random.randint(0,2**32-1) for _ in range(opt.num_experiment)]
    else:
        seed=[opt.manualSeed]
    for i in range(opt.num_experiment):
        logger.info ("--------------------------------------random experiment %d on %s start--------------------------------------"%(i,opt.dataset))
        results=random_experiment(opt,data,num_test,test_res,test_label,num_task,logger,seed[i])
        acu_i,acs_i,ach_i=results
        acu.append(acu_i)
        acs.append(acs_i)
        ach.append(ach_i)
        logger.info ("--------------------------------------random experiment %d on %s finished--------------------------------------"%(i,opt.dataset))
    acu=np.stack(acu) *100
    acs=np.stack(acs) *100
    ach=np.stack(ach) *100
    acu_mean=acu.mean()
    acs_mean=acs.mean()
    ach_mean=ach.mean()
    acu_std=acu.std()
    acs_std=acs.std()
    ach_std=ach.std()
    
    logger.info('[%s] statistics in %d experiments: acu = %.2f±%.2f acs = %.2f±%.2f ach = %.2f±%.2f'
                %(opt.dataset,opt.num_experiment,acu_mean,acu_std,acs_mean,acs_std,ach_mean,ach_std) )
    # import pdb;pdb.set_trace()
if __name__ == '__main__':
    main()


