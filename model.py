import os
import numpy as np
from ResNet import ResNet18, ResNet50
from dataloader import CVLoader
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torchvision.models as torch_model
import matplotlib.pyplot as plt
import pandas as pd
from efficientnet_pytorch import EfficientNet
from focal_loss import FocalLoss
from ranger import Ranger

def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Calculates the quadratic weighted kappa
    quadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0
    
    if num_ratings == 1:
        return 0
    else:
        for i in range(num_ratings):
            for j in range(num_ratings):
                expected_count = (hist_rater_a[i] * hist_rater_b[j]
                                  / num_scored_items)
                d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
                numerator += d * conf_mat[i][j] / num_scored_items
                denominator += d * expected_count / num_scored_items
    
        return 1.0 - numerator / denominator

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def train_it(batch_data, batch_label, net, loss_function, optimizer, using_mixup, it):
    batch_data = batch_data.float().cuda()
    batch_label = batch_label.long()
    
    if using_mixup:
        batch_datas, batch_labels, batch_labelss, lam = mixup_data(batch_data, batch_label)
    
    net.train()
    
    loss = 0
    prediction = net(batch_data).cpu()
    
    if prediction.shape[1] == 5:
        
        if using_mixup:
            loss = mixup_criterion(loss_function, prediction, batch_labels, batch_labelss, lam)
        else:
            loss = loss_function(prediction, batch_label)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tb_log.add_scalar('loss', loss, it)
        
        return loss.detach()
    
    elif prediction.shape[1] == 6:
        prediction_classfication = prediction[:,0:5]
        prediction_regression = prediction[:,5]
        
        loss_classfication = loss_function(prediction_classfication, batch_label)
        # regression_loss = nn.L1Loss()
        regression_loss = nn.SmoothL1Loss()
        
        loss_regression = regression_loss(prediction_regression*2, batch_label.float()*2)
        loss = loss_classfication*0.1 + loss_regression*0.9
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tb_log.add_scalar('loss', loss, it)
        tb_log.add_scalar('loss_classfication', loss_classfication, it)
        tb_log.add_scalar('loss_regression', loss_regression, it)
        
        return loss.detach()
        
    else:
        pass


def eval_it(data, label, net, using_quadratic_weighted_kappa):
    data = data.float().cuda()
    label = label.long()
    
    net.eval()
    
    if using_quadratic_weighted_kappa:
       # prediction = np.argmax(F.softmax(net(data).cpu().detach(), dim=1).data.numpy(), axis=1)
        prediction = net(data)[:,5].cpu().detach()
        prediction[prediction > 4] = 4 
        prediction[prediction < 0] = 0 
        acc = quadratic_weighted_kappa(torch.round(prediction), label)
    else:
        prediction = np.argmax(F.softmax(net(data).cpu(), dim=1).data.numpy(), axis=1)
        acc = np.mean(np.equal(prediction.data,label.data.numpy()))
    
    return acc

def eval_one_weight(data, label, net, using_quadratic_weighted_kappa):
    data = data.float().cuda()
    label = label.long()
    
    net.eval()
 
    if using_quadratic_weighted_kappa:
        # prediction = np.argmax(F.softmax(net(data).cpu().detach(), dim=1).data.numpy(), axis=1)
        prediction = net(data)[:,5].cpu().detach()
        prediction[prediction > 4] = 4 
        prediction[prediction < 0] = 0 
        acc = quadratic_weighted_kappa(torch.round(prediction), label)
    else:
        prediction = np.argmax(F.softmax(net(data).cpu().detach(), dim=1).data.numpy(), axis=1)
        acc = np.mean(np.equal(prediction.data,label.data.numpy()))
    
    return acc, np.asarray(prediction.data), label.data.numpy()

def submit_one_weight(data, net):
    data = data.float().cuda()
    
    net.eval()
        
    prediction = np.argmax(F.softmax(net(data).cpu(), dim=1).data.numpy(), axis=1)
    
    return np.asarray(prediction.data)

def get_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    
    plt.matshow(df_confusion, cmap=cmap)
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')




def get_torch_model(model_name, fix_weight=False, fc_out_num=5):
    if model_name == "restnet18":        
        net = torch_model.resnet18(pretrained = True)
        if fix_weight:
            for param in net.parameters():
                param.requires_grad = False
        net.fc = nn.Linear(net.fc.in_features, fc_out_num)
        
    elif model_name == "restnet50": 
        net = torch_model.resnet50(pretrained = True)
        if fix_weight:
            for param in net.parameters():
                param.requires_grad = False
        net.fc = nn.Linear(net.fc.in_features, fc_out_num)
    
    elif model_name == "efficientnet": 
        net = EfficientNet.from_pretrained('efficientnet-b5')
        if fix_weight:
            for param in net.parameters():
                param.requires_grad = False
        net._fc = nn.Linear(net._fc.in_features, fc_out_num)
    elif model_name == "wide_resnet50_2": 
        net = torch_model.wide_resnet50_2(pretrained = True)
        if fix_weight:
            for param in net.parameters():
                param.requires_grad = False
        net.fc = nn.Linear(net.fc.in_features, fc_out_num)
        
    return net

def adjust_lr(optimizer, epoch):
    
    if epoch < 10:
        lr = 5e-4
    elif (epoch >= 10) and (epoch < 16):
        lr = 1e-5
    # elif (epoch > 12) and (epoch <= 30):
    #     lr = 1e-5
    elif (epoch >= 16) and (epoch < 22):
        lr = 5e-5
    else:
        lr = 1e-6
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

if __name__ == '__main__':

    #####################################################  Hyperparameters #####################################################
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    network = 5 # 0:my_resnet18 1: my_resnet50 2:torch_resnet18 3:torch_resnet50 4:torch_wide_resnet50_2 5:torch_efficientnet
    batch_size = 16
    stop_epoch = 100
    model_state = "train"  # train, eval, submit
    model_weight = "ckpt_11th_aug+focal+mixup+crop_first+2015/epoch_10.pkl"
    tensorboard_path = "./tensorboard/11th_aug+CE+mixup+crop_first+2015+regression+smoothL1+pretrain"
    ckpt_path = "ckpt_11th_aug+CE+mixup+crop_first+2015+regression+smoothL1+pretrain"
    img_size = 288
    fc_out_num = 5
    optimizers = 1 # 0:SGD 1:ADAM 2:RANGER
    loss_functions = 0 # 0:CE 1:Focal 2:Smooth L1
    initial_lr = 1e-4
    dynamic_lr = True
    using_2015 = True
    using_mixup = False
    pretrain_fix_weight = False
    using_quadratic_weighted_kappa = True
    #####################################################  Hyperparameters #####################################################    
        
    
    data_train = CVLoader("./","train", img_size, using_2015=using_2015)   
    data_test = CVLoader("./","test", img_size) 
    data_submit = CVLoader("./","submit", img_size) 


    if network == 0:
        net = ResNet18()
    elif network == 1:
        net = ResNet50()
    elif network == 2:
        net = get_torch_model("restnet18", fix_weight=pretrain_fix_weight, fc_out_num=fc_out_num)
    elif network == 3:
        net = get_torch_model("restnet50", fix_weight=pretrain_fix_weight, fc_out_num=fc_out_num)
    elif network == 4:
        net = get_torch_model("wide_resnet50_2", fix_weight=pretrain_fix_weight, fc_out_num=fc_out_num)
    elif network == 5:
        net = get_torch_model("efficientnet", fix_weight=pretrain_fix_weight, fc_out_num=fc_out_num)
        
    
    if model_weight == None:
        pass
    else:
        print("start loading weight...")
        net.load_state_dict(torch.load("./" + model_weight))
        print("finish loading weight")
        net._fc = nn.Linear(net._fc.in_features, 6)
 
    net.cuda()
    
    if loss_functions == 0:
        loss_function = nn.CrossEntropyLoss()
    elif loss_functions == 1:
        loss_function = FocalLoss()
    elif loss_functions == 2:
        loss_function = nn.SmoothL1Loss() 
        
    if optimizers == 0:
        optimizer = torch.optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)
    elif optimizers == 1:
        optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)
    elif optimizers == 2:
        optimizer = Ranger(net.parameters())
    
    highest_test_acc = 0
    it = 0  
    
    if model_state == "train":
        if not os.path.isdir(tensorboard_path):
            os.makedirs(tensorboard_path)
        tb_log = SummaryWriter(log_dir=tensorboard_path)  
        
        train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=4)
        
        for epoch in range(1, stop_epoch):
            
            if dynamic_lr:
                adjust_lr(optimizer, epoch)
            
            for cur_it, (batch_data, batch_label) in enumerate(train_loader):
                
                loss = train_it(batch_data, batch_label, net, loss_function, optimizer, using_mixup, it)
                
                it += 1
                print("================Epoch: ", epoch, "||  Batch: ", cur_it, "================")
                print("Loss: %.5f" % loss.data.numpy())
                
                train_acc = 0
                train_acc = eval_it(batch_data, batch_label, net, using_quadratic_weighted_kappa)
                tb_log.add_scalar('train_acc', train_acc, it)
                print("train_acc: %.4f" % train_acc)
                print("highest_test_acc: %.4f" % highest_test_acc)
                
            if epoch % 1 == 0:
                
                print("===============Start eval testing data=================")
                
                test_acc = 0

                for cur_it, (batch_data, batch_label) in enumerate(test_loader):
                    test_acc += eval_it(batch_data, batch_label, net, using_quadratic_weighted_kappa)

                    
                test_acc = test_acc / (cur_it + 1)
                
                if test_acc > highest_test_acc :
                    if not os.path.isdir("./" + ckpt_path):
                        os.makedirs("./" + ckpt_path)
                    save_name = "./" + ckpt_path + "/epoch_" + str(epoch) + ".pkl"
                    
                    if isinstance(net, nn.DataParallel):
                        torch.save(net.module.state_dict(), save_name)
                    else:
                        torch.save(net.state_dict(), save_name)
                
                highest_test_acc = max(highest_test_acc, test_acc)
                
                print("test_acc: %.4f " % test_acc)
                print("highest_test_acc: %.4f" % highest_test_acc)
                    
                tb_log.add_scalar('test_acc', test_acc, epoch)
                
                

    elif model_state == "eval":
        
        print("start loading weight...")
        net.load_state_dict(torch.load("./" + ckpt_path + "/" + model_weight))
        print("finish loading weight")
        
        # train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=4)
        # train_acc = 0
        test_acc = 0
        test_accs = 0
        # for cur_it, (batch_data, batch_label) in enumerate(train_loader):
        #     train_acc += eval_one_weight(batch_data, batch_label, net)
        # train_acc = train_acc / (cur_it + 1)
            
        for cur_it, (batch_data, batch_label) in enumerate(test_loader):
            print("iter: ", cur_it)
            test_acc, part_predic, part_label = eval_one_weight(batch_data, batch_label, net, using_quadratic_weighted_kappa)
            test_accs += test_acc
            if cur_it == 0:
                total_predic = part_predic*1
                total_label = part_label*1
            else:
                total_predic = np.concatenate((total_predic,part_predic), axis=0)
                total_label = np.concatenate((total_label,part_label), axis=0)
            
        test_accs = test_accs / (cur_it + 1)

        print("result of " + model_weight)
        # print("train_acc: %.4f" % train_acc)
        print("test_acc: %.4f " % test_accs)
        
        # y_actu = pd.Series(total_label.tolist(), name='Actual')
        # y_pred = pd.Series(total_predic.tolist(), name='Predicted')
        # df_confusion = pd.crosstab(y_actu, y_pred)
        # df_confusion = df_confusion.div(df_confusion.sum(axis=1),axis=0)
        # get_confusion_matrix(df_confusion)
    
    
    elif model_state == "submit":
        
        print("start loading weight...")
        net.load_state_dict(torch.load("./" + ckpt_path + "/" + model_weight))
        print("finish loading weight")
        submit_loader = DataLoader(data_submit, batch_size=batch_size, shuffle=True, num_workers=4)
        
        for cur_it, (batch_data,img_name) in enumerate(submit_loader):
            print("Batch:", cur_it)
            submit_prediction = submit_one_weight(batch_data, net)
            
            if cur_it == 0:
                temp_prediction = submit_prediction.copy()
                temp_img_name = img_name.numpy().copy()
                
            else:
                temp_prediction = np.concatenate((temp_prediction,submit_prediction))
                temp_img_name = np.concatenate((temp_img_name, img_name.numpy().copy()))

        output_array = np.vstack((temp_img_name,temp_prediction)).T
        
        df = pd.DataFrame(output_array, columns = ["id_code","diagnosis"])
        
        df.to_csv("./submit_result.csv", index=False)
        
        