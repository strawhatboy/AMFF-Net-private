import torch
import numpy as np
from model import MTD_IQA_modify
import random
from torch.utils.tensorboard import SummaryWriter as sum_writer
from MNL_Loss import loss_m3
from tools import set_dataset4, set_dataset_csv, _preprocess2, _preprocess3, convert_models_to_fp32, compute_metric
import os
from tqdm import tqdm
import pickle
import argparse
import pandas as pd

##############################general setup####################################
AGIQA3K_set = r'/nas_pool_a/cuichuan/dataset/agiqa-3k/images'
AIGCIQA2023_set = r'/nas_pool_a/cuichuan/dataset/AIGCIQA2023/images'
AIGCQA20K_set = r'/nas_pool_a/cuichuan/dataset/AIGCQA-30K-Image/images'  # Replace PKU-I2IQA with AIGCQA-20k

# CSV file paths - now using local IQA_Database directory
AGIQA3K_csv_path = '/nas_pool_a/cuichuan/dataset/agiqa-3k/712_splits'
AIGCIQA2023_csv_path = '/nas_pool_a/cuichuan/dataset/AIGCIQA2023/712_splits'
AIGCQA20K_csv_path = '/nas_pool_a/cuichuan/dataset/AIGCQA-30K-Image'

seed = 2222

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#################### hyperparameter #####################
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Parse command line arguments
parser = argparse.ArgumentParser(description='MTD-IQA Training')
parser.add_argument('--dataset', type=str, required=True, 
                    choices=['AGIQA3K', 'AIGCIQA2023', 'AIGCQA20K'],
                    help='Dataset to train on: AGIQA3K, AIGCIQA2023, or AIGCQA20K')
args = parser.parse_args()

# Use the specified dataset
dataset = args.dataset
print(f'Running training on dataset: {dataset}')

radius = [336, 224, 112]
initial_lr1 = 5e-4
initial_lr2 = 5e-6
weight_decay = 0.001
num_epoch = 100
bs = 32
early_stop = 0
clip_net = 'RN50'
in_size = 1024
istrain = True

##############################general setup####################################

preprocess2 = [_preprocess2(radius[0]), _preprocess2(radius[1]), _preprocess2(radius[2])]
preprocess3 = [_preprocess3(radius[0]), _preprocess3(radius[1]), _preprocess3(radius[2])]
loss_fn = torch.nn.MSELoss().to(device)


def freeze_model(f_model, opt):
    f_model.logit_scale.requires_grad = False
    if opt == 0: #do nothing
        return
    elif opt == 1: # freeze text encoder
        for p in f_model.token_embedding.parameters():
            p.requires_grad = False
        for p in f_model.transformer.parameters():
            p.requires_grad = False
        f_model.positional_embedding.requires_grad = False
        f_model.text_projection.requires_grad = False
        for p in f_model.ln_final.parameters():
            p.requires_grad = False
    elif opt == 2: # freeze visual encoder
        for p in f_model.visual.parameters():
            p.requires_grad = False
    elif opt == 3:
        for p in f_model.parameters():
            p.requires_grad = False
    elif opt == 4:
        for p in f_model.parameters():
            p.requires_grad = True


def do_batch(x_l, x_m, x_s, con_text):

    input_token_c = con_text.view(-1, 77)
    logits_per_qua, logits_per_con, logits_per_aes = model.forward(x_l, x_m, x_s, input_token_c)

    return logits_per_qua, logits_per_con, logits_per_aes


def train(model, best_result, best_epoch):
    model.train()  # Set to training mode for proper training

    global early_stop
    print(optimizer.state_dict()['param_groups'][0]['lr'])

    for idx, sample_batched in enumerate(tqdm(train_loaders)):

        x_l, x_m, x_s, mos_q, mos_a, mos_c, con_tokens = sample_batched['img_l'], sample_batched['img_m'], \
                                                         sample_batched['img_s'], sample_batched['mos_q'], \
                                                         sample_batched['mos_a'], sample_batched['mos_c'], \
                                                         sample_batched['con_tokens']

        img_name = sample_batched['img_name']
        x_l = x_l.to(torch.float32).to(device)
        x_m = x_m.to(torch.float32).to(device)
        x_s = x_s.to(torch.float32).to(device)

        mos_q = mos_q.to(torch.float32).to(device)
        mos_a = mos_a.to(torch.float32).to(device)
        mos_c = mos_c.to(torch.float32).to(device)
        con_tokens = con_tokens.to(device)

        optimizer.zero_grad()
        logits_per_qua, logits_per_con, logits_per_aes = do_batch(x_l, x_m, x_s, con_tokens)

        weight_qua = logits_per_qua[:, 0]
        weight_con = logits_per_con[:, 0]
        weight_aes = logits_per_aes[:, 0]
        loss_q = loss_fn(weight_qua, mos_q.detach())
        loss_c = loss_m3(weight_con, mos_c.detach())

        if mtl == 0:  # AGIQA3K - quality + correspondence
            total_loss = loss_q + loss_c
        elif mtl == 1:  # AIGCIQA2023 - quality + aesthetic + correspondence
            loss_a = loss_fn(weight_aes, mos_a.detach())
            total_loss = loss_q + loss_a + loss_c
        elif mtl == 2:  # AIGCQA20K - only quality (single MOS)
            total_loss = loss_q  # Only use quality loss for single MOS dataset

        if torch.any(torch.isnan(total_loss)):
            print('nan in', idx)

        total_loss.backward()
        # statistics
        if not pretrain:
            global global_step
            logger.add_scalar(tag='total_loss', scalar_value=total_loss.item(), global_step=global_step)
            logger.add_scalar(tag='loss_q', scalar_value=loss_q.item(), global_step=global_step)
            if mtl == 1:  # AIGCIQA2023 has aesthetic score
                logger.add_scalar(tag='loss_a', scalar_value=loss_a.item(), global_step=global_step)
            if mtl in [0, 1]:  # AGIQA3K and AIGCIQA2023 have correspondence score
                logger.add_scalar(tag='loss_c', scalar_value=loss_c.item(), global_step=global_step)
            global_step += 1

        convert_models_to_fp32(model)
        optimizer.step()

    # Validate on validation set
    val_out = eval(loader=val_loaders)
    val_srcc_q = val_out[0]
    val_srcc_a = val_out[3] if mtl == 1 else 0.0
    val_srcc_c = val_out[6] if mtl in [0, 1] else 0.0
    
    if mtl == 2:  # AIGCQA20K - only quality score
        val_srcc_avg = val_srcc_q
    else:  # AGIQA3K or AIGCIQA2023 - multiple scores
        val_srcc_avg = (val_srcc_q + val_srcc_a + val_srcc_c) / 3
        
    print("VAL - srccc_avg: {:.3f}\tsrcc_q: {:.3f}\tsrcc_a: {:.3f}\tsrcc_c: {:.3f}\tloss: {:.3f}".format(val_srcc_avg, val_srcc_q, val_srcc_a, val_srcc_c, total_loss))

    if not os.path.exists(os.path.join('checkpoints', dataset, 'MTD_IQA')):
        os.makedirs(os.path.join('checkpoints', dataset, 'MTD_IQA'))
    if val_srcc_avg > best_result['avg']:
        early_stop = 0
        ckpt_name = os.path.join('checkpoints', dataset, 'MTD_IQA', 'avg_best_ckpt.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_results': val_out
        }, ckpt_name)
        best_epoch['avg'] = epoch
        best_result['avg'] = val_srcc_avg

    early_stop += 1

    return best_result, best_epoch, val_out

def eval(loader):
    model.eval()
    y_q = []
    y_pred_q = []
    y_a = []
    y_pred_a = []
    y_c = []
    y_pred_c = []
    for step, sample_batched in enumerate(loader):
        x_l, x_m, x_s, mos_q, mos_a, mos_c, con_tokens = sample_batched['img_l'], sample_batched['img_m'], \
            sample_batched['img_s'], sample_batched['mos_q'], \
            sample_batched['mos_a'], sample_batched['mos_c'], \
            sample_batched['con_tokens']

        img_name = sample_batched['img_name']
        x_l = x_l.to(torch.float32).to(device)
        x_m = x_m.to(torch.float32).to(device)
        x_s = x_s.to(torch.float32).to(device)

        mos_q = mos_q.to(torch.float32).to(device)
        mos_a = mos_a.to(torch.float32).to(device)
        mos_c = mos_c.to(torch.float32).to(device)
        con_tokens = con_tokens.to(device)

        with torch.no_grad():
            logits_per_qua, logits_per_con, logits_per_aes = do_batch(x_l, x_m, x_s, con_tokens)

            weight_qua = logits_per_qua[:, 0]
            weight_aes = logits_per_aes[:, 0]
            weight_con = logits_per_con[:, 0]

            y_pred_q.extend(weight_qua.cpu().numpy())
            y_pred_a.extend(weight_aes.cpu().numpy())
            y_pred_c.extend(weight_con.cpu().numpy())
            y_q.extend(mos_q.cpu().numpy())
            y_a.extend(mos_a.cpu().numpy())
            y_c.extend(mos_c.cpu().numpy())

    _, PLCC1, SRCC1, KRCC1 = compute_metric(np.array(y_q), np.array(y_pred_q), istrain)
    if mtl == 1:  # AIGCIQA2023 has aesthetic score
        _, PLCC2, SRCC2, KRCC2 = compute_metric(np.array(y_a), np.array(y_pred_a), istrain)
    else:
        _, PLCC2, SRCC2, KRCC2 = 0.0, 0.0, 0.0, 0.0
    if mtl in [0, 1]:  # AGIQA3K and AIGCIQA2023 have correspondence score
        _, PLCC3, SRCC3, KRCC3 = compute_metric(np.array(y_c), np.array(y_pred_c), istrain)
    else:
        _, PLCC3, SRCC3, KRCC3 = 0.0, 0.0, 0.0, 0.0

    out = [SRCC1, PLCC1, KRCC1,
           SRCC2, PLCC2, KRCC2,
           SRCC3, PLCC3, KRCC3]
    return out

def final_test_evaluation(model, test_loaders):
    """
    Final evaluation on test set with the best model
    """
    model.eval()
    
    # Collect detailed predictions for CSV output
    detailed_results = []
    y_q = []
    y_pred_q = []
    y_a = []
    y_pred_a = []
    y_c = []
    y_pred_c = []
    img_names = []
    
    for step, sample_batched in enumerate(test_loaders):
        x_l, x_m, x_s, mos_q, mos_a, mos_c, con_tokens = sample_batched['img_l'], sample_batched['img_m'], \
            sample_batched['img_s'], sample_batched['mos_q'], \
            sample_batched['mos_a'], sample_batched['mos_c'], \
            sample_batched['con_tokens']

        img_name = sample_batched['img_name']
        x_l = x_l.to(torch.float32).to(device)
        x_m = x_m.to(torch.float32).to(device)
        x_s = x_s.to(torch.float32).to(device)

        mos_q = mos_q.to(torch.float32).to(device)
        mos_a = mos_a.to(torch.float32).to(device)
        mos_c = mos_c.to(torch.float32).to(device)
        con_tokens = con_tokens.to(device)

        with torch.no_grad():
            logits_per_qua, logits_per_con, logits_per_aes = do_batch(x_l, x_m, x_s, con_tokens)

            weight_qua = logits_per_qua[:, 0]
            weight_aes = logits_per_aes[:, 0]
            weight_con = logits_per_con[:, 0]

            # Collect batch results
            batch_pred_q = weight_qua.cpu().numpy()
            batch_pred_a = weight_aes.cpu().numpy()
            batch_pred_c = weight_con.cpu().numpy()
            batch_mos_q = mos_q.cpu().numpy()
            batch_mos_a = mos_a.cpu().numpy()
            batch_mos_c = mos_c.cpu().numpy()
            
            # Store for overall metrics
            y_pred_q.extend(batch_pred_q)
            y_pred_a.extend(batch_pred_a)
            y_pred_c.extend(batch_pred_c)
            y_q.extend(batch_mos_q)
            y_a.extend(batch_mos_a)
            y_c.extend(batch_mos_c)
            
            # Store detailed results for CSV
            for i in range(len(batch_pred_q)):
                img_basename = os.path.basename(img_name[i])
                result_row = {'name': img_basename}
                
                # Add dataset-specific columns
                if mtl == 0:  # AGIQA3K
                    result_row['predicted_mos_quality'] = batch_pred_q[i]
                    result_row['predicted_mos_align'] = batch_pred_c[i]
                    result_row['mos_quality'] = batch_mos_q[i]
                    result_row['mos_align'] = batch_mos_c[i]
                elif mtl == 1:  # AIGCIQA2023
                    result_row['predicted_mos_quality'] = batch_pred_q[i]
                    result_row['predicted_mos_authenticity'] = batch_pred_a[i]
                    result_row['predicted_mos_correspondence'] = batch_pred_c[i]
                    result_row['mos_quality'] = batch_mos_q[i]
                    result_row['mos_authenticity'] = batch_mos_a[i]
                    result_row['mos_correspondence'] = batch_mos_c[i]
                elif mtl == 2:  # AIGCQA20K
                    result_row['predicted_mos'] = batch_pred_q[i]
                    result_row['mos'] = batch_mos_q[i]
                
                detailed_results.append(result_row)

    # Calculate overall metrics
    _, PLCC1, SRCC1, KRCC1 = compute_metric(np.array(y_q), np.array(y_pred_q), istrain)
    if mtl == 1:  # AIGCIQA2023 has aesthetic score
        _, PLCC2, SRCC2, KRCC2 = compute_metric(np.array(y_a), np.array(y_pred_a), istrain)
    else:
        _, PLCC2, SRCC2, KRCC2 = 0.0, 0.0, 0.0, 0.0
    if mtl in [0, 1]:  # AGIQA3K and AIGCIQA2023 have correspondence score
        _, PLCC3, SRCC3, KRCC3 = compute_metric(np.array(y_c), np.array(y_pred_c), istrain)
    else:
        _, PLCC3, SRCC3, KRCC3 = 0.0, 0.0, 0.0, 0.0

    test_out = [SRCC1, PLCC1, KRCC1,
                SRCC2, PLCC2, KRCC2,
                SRCC3, PLCC3, KRCC3]
    
    test_srcc_q = test_out[0]
    test_srcc_a = test_out[3] if mtl == 1 else 0.0
    test_srcc_c = test_out[6] if mtl in [0, 1] else 0.0
    
    if mtl == 2:  # AIGCQA20K - only quality score
        test_srcc_avg = test_srcc_q
    else:  # AGIQA3K or AIGCIQA2023 - multiple scores
        test_srcc_avg = (test_srcc_q + test_srcc_a + test_srcc_c) / 3
        
    print("FINAL TEST - srccc_avg: {:.3f}\tsrcc_q: {:.3f}\tsrcc_a: {:.3f}\tsrcc_c: {:.3f}".format(test_srcc_avg, test_srcc_q, test_srcc_a, test_srcc_c))
    
    # Save detailed results to CSV
    results_df = pd.DataFrame(detailed_results)
    csv_path = os.path.join('checkpoints', dataset, 'MTD_IQA', 'test_predictions.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Detailed test results saved to: {csv_path}")
    
    return test_out

num_workers = 8

# Setup dataset configuration
mtl_map = {'AGIQA3K': 0, 'AIGCIQA2023': 1, 'AIGCQA20K': 2}
mtl = mtl_map[dataset]
change_epoch = {'AGIQA3K': 60, 'AIGCIQA2023': 60, 'AIGCQA20K': 60}

print('train on ', dataset)

# Use single session with train/val/test splits from CSV files
model = MTD_IQA_modify.MTD_IQA(device=device, clip_net=clip_net, in_size=in_size)
model = model.to(device)

runs_path = os.path.join('./log', dataset, 'MTD_IQA')
logger = sum_writer(runs_path)
train_loss = []
early_stop = 0
start_epoch = 0
global_step = 0
pretrain = True
best_result = {'avg': 0.0}
best_epoch = {'avg': 0}

# Set up dataset paths based on dataset type
if dataset == 'AGIQA3K':
    dataset_path = AGIQA3K_set
    csv_base_path = AGIQA3K_csv_path
    dataset_idx = 0
elif dataset == 'AIGCIQA2023':
    dataset_path = AIGCIQA2023_set
    csv_base_path = AIGCIQA2023_csv_path
    dataset_idx = 1
else:  # AIGCQA20K
    dataset_path = AIGCQA20K_set
    csv_base_path = AIGCQA20K_csv_path
    dataset_idx = 2

# Use CSV files for all datasets
train_csv = os.path.join(csv_base_path, 'train.csv')
val_csv = os.path.join(csv_base_path, 'val.csv')
test_csv = os.path.join(csv_base_path, 'test.csv')

train_loaders = set_dataset_csv(train_csv, bs, dataset_path, radius, num_workers, preprocess3, dataset_idx, False)
val_loaders = set_dataset_csv(val_csv, bs, dataset_path, radius, num_workers, preprocess2, dataset_idx, True)
test_loaders = set_dataset_csv(test_csv, bs, dataset_path, radius, num_workers, preprocess2, dataset_idx, True)

initial_lr1 = 1e-3 if dataset == 'AIGCIQA2023' else 5e-4
optimizer1 = torch.optim.AdamW(model.parameters(), lr=initial_lr1, weight_decay=weight_decay)
optimizer2 = torch.optim.AdamW(model.parameters(), lr=initial_lr2, weight_decay=weight_decay)

result_pkl = {}

# pretrain
for epoch in range(0, 20):
    freeze_model(model.base, opt=3)
    optimizer = optimizer1
    best_result, best_epoch, all_result = train(model, best_result, best_epoch)
    print(epoch, best_result)

pre_pth = torch.load(os.path.join('checkpoints', dataset, 'MTD_IQA', 'avg_best_ckpt.pt'))
model.load_state_dict(pre_pth['model_state_dict'], strict=True)
pretrain = False
freeze_model(model.base, opt=4)

for epoch in range(0, num_epoch):
    if epoch >= change_epoch[dataset]:
        optimizer2 = torch.optim.AdamW(model.parameters(), lr=initial_lr2/10, weight_decay=weight_decay)
    optimizer = optimizer2

    print(f'begin epoch {epoch}')
    best_result, best_epoch, all_result = train(model, best_result, best_epoch)

    result_pkl[str(epoch)] = all_result

    if epoch % 5 == 0:
        print('...............current average best...............')
        print('best average epoch:{}'.format(best_epoch['avg']))
        print('best average result:{}'.format(best_result['avg']))

    if early_stop > 20:
        print(f'early stopping at epoch {epoch}!')
        break

# Final evaluation on test set
print("\n" + "="*50)
print("FINAL EVALUATION ON TEST SET")
print("="*50)

# Load best model
best_ckpt = torch.load(os.path.join('checkpoints', dataset, 'MTD_IQA', 'avg_best_ckpt.pt'))
model.load_state_dict(best_ckpt['model_state_dict'], strict=True)

# Evaluate on test set
final_test_results = final_test_evaluation(model, test_loaders)
result_pkl['final_test'] = final_test_results

pkl_name = os.path.join('checkpoints', dataset, 'MTD_IQA', 'all_results.pkl')
with open(pkl_name, 'wb') as f:
    pickle.dump(result_pkl, f)
