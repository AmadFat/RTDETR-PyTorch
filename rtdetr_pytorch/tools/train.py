import copy
import datetime
import json
import os 
import sys
import re
from torch.cuda.amp import GradScaler, autocast
from torch import nn
from src.data.coco.coco_eval import CocoEvaluator
from src.misc.logger import MetricLogger
from src.solver.det_engine import train_one_epoch
from src.core.yaml_config import YAMLConfig
from src.solver.det_solver import DetSolver
from src.data.coco.coco_utils import get_coco_api_from_dataset
from src.data.dataloader import DataLoader, default_collate_fn
from src.data import transforms as T
from src.data.coco.coco_dataset import CocoDetection
from src.optim.optim import AdamW
from src.optim.ema import ModelEMA
from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
from src.zoo.rtdetr.rtdetr_criterion import SetCriterion
from src.zoo.rtdetr.matcher import HungarianMatcher
import src.misc.dist as dist
from rtest.utils import *
from src.zoo.rtdetr import rtdetr
from typing import Dict
import torch.optim.lr_scheduler as lr_scheduler

def rtdetr_criterion():
    matcher = HungarianMatcher(weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
                               use_focal_loss=True,
                               alpha=0.25,
                               gamma=2.0)
    
    criterion = SetCriterion(matcher=matcher,
                             weight_dict= {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2},
                             losses= ['vfl', 'boxes'],
                             alpha= 0.75,
                             gamma= 2.0)
    return criterion

def rtdetr_train_dataloader():
    train_dataset = CocoDetection(img_folder= "./dataset/coco/train2017/",
                         ann_file= "./dataset/coco/annotations/instances_train2017.json",
                         transforms = T.Compose([T.RandomPhotometricDistort(p=0.5), 
                                                 T.RandomZoomOut(fill=0), 
                                                 T.RandomIoUCrop(p=0.8),
                                                 T.SanitizeBoundingBox(min_size=1),
                                                 T.RandomHorizontalFlip(),
                                                 T.Resize(size=[640, 640]),
                                                 # transforms.Resize(size=639, max_size=640),
                                                 # # transforms.PadToSize(spatial_size=640),
                                                 T.ToImageTensor(),
                                                 T.ConvertDtype(),
                                                 T.SanitizeBoundingBox(min_size=1),
                                                 T.ConvertBox(out_fmt='cxcywh', normalize=True)]),
                         return_masks=False,
                         remap_mscoco_category=True)
    return DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=default_collate_fn, drop_last=True)

def rtdetr_val_dataloader():
    val_dataset = CocoDetection(img_folder= "./dataset/coco/val2017/",
                         ann_file= "./dataset/coco/annotations/instances_val2017.json",
                         transforms=T.Compose([T.Resize(size=[640, 640]), 
                                                 T.ToImageTensor(), 
                                                 T.ConvertDtype()]),
                         return_masks=False,
                         remap_mscoco_category=True)
    return DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=default_collate_fn, drop_last=False)

def load_tuning_state(path, model):
    def matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}

    """only load model for tuning and skip missed/dismatched keys
    """
    if 'http' in path:
        state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
    else:
        state = torch.load(path, map_location='cpu')

    module = dist.de_parallel(model)
    
    # TODO hard code
    if 'ema' in state:
        stat, infos = matched_state(module.state_dict(), state['ema']['module'])
    else:
        stat, infos = matched_state(module.state_dict(), state['model'])

    module.load_state_dict(stat, strict=False)
    print(f'Load model.state_dict, {infos}')


def get_optim_params(params, model: nn.Module):
    '''
    E.g.:
        ^(?=.*a)(?=.*b).*$         means including a and b
        ^((?!b.)*a((?!b).)*$       means including a but not b
        ^((?!b|c).)*a((?!b|c).)*$  means including a but not (b | c)
    '''

    if params == None:
        return model.parameters() 

    assert isinstance(params, list), ''

    param_groups = []
    visited = []
    for pg in params:
        pattern = pg['params']
        params = {k: v for k, v in model.named_parameters() if v.requires_grad and len(re.findall(pattern, k)) > 0}
        pg['params'] = params.values()
        param_groups.append(pg)
        visited.extend(list(params.keys()))

    names = [k for k, v in model.named_parameters() if v.requires_grad]

    if len(visited) < len(names):
        unseen = set(names) - set(visited)
        params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
        param_groups.append({'params': params.values()})
        visited.extend(list(params.keys()))

    assert len(visited) == len(names), ''

    return param_groups

def state_dict(model, last_epoch, optimizer, ema, scaler):
    '''state dict
    '''
    state = {}
    state['model'] = dist.de_parallel(model).state_dict()
    state['date'] = datetime.now().isoformat()

    # TODO
    state['last_epoch'] = last_epoch

    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()

    if lr_scheduler is not None:
        state['lr_scheduler'] = lr_scheduler.state_dict()
        # state['last_epoch'] = self.lr_scheduler.last_epoch

    if ema is not None:
        state['ema'] = ema.state_dict()

    if scaler is not None:
        state['scaler'] = scaler.state_dict()

    return state

def train(model, 
          weight_path, 
          optimizer, 
          output_dir,
          train_dataloader, 
          val_dataloader,
          epoch=72,
          amp=True, 
          criterion=rtdetr_criterion()):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    if weight_path != None:
        load_tuning_state(weight_path, model)
        
    if amp == True:
        ema_model = ModelEMA(model, decay=0.9999, warmups=2000)
        scaler = GradScaler()
   
    criterion.to(device)  

    scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[1000], gamma=0.1) 
    

    print("Start training")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
    best_stat = {'epoch': -1, }

    start_time = time.time()
    last_epoch = 0
    for epoch in range(last_epoch + 1, epoch):
        if dist.is_dist_available_and_initialized():
            train_dataloader.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, criterion, train_dataloader, optimizer, device, epoch,
            clip_max_norm=0.1, print_freq=100, ema=ema_model, scaler=scaler)

        scheduler.step()
        
 
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        # extra checkpoint before LR drop and every 100 epochs
        if (epoch + 1) % args.checkpoint_step == 0:
            checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
        for checkpoint_path in checkpoint_paths:
            dist.save_on_master(state_dict(epoch), checkpoint_path)

        module = ema_model.module if amp == True else model
        test_stats, coco_evaluator = val(model=module, weight_path=None, criterion=criterion, val_dataloader=val_dataloader)

        # TODO 
        for k in test_stats.keys():
            if k in best_stat:
                best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                best_stat[k] = max(best_stat[k], test_stats[k][0])
            else:
                best_stat['epoch'] = epoch
                best_stat[k] = test_stats[k][0]
        print('best_stat: ', best_stat)


        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch,
                    'n_parameters': n_parameters}

        if dist.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def val(model, weight_path, criterion, val_dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    if weight_path != None:
        load_tuning_state(weight_path, model)

    criterion.to(device) 

    base_ds = get_coco_api_from_dataset(val_dataloader.dataset)
    postprocessor = RTDETRPostProcessor(num_top_queries= 300)

    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")

    header = 'Test:'

    iou_types = postprocessor.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    panoptic_evaluator = None

    for samples, targets in metric_logger.log_every(val_dataloader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        results = postprocessor(outputs, orig_target_sizes)


        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)



    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}

    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    return stats, coco_evaluator


def rtdetr_r18vd_train():
    weight_path = "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"
    output_dir = "./output/rtdetr_r18vd_6x_coco"

    model = rtdetr.rtdetr_r18vd()
    model = dist.warp_model(model, find_unused_parameters=True, sync_bn=True)

    params= [{'params': '^(?=.*backbone)(?=.*norm).*$', 'lr': 0.00001, 'weight_decay': 0.},
             {'params': '^(?=.*backbone)(?!.*norm).*$', 'lr': 0.00001},
             {'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bias)).*$', 'weight_decay': 0.}]
    
    optimizer = AdamW(params=get_optim_params(params, model), lr=0.0001, betas=[0.9, 0.999], weight_decay=0.0001)

    train(model=model, weight_path=weight_path, optimizer=optimizer, output_dir=output_dir, train_dataloader=rtdetr_train_dataloader(), val_dataloader=rtdetr_val_dataloader())


def rtdetr_r34vd_train():
    weight_path = "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth"
    output_dir = "./output/rtdetr_r34vd_6x_coco"

    model = rtdetr.rtdetr_r34vd()
    model = dist.warp_model(model, find_unused_parameters=True, sync_bn=True)

    params = [{'params': '^(?=.*backbone)(?=.*norm|bn).*$', 'weight_decay': 0., 'lr': 0.00001},
              {'params': '^(?=.*backbone)(?!.*norm|bn).*$', 'lr': 0.00001}, 
              {'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$', 'weight_decay': 0.}]

    optimizer = AdamW(params=get_optim_params(params, model), lr=0.0001, betas=[0.9, 0.999], weight_decay=0.0001)

    train(model=model, weight_path=weight_path, optimizer=optimizer, output_dir=output_dir, train_dataloader=rtdetr_train_dataloader(), val_dataloader=rtdetr_val_dataloader())


def rtdetr_r50vd_train():
    weight_path = "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth"
    output_dir = "./output/rtdetr_r50vd_6x_coco"

    model = rtdetr.rtdetr_r50vd()
    model = dist.warp_model(model, find_unused_parameters=True, sync_bn=True)

    params= [{'params': 'backbone', 'lr': 0.00001},
             {'params': '^(?=.*encoder(?=.*bias|.*norm.*weight)).*$', 'weight_decay': 0.},
             {'params': '^(?=.*decoder(?=.*bias|.*norm.*weight)).*$', 'weight_decay': 0.}]

    optimizer=AdamW(params=get_optim_params(params, model), lr=0.0001, betas=[0.9, 0.999], weight_decay=0.0001)

    train(model=model, weight_path=weight_path, optimizer=optimizer, output_dir=output_dir, train_dataloader=rtdetr_train_dataloader(), val_dataloader=rtdetr_val_dataloader())


def rtdetr_r50vd_m_train():
    weight_path = "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth"
    output_dir = "./output/rtdetr_r50vd_m_6x_coco"

    model = rtdetr.rtdetr_r50vd_m()
    model = dist.warp_model(model, find_unused_parameters=True, sync_bn=True)

    params= [{'params': 'backbone', 'lr': 0.00001},
             {'params': '^(?=.*encoder(?=.*bias|.*norm.*weight)).*$', 'weight_decay': 0.},
             {'params': '^(?=.*decoder(?=.*bias|.*norm.*weight)).*$', 'weight_decay': 0.}]

    optimizer=AdamW(params=get_optim_params(params, model), lr=0.0001, betas=[0.9, 0.999], weight_decay=0.0001)

    train(model=model, weight_path=weight_path, optimizer=optimizer, output_dir=output_dir, train_dataloader=rtdetr_train_dataloader(), val_dataloader=rtdetr_val_dataloader())


def rtdetr_r101vd_train():
    weight_path = "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth"
    output_dir = "./output/rtdetr_r101vd_6x_coco"

    model = rtdetr.rtdetr_r101vd()
    model = dist.warp_model(model, find_unused_parameters=True, sync_bn=True)

    params= [{'params': 'backbone', 'lr': 0.00001},
             {'params': '^(?=.*encoder(?=.*bias|.*norm.*weight)).*$', 'weight_decay': 0.},
             {'params': '^(?=.*decoder(?=.*bias|.*norm.*weight)).*$', 'weight_decay': 0.}]

    optimizer=AdamW(params=get_optim_params(params, model), lr=0.00001, betas=[0.9, 0.999], weight_decay=0.0001)

    train(model=model, weight_path=weight_path, optimizer=optimizer, output_dir=output_dir, train_dataloader=rtdetr_train_dataloader(), val_dataloader=rtdetr_val_dataloader())



if __name__ == '__main__':
    rtdetr_r18vd_train()
    Setting.print_shape = True
    #변수 초기화 'Only support from_scrach or resume or tuning at one time'
    config = 'rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml'  #설정 파일 경로
    resume = None  # resume = '../checkpoint'
    path = 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth' # 저장된 가중치 경로
    amp = True # 자동 혼합 정밀도(Automatic Mixed Precision, AMP) FP16 FP32 섞어서 사용. 메모리 사용 감소, 에너지 사용 감소, 계산 속도 향상의 장점
    test_only = False


    