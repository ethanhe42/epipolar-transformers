from tqdm import tqdm
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import torch

from modeling.model import Modelbuilder
from utils.checkpoint import Checkpointer
from utils.metric_logger import MetricLogger
from utils.logger import setup_logger
from utils.misc import mkdir, prefix_dict
from utils.timer import Timer, get_time_str
from data.build import make_data_loader
from vision.visualizer_human import draw_2d_pose
from vision.visualizer_hand import plot_two_hand_2d
from vision.visualization import de_transform

def test(cfg, model=None):
    torch.cuda.empty_cache()  # TODO check if it helps
    cpu_device = torch.device("cpu")
    if cfg.VIS.FLOPS:
        # device = cpu_device
        device = torch.device("cuda:0")
    else:
        device = torch.device(cfg.DEVICE)
    if model is None:
        # load model from outputs
        model = Modelbuilder(cfg)
        model.to(device)
        checkpointer = Checkpointer(model, save_dir=cfg.OUTPUT_DIR)
        _ = checkpointer.load(cfg.WEIGHTS)
    data_loaders = make_data_loader(cfg, is_train=False)
    if cfg.VIS.FLOPS:
        model.eval()
        from thop import profile
        for idx, batchdata in enumerate(data_loaders[0]):
            with torch.no_grad():
                flops, params = profile(model, inputs=({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batchdata.items()}, False))
                print('flops', flops, 'params', params)
                exit()
    if cfg.TEST.RECOMPUTE_BN:
        tmp_data_loader = make_data_loader(cfg, is_train=True, dataset_list=cfg.DATASETS.TEST)
        model.train()
        for idx, batchdata in enumerate(tqdm(tmp_data_loader)):
            with torch.no_grad():
                model({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batchdata.items()}, is_train=True)
        #cnt = 0
        #while cnt < 1000:
        #    for idx, batchdata in enumerate(tqdm(tmp_data_loader)):
        #        with torch.no_grad():
        #            model({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batchdata.items()}, is_train=True)
        #        cnt += 1
        checkpointer.save("model_bn")
        model.eval()
    elif cfg.TEST.TRAIN_BN:
        model.train()
    else:
        model.eval()
    dataset_names = cfg.DATASETS.TEST
    meters = MetricLogger()

    #if cfg.TEST.PCK and cfg.DOTEST and 'h36m' in cfg.OUTPUT_DIR:
    #    all_preds = np.zeros((len(data_loaders), cfg.KEYPOINT.NUM_PTS, 3), dtype=np.float32)
    cpu = lambda x: x.to(cpu_device).numpy() if isinstance(x, torch.Tensor) else x

    logger = setup_logger("tester", cfg.OUTPUT_DIR)
    for data_loader, dataset_name in zip(data_loaders, dataset_names):
        print('Loading ', dataset_name)
        dataset = data_loader.dataset

        logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
        total_timer = Timer()
        total_timer.tic()

        predictions = []
        #if 'h36m' in cfg.OUTPUT_DIR:
        #    err_joints = 0
        #else:
        err_joints = np.zeros((cfg.TEST.IMS_PER_BATCH, int(cfg.TEST.MAX_TH)))
        total_joints = 0
        
        for idx, batchdata in enumerate(tqdm(data_loader)):
            if cfg.VIS.VIDEO and not 'h36m' in cfg.OUTPUT_DIR:
                for k, v in batchdata.items():
                    try:
                        #good 1 2 3 4 5 6 7 8 12 16 30
                        # 4 17.4 vs 16.5
                        # 30 41.83200 vs 40.17562
                        #bad 0 22
                        #0 43.78544 vs 45.24059
                        #22 43.01385 vs 43.88636
                        vis_idx = 16
                        batchdata[k] = v[:, vis_idx, None]
                    except:
                        pass
            if cfg.VIS.VIDEO_GT:
                for k, v in batchdata.items():
                    try:
                        vis_idx = 30
                        batchdata[k] = v[:, vis_idx:vis_idx+2]
                    except:
                        pass                     
                joints = cpu(batchdata['points-2d'].squeeze())[0]              
                orig_img = de_transform(cpu(batchdata['img'].squeeze()[None, ...])[0][0])
                # fig = plt.figure()
                # ax = fig.add_subplot(111)
                ax = display_image_in_actual_size(orig_img.shape[1], orig_img.shape[2])
                if 'h36m' in cfg.OUTPUT_DIR:
                    draw_2d_pose(joints, ax)
                    orig_img = orig_img[::-1]
                else:
                    visibility = cpu(batchdata['visibility'].squeeze())[0]
                    plot_two_hand_2d(joints, ax, visibility)
                    # plot_two_hand_2d(joints, ax)
                ax.imshow(orig_img.transpose((1,2,0)))
                ax.axis('off')
                output_folder = os.path.join("outs", "video_gt", dataset_name)
                mkdir(output_folder)
                plt.savefig(
                    os.path.join(output_folder, "%08d" % idx), 
                    bbox_inches="tight", pad_inches=0)
                plt.cla()
                plt.clf()
                plt.close()
                continue
            #print('batchdatapoints-3d', batchdata['points-3d'])
            batch_size = cfg.TEST.IMS_PER_BATCH
            with torch.no_grad():
                loss_dict, metric_dict, output = model(
                        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batchdata.items()}, 
                        is_train=False)
            meters.update(**prefix_dict(loss_dict, dataset_name))
            meters.update(**prefix_dict(metric_dict, dataset_name))
            # udpate err_joints
            if cfg.VIS.VIDEO:
                joints = cpu(output['batch_locs'].squeeze())
                if joints.shape[0] == 1:
                    joints= joints[0]                     
                try:
                    orig_img = de_transform(cpu(batchdata['img'].squeeze()[None, ...])[0][0])
                except:
                    orig_img = de_transform(cpu(batchdata['img'].squeeze()[None, ...])[0])                # fig = plt.figure()
                # ax = fig.add_subplot(111)
                ax = display_image_in_actual_size(orig_img.shape[1], orig_img.shape[2])
                if 'h36m' in cfg.OUTPUT_DIR:
                    draw_2d_pose(joints, ax)
                    orig_img = orig_img[::-1]
                else:
                    visibility = cpu(batchdata['visibility'].squeeze())
                    if visibility.shape[0] == 1:
                        visibility= visibility[0]
                    plot_two_hand_2d(joints, ax, visibility)
                ax.imshow(orig_img.transpose((1,2,0)))
                ax.axis('off')
                output_folder = os.path.join(cfg.OUTPUT_DIR, "video", dataset_name)
                mkdir(output_folder)
                plt.savefig(
                    os.path.join(output_folder, "%08d" % idx), 
                    bbox_inches="tight", pad_inches=0)
                plt.cla()
                plt.clf()                    
                plt.close()
                # plt.show()
                
            if cfg.TEST.PCK and cfg.DOTEST:
                #if 'h36m' in cfg.OUTPUT_DIR:
                #    err_joints += metric_dict['accuracy'] * output['total_joints']
                #    total_joints += output['total_joints']
                #    # all_preds
                #else:
                for i in range(batch_size):
                    err_joints = np.add(err_joints, output['err_joints'])
                    total_joints += sum(output['total_joints'])
            
            if idx % cfg.VIS.SAVE_PRED_FREQ == 0 and (cfg.VIS.SAVE_PRED_LIMIT == -1 or idx < cfg.VIS.SAVE_PRED_LIMIT * cfg.VIS.SAVE_PRED_FREQ):
                # print(meters)
                for i in range(batch_size):
                    predictions.append(
                            (
                                {k: (cpu(v[i]) if not isinstance(v, int) else v) for k, v in batchdata.items()}, 
                                {k: (cpu(v[i]) if not isinstance(v, int) else v) for k, v in output.items()}, 
                            )
                    )
            if cfg.VIS.SAVE_PRED_LIMIT!= -1 and idx > cfg.VIS.SAVE_PRED_LIMIT * cfg.VIS.SAVE_PRED_FREQ:
                break

            # if not cfg.DOTRAIN and cfg.SAVE_PRED:
            #     if cfg.VIS.SAVE_PRED_LIMIT != -1 and idx < cfg.VIS.SAVE_PRED_LIMIT:
            #         for i in range(batch_size):
            #             predictions.append(
            #                     (
            #                         {k: (cpu(v[i]) if not isinstance(v, int) else v) for k, v in batchdata.items()}, 
            #                         {k: (cpu(v[i]) if not isinstance(v, int) else v) for k, v in output.items()}, 
            #                     )
            #             )
            #     if idx == cfg.VIS.SAVE_PRED_LIMIT:
            #         break
        #if cfg.TEST.PCK and cfg.DOTEST and 'h36m' in cfg.OUTPUT_DIR:
        #    logger.info('accuracy0.5: {}'.format(err_joints/total_joints))
            # dataset.evaluate(all_preds)
            # name_value, perf_indicator = dataset.evaluate(all_preds)
            # names = name_value.keys()
            # values = name_value.values()
            # num_values = len(name_value)
            # logger.info(' '.join(['| {}'.format(name) for name in names]) + ' |')
            # logger.info('|---' * (num_values) + '|')
            # logger.info(' '.join(['| {:.3f}'.format(value) for value in values]) + ' |')    

        total_time = total_timer.toc()
        total_time_str = get_time_str(total_time)
        logger.info("Total run time: {} ".format(total_time_str))

        if cfg.OUTPUT_DIR: #and cfg.VIS.SAVE_PRED:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            torch.save(predictions, os.path.join(output_folder, cfg.VIS.SAVE_PRED_NAME))
            if cfg.DOTEST and cfg.TEST.PCK:
                print(err_joints.shape)
                torch.save(err_joints * 1.0 / total_joints, os.path.join(output_folder, "pck.pth"))

    logger.info("{}".format(str(meters)))

    model.train()
    return meters.get_all_avg()
        

def display_image_in_actual_size(height, width):

    dpi = mpl.rcParams['figure.dpi']

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')
    return ax
