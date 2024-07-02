import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.models import lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class DFRscModel(SRModel):
    @torch.no_grad()
    def feed_data(self, data):
        self.gt = data["gt_frames"].flatten(0, 1).to(self.device)
        self.lq = data["input_frames"].to(self.device)

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == "CosineAnnealingLR":
            for optimizer in self.optimizers:
                self.schedulers.append(CosineAnnealingLR(optimizer, **train_opt['scheduler']))

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        # NOTE: output: [pred_x0, [flow], [intermediates]]
        out_dict['result'] = self.output[0].detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def backward_warp(self, img, flow):
        B, _, H, W = flow.shape
        xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
        yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
        grid = torch.cat([xx, yy], 1).to(img)
        flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
        grid_ = (grid + flow_).permute(0, 2, 3, 1)
        img = torch.cat([img, torch.ones_like(img[:, :1, :, :])], 1)
        output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
        mask = output[:, -1:, :, :]
        mask[mask > 0.999] = 1.0
        mask[mask < 1.0] = 0.0
        output = output[:, :-1, :, :]
        return output, mask  # (B, C, H, W), (B, 1, H, W)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # GS image supervision
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output[0], self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output[0], self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        # distortion flow supervision
        flow_t_pred = self.output[1]  # (B, N, 2, H_, W_)
        if not isinstance(flow_t_pred, list):
            flow_t_pred = [flow_t_pred]
        # RS + flow -> GS
        loss_flow = 0.
        for i in range(self.lq.shape[1]):
            for flow in flow_t_pred:
                scale_factor = flow.shape[-1] / self.gt.shape[-1]
                rs_img = F.interpolate(self.lq[:, i, :3], scale_factor=scale_factor, mode="bilinear", align_corners=False)
                gs_img = F.interpolate(self.gt, scale_factor=scale_factor, mode="bilinear", align_corners=False)
                rs_img_warped, mask = self.backward_warp(rs_img, flow[:, i])
                loss_flow += self.cri_pix(rs_img_warped * mask, gs_img * mask)

        flow_weight = self.opt.get("loss_flow_weight", 0.05)
        loss_flow *= flow_weight

        loss_dict["l_flow"] = loss_flow
        l_total += loss_flow

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['gt_names'][0][0]))[0]
            video_name = val_data.get("video_name", None)[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], f"{current_iter}", video_name, f'{img_name}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, video_name,
                                                 f'{img_name}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
