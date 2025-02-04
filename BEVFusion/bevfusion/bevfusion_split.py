from collections import OrderedDict
from copy import deepcopy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization

@MODELS.register_module()
class BEVFusion_split(Base3DDetector):

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,

        image_encoder: Optional[dict] = None,
        lidar_encoder: Optional[dict] = None,
        fuser_neck: Optional[dict] = None,
        fuser_decoder: Optional[dict] = None,
        use_entropy:int = None,
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        #stuff for the orginal bevfusion model

        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)

        self.pts_voxel_encoder = MODELS.build(
            pts_voxel_encoder) if pts_voxel_encoder is not None else None

        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(
            view_transform) if view_transform is not None else None
        self.pts_middle_encoder = MODELS.build(
            pts_middle_encoder) if pts_middle_encoder is not None else None

        self.fusion_layer = MODELS.build(
            fusion_layer) if fusion_layer is not None else None

        #stuff for split bevfusion WE SHOULD ALWAYS HAVE THESE
        self.image_encoder = MODELS.build(image_encoder) if image_encoder is not None else None
        self.lidar_encoder = MODELS.build(lidar_encoder) if lidar_encoder is not None else None
        self.fuser_neck = MODELS.build(fuser_neck) if fuser_neck is not None else None
        self.fuser_decoder = MODELS.build(fuser_decoder) if fuser_decoder is not None else None

        # currently I am having a hardcoded global variable to  
        # determine training/testing stages, for speeeeeeeeeeeed
        # self.is_training = True
        self.training_stage = 1 #0 or 1

        self.use_entropy = use_entropy
        if use_entropy is not None:
            self.use_entropy = use_entropy
            self.fuser_neck.stage = use_entropy

        #this portion is also from the orginal bevfusion but will be our tail

        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)

        self.bbox_head = MODELS.build(bbox_head)

        self.init_weights()

        #freeze some parameters for training
        if self.pts_voxel_layer is not None:
            for param in self.pts_voxel_layer.parameters():
                    param.requires_grad = False
        if self.pts_voxel_encoder is not None:
            for param in self.pts_voxel_encoder.parameters():
                    param.requires_grad = False
        if self.img_backbone is not None:
            for param in self.img_backbone.parameters():
                    param.requires_grad = False
        if self.img_neck is not None:
            for param in self.img_neck.parameters():
                    param.requires_grad = False
        if self.pts_voxel_encoder is not None:
            for param in self.pts_voxel_encoder.parameters():
                    param.requires_grad = False
        if self.view_transform is not None:
            for param in self.view_transform.parameters():
                    param.requires_grad = False
        if self.fusion_layer is not None:
            for param in self.fusion_layer.parameters():
                    param.requires_grad = False
        if self.pts_middle_encoder is not None:
            for param in self.pts_middle_encoder.parameters():
                    param.requires_grad = False

        if self.training_stage == 0:
            self.s1_freeze()
        else:
            self.s2_freeze()

        # if self.pts_middle_encoder is not None:
        #      print(self.pts_middle_encoder)
        # if self.img_backbone is not None:
        #      print(self.img_backbone)
        if self.img_neck is not None:
             print(self.img_neck)
        if self.view_transform is not None:
             print(self.view_transform)

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def s1_freeze(self):
        if self.pts_backbone is not None:
            for param in self.pts_backbone.parameters():
                    param.requires_grad = False
        if self.pts_neck is not None:
            for param in self.pts_neck.parameters():
                    param.requires_grad = False
        if self.bbox_head is not None:
            for param in self.bbox_head.parameters():
                    param.requires_grad = False

    def s2_freeze(self):
        if self.image_encoder is not None:
            for param in self.image_encoder.parameters():
                    param.requires_grad = False
        if self.lidar_encoder is not None:
            for param in self.lidar_encoder.parameters():
                    param.requires_grad = False
        if self.fuser_neck is not None:
            for param in self.fuser_neck.parameters():
                    param.requires_grad = False
        if self.fuser_decoder is not None:
            for param in self.fuser_decoder.parameters():
                    param.requires_grad = False

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def split_extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.image_encoder(x)

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        return x

    @torch.no_grad()
    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:    
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )

        return x

    def split_extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        #currently expecting lidar lists to be around ~287496 in points so
        #I will hardcode for that a the moment 

        #convert all point lists into a batched tensor
        # points = batch_inputs_dict['points']
        # with torch.autocast('cuda', enabled=False):
        #     points = [point.float() for point in points]
        #     for i in range(len(points)):
        #         if points[i].shape[0] <= 280000:
        #             points[i] = F.pad(points[i], (0,0,0,(280000%points[i].shape[0])), value=0)
        #         else:
        #             points[i] = points[i][:280000]

        #     points = torch.stack(points)
        #     points = points.reshape(points.shape[0], 5, -1, points.shape[1])
        #     # print(points.shape)

        '''Do the voxelization but we will use our own custom bevsparseencoder'''
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = coords[-1, 0] + 1

        x = self.lidar_encoder(feats, coords, batch_size)#(points)
        return x

    @torch.no_grad()
    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = coords[-1, 0] + 1
        x = self.pts_middle_encoder(feats, coords, batch_size)
        # print(x.shape)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    #TODO:overide this
    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        _, _, feats, _ = self.split_extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

    def split_extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature = self.split_extract_img_feat(imgs, deepcopy(points),
            # img_feature = self.extract_img_feat(imgs, deepcopy(points),   
                                                lidar2image, camera_intrinsics,
                                                camera2lidar, img_aug_matrix,
                                                lidar_aug_matrix,
                                                batch_input_metas)
            features.append(img_feature)
        pts_feature = self.split_extract_pts_feat(batch_inputs_dict)
        features.append(pts_feature)

        # decoder_out = pts_feature
        if self.use_entropy is not None:
            x, likelihoods = self.fuser_neck(features)
        else:
            x = self.fuser_neck(features)
        decoder_out = self.fuser_decoder(x)

        if self.training_stage == 0:
            x = likelihoods

        # if self.fusion_layer is not None:
        #     x = self.fusion_layer(features)
        # else:
        #     assert len(features) == 1, features
        #     x = features[0]
        # decoder_out = x

        x_backbone = self.pts_backbone(decoder_out) #(x)
        x_fpn = self.pts_neck(x_backbone)

        return decoder_out, x_backbone, x_fpn, x

    @torch.no_grad()
    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature = self.extract_img_feat(imgs, deepcopy(points),
                                                lidar2image, camera_intrinsics,
                                                camera2lidar, img_aug_matrix,
                                                lidar_aug_matrix,
                                                batch_input_metas)
            features.append(img_feature)
        pts_feature = self.extract_pts_feat(batch_inputs_dict)
        features.append(pts_feature)

        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        reconstruction_target = x #pts_feature #

        x_backbone = self.pts_backbone(x)
        x_fpn = self.pts_neck(x_backbone)

        return reconstruction_target, x_backbone, x_fpn, x 

    #TODO: override this
    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        losses = None

        if self.training_stage == 0:
            reconstruction_target, backbone_original, fpn_original, x_og = self.extract_feat(batch_inputs_dict, batch_input_metas)
            decoder_out, backbone_new, fpn_new, x_new = self.split_extract_feat(batch_inputs_dict, batch_input_metas)
            losses = {}

            loss = F.mse_loss(decoder_out, reconstruction_target)
            losses['recon_loss'] = loss

            # fusion_loss = F.mse_loss(x_new,x_og)
            # losses['fusion_recon_loss'] = fusion_loss

            if self.use_entropy is not None:
                #x_new should be logits in this case
                num_pixels = 4 * 50 * 50 * 50

                bpp_loss = (0.01*(torch.log(x_new).sum() / (-math.log(2) * num_pixels)))
                aux_loss = self.fuser_neck.entropy_bb.loss()
                losses['bpp_loss'] = bpp_loss
                losses['aux_loss'] = aux_loss

            backbone_loss = 0
            for i in range(len(backbone_original)):
                backbone_loss += (F.mse_loss(backbone_new[i], backbone_original[i]))/len(backbone_original)
            losses['backbone_recon_loss'] = backbone_loss

            # fpn_loss = 0
            # for i in range(len(fpn_original)):
            #     fpn_loss += (F.mse_loss(fpn_new[i], fpn_original[i]))/len(fpn_original)
            # losses['fpn_recon_loss'] = fpn_loss

        elif self.training_stage >=1:
            decoder_out, x_backbone, feats, x_new = self.split_extract_feat(batch_inputs_dict, batch_input_metas)
            losses = dict()

            if self.with_bbox_head:
                bbox_loss = self.bbox_head.loss(feats, batch_data_samples)

            losses.update(bbox_loss)

        return losses
