import torch
import torch.nn as nn
from einops.einops import rearrange

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module.transformer_third import LocalFeatureTransformer
from .loftr_module.fine_preprocess_third import FinePreprocess
from .utils.coarse_matching_third import CoarseMatching
from .utils.fine_matching import FineMatching

from superglue_models.superglue import KeypointEncoder, AttentionalGNN

from .MultiScaleLocalFeature import MultiScaleLocalFeature
from .HardNet import HardNet


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class LoFTR(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 40,
        'match_threshold': 0.2
    }

    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules

        # SuperGlue
        self.kenc = KeypointEncoder(
            256, [32, 64, 128, 256])

        # self.gnn = AttentionalGNN(
        #     256, self.['self', 'cross'] * 9)

        # self.backbone = build_backbone(config)  # ResNetFPN_8_2
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        coarse = True  # Coarse: True  Fine: False
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'], coarse)
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"], not coarse)
        self.fine_matching = FineMatching()

        # MultiScaleLocalFeature
        # self.local_feat = MultiScaleLocalFeature(dim=256)

        # HardNet
        # self.hardnet = HardNet()

    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        # 1/8 , 1/2
        # if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
        #     feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
        #     (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])
        # else:  # handle different input shapes
        #     (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])

        feat_c0, feat_f0 = data['feat_c0'], data['feat_f0']
        feat_c1, feat_f1 = data['feat_c1'], data['feat_f1']
        # feats_c = torch.cat((data['feat_c0'], data['feat_c1']), dim=0)
        # feats_f = torch.cat((data['feat_f0'], data['feat_f1']), dim=0)

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # SuperGlue
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        scores0, scores1 = data['scores0'].unsqueeze(1), data['scores1'].unsqueeze(1)

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }
        # kpts0_o, kpts1_o = kpts0, kpts1
        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape[-2:])
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape[-2:])
        kpts0, kpts1 = kpts0.transpose(1, 2), kpts1.transpose(1, 2)

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, scores0)
        desc1 = desc1 + self.kenc(kpts1, scores1)

        # Multi-layer Transformer network.
        # desc0, desc1 = self.gnn(desc0, desc1)

        # HardNet
        # patch_feat0, patch_feat1 = self.hardnet(data['image0']), self.hardnet(data['image1'])
        # patch_feat0 = rearrange(patch_feat0, 'n c h w -> n (h w) c')
        # patch_feat1 = rearrange(patch_feat1, 'n c h w -> n (h w) c')

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')
        desc0 = rearrange(desc0, 'b c n-> b n c')
        desc1 = rearrange(desc1, 'b c n-> b n c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)

        # feat_c0 = torch.cat((feat_c0, patch_feat0), dim=2)
        # feat_c1 = torch.cat((feat_c1, patch_feat1), dim=2)

        # output: 1*6400*256 (1/8)
        # feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        feat_c0, feat_c1, desc0, desc1 = self.loftr_coarse(
            feat_c0, feat_c1, desc0, desc1, mode='coarse',
            mask0=mask_c0, mask1=mask_c1
        )

        # lofet&HardNet
        # feat_c0 = torch.cat((feat_c0, patch_feat0), dim=2)
        # feat_c1 = torch.cat((feat_c1, patch_feat1), dim=2)

        # 3. match coarse-level
        # self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)
        self.coarse_matching(
            feat_c0, feat_c1, data, desc0, desc1,
            mask_c0=mask_c0, mask_c1=mask_c1
        )

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
