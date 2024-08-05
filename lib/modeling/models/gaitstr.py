import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper

from .utils.tgcn import ConvTemporalGraphical 
from .utils.graph import Graph
from torch.autograd import Variable


class CombineRevModel(nn.Module):
    def __init__(self, rev_model1, rev_model2):
        super(CombineRevModel, self).__init__()
        self.rev_model1 = rev_model1
        self.rev_model2 = rev_model2
    
    def forward(self, x1, x2):
        inputs = []
        for x in [x1, x2]:
            N, C, T, V = x.size()
            x = x.permute(0, 3, 1, 2).contiguous()
            x = x.view(N, V * C, T)
            x = x.view(N, V, C, T)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(N, C, T, V)
            inputs.append(x)
        x1, x2 = inputs

        for i, (gcn1, gcn2, importance1, importance2) in enumerate(zip(self.rev_model1.st_gcn_networks, self.rev_model2.st_gcn_networks, self.rev_model1.edge_importance, self.rev_model2.edge_importance)):
            # If it is a middle layer, exchange information between the models
            if i < len(self.rev_model1.st_gcn_networks):
                # Forward through the first model's layer
                x1, _ = gcn1(x1, self.rev_model1.A * importance1)
                # Forward through the second model's layer
                x2, _ = gcn2(x2, self.rev_model2.A * importance2)

                # Extra trainable parameters for extrange information between two gcns
                extra_res1 = gcn1.extra_info_residual(x1)
                extra_res2 = gcn2.extra_info_residual(x2)

                x1 = x1 + extra_res2
                x2 = x2 + extra_res1
        
        return x1, x2


class RevModel(nn.Module):
    def __init__(self, in_channels, graph_args, edge_importance_weighting, extra_info_residual=0, **kwargs):
        super().__init__()
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(8322, 128, kernel_size, 1, extra_info_residual=extra_info_residual, **kwargs),
            st_gcn(128, 64, kernel_size, 1, extra_info_residual=extra_info_residual, **kwargs),
            st_gcn(64, 64, kernel_size, 1, extra_info_residual=extra_info_residual, **kwargs),
            st_gcn(64, in_channels, kernel_size, 1, residual=False, **kwargs0),
        ))
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)


class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction

    def forward(self, x):

        # data normalization
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N , V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N , C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        return x


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
        extra_info_residual (int, optional): If 1 or 2, applies a residual mechanism of extra information. Default: 0
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True,
                 extra_info_residual=0):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        if not extra_info_residual:
            self.extra_info_residual = lambda x: 0

        elif extra_info_residual == 1:
            self.extra_info_residual = nn.Sequential(
                nn.ConstantPad2d((0, 1, 0, 0), 0),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        elif extra_info_residual == 2:
            self.extra_info_residual = nn.Sequential(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=(1, 2),
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        elif extra_info_residual == 3:
            self.extra_info_residual = nn.Linear(17, 18)

        elif extra_info_residual == 4:
            self.extra_info_residual = nn.Linear(18, 17)

        elif extra_info_residual == 5:
            self.extra_info_residual = nn.Sequential(
                nn.ConstantPad2d((0, 1, 0, 0), 0),
                nn.Conv2d(
                    out_channels,
                    out_channels // 2,  # Reduce channels in the first layer
                    kernel_size=(1, 1),
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels // 2),  # BatchNorm for the first layer
                nn.Conv2d(
                    out_channels // 2,  # Use the same reduced channels for the second layer
                    out_channels,  # Restore the original number of channels
                    kernel_size=(1, 1),
                    stride=(1, 1)),  # No additional stride for the second layer
                nn.BatchNorm2d(out_channels),  # BatchNorm for the second layer
            )

        elif extra_info_residual == 6:
            self.extra_info_residual = nn.Sequential(
                nn.Conv2d(
                    out_channels,
                    out_channels // 2,  # Reduce channels in the first layer
                    kernel_size=(1, 2),
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels // 2),  # BatchNorm for the first layer
                nn.Conv2d(
                    out_channels // 2,  # Use the same reduced channels for the second layer
                    out_channels,  # Restore the original number of channels
                    kernel_size=(1, 1),
                    stride=(1, 1)),  # No additional stride for the second layer
                nn.BatchNorm2d(out_channels),  # BatchNorm for the second layer
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


class GLConv(nn.Module):
    def __init__(self, in_channels, out_channels, halving, fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(GLConv, self).__init__()
        self.halving = halving
        self.fm_sign = fm_sign
        self.global_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
        self.local_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        gob_feat = self.global_conv3d(x)
        if self.halving == 0:
            lcl_feat = self.local_conv3d(x)
        else:
            h = x.size(3)
            split_size = int(h // 2**self.halving)
            lcl_feat = x.split(split_size, 3)
            lcl_feat = torch.cat([self.local_conv3d(_) for _ in lcl_feat], 3)

        if not self.fm_sign:
            feat = F.leaky_relu(gob_feat) + F.leaky_relu(lcl_feat)
        else:
            feat = F.leaky_relu(torch.cat([gob_feat, lcl_feat], dim=3))
        return feat


class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1)*p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)


class GaitSTR(BaseModel):
    """
        Title: Gait Recognition via Effective Global-Local Feature Representation and Local Temporal Aggregation
        ICCV2021: https://openaccess.thecvf.com/content/ICCV2021/papers/Lin_Gait_Recognition_via_Effective_Global-Local_Feature_Representation_and_Local_Temporal_ICCV_2021_paper.pdf
    """

    def __init__(self, *args, **kargs):
        super(GaitSTR, self).__init__(*args, **kargs)

    def build_network(self, model_cfg):
        in_c = model_cfg['channels']
        class_num = model_cfg['class_num']

        # For CASIA-B
        self.conv3d = nn.Sequential(
            BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )
        self.LTA = nn.Sequential(
            BasicConv3d(in_c[0], in_c[0], kernel_size=(
                3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(inplace=True)
        )

        self.GLConvA0 = GLConv(in_c[0], in_c[1], halving=3, fm_sign=False, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.MaxPool0 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.GLConvA1 = GLConv(in_c[1], in_c[2], halving=3, fm_sign=False, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.GLConvB2 = GLConv(in_c[2], in_c[2], halving=3, fm_sign=True,  kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.Head0 = SeparateFCs(67, in_c[2], in_c[2])
        self.Bn = nn.BatchNorm1d(in_c[2])
        self.Head1 = SeparateFCs(67, in_c[2], class_num)

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = GeMHPP()
        self.network = Model(2, {'layout': 'casia-b', 'strategy': 'spatial'}, edge_importance_weighting=True)
        self.bone_network = Model(2, {'layout': 'casia-b_bone', 'strategy': 'uniform'}, edge_importance_weighting=True)
        self.denetwork = RevModel(2, {'layout': 'casia-b', 'strategy': 'spatial'}, edge_importance_weighting=True, extra_info_residual=model_cfg['skeleton_extra_residual'])
        self.bone_denetwork = RevModel(2, {'layout': 'casia-b_bone', 'strategy': 'uniform'}, edge_importance_weighting=True, extra_info_residual=model_cfg['bone_extra_residual'])
        self.combine_denetwork = CombineRevModel(self.denetwork, self.bone_denetwork)

    def forward(self, inputs):
        ipts, pose, bone, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL

        sils = ipts[0].unsqueeze(1)
        del ipts
        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)

        outs = self.conv3d(sils)
        outs = self.LTA(outs)

        outs = self.GLConvA0(outs)
        outs = self.MaxPool0(outs)

        outs = self.GLConvA1(outs)
        outs = self.GLConvB2(outs)  # [n, c, s, h, w]

        outs = self.TP(outs, dim=2, seq_dim=2, seqL=seqL)[0]  # [n, c, h, w]
        outs = self.HPP(outs)  # [n, c, p]
        outs = outs.permute(2, 0, 1).contiguous()  # [p, n, c]

        gembs = outs.view([n, -1])

        pembs = self.network(pose.squeeze(1)) # n, c, t, v
        _, _, t, v = pembs.shape
        p = torch.cat([pembs, gembs[...,None,None].repeat(1,1,t,v), pose.squeeze(1)], axis=1)

        bembs = self.bone_network(bone.squeeze(1))
        _, _, t, v = bembs.shape
        b = torch.cat([bembs, gembs[...,None,None].repeat(1,1,t,v), bone.squeeze(1)], axis=1)

        repose, rebone = self.combine_denetwork(p, b)

        refpembs = self.network(pose.squeeze(1) + repose).mean([-1, -2])[None, ...]
        refbembs = self.bone_network(bone.squeeze(1) + rebone).mean([-1, -2])[None, ...]

        outs = torch.cat([outs, pembs.mean([-1,-2])[None, ...], refpembs, refbembs], axis=0)

        gait = self.Head0(outs)  # [p, n, c]
        gait = gait.permute(1, 2, 0).contiguous()  # [n, c, p]
        bnft = self.Bn(gait)  # [n, c, p]
        logi = self.Head1(bnft.permute(2, 0, 1).contiguous())  # [p, n, c]

        gait = gait.permute(0, 2, 1).contiguous()  # [n, p, c]
        bnft = bnft.permute(0, 2, 1).contiguous()  # [n, p, c]
        logi = logi.permute(1, 0, 2).contiguous()  # [n, p, c]

        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': bnft, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': bnft,
            }
        }
        return retval
