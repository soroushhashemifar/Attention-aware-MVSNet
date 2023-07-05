import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):

    def __init__(self, input_channels):
        super(FeatureExtractor, self).__init__()

        self.conv0_0 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.01)
        )
        self.conv0_1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.01)
        )
        self.conv0_2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.01)
        )

        self.conv1_0 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.01)
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.01)
        )
        self.conv1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.conv2_0 = nn.Sequential(
            nn.Conv2d(64, 64, 5, stride=2, padding=2),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.01)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.01)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.01)
        )
        self.conv2_3 = nn.Conv2d(64, 16, 1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv0_0(x)
        x = self.conv0_1(x)
        x = self.conv0_2(x)

        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)

        x_skip = self.conv2_0(x)
        x = self.conv2_1(x_skip)
        x = self.conv2_2(x)
        x = x + x_skip
        x = self.conv2_3(x)

        return x


class SE_Block(nn.Module):

    "source: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"

    def __init__(self, c, r=16):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x)
        y = y.view(bs, c)
        y = self.excitation(y)
        y = y.view(bs, c, 1, 1)
        
        return x * y.expand_as(x)


def homography_warping(src_fea, src_proj, ref_proj, depth_values):
    # source: https://github.com/xy-guo/MVSNet_pytorch

    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]

        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]

        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


class SimpleRFM(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(SimpleRFM, self).__init__()

        # precontext
        self.block1 = nn.Sequential(
            nn.Conv3d(input_channels, input_channels, 3, stride=1, padding=1),
            nn.BatchNorm3d(input_channels),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(input_channels, input_channels*2, 3, stride=1, padding=1),
            nn.BatchNorm3d(input_channels*2),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(input_channels*2, input_channels*2, 3, stride=1, padding=1),
            nn.BatchNorm3d(input_channels*2),
            nn.ReLU()
        )

        # postcontext
        self.block4 = nn.Sequential(
            nn.Conv3d(input_channels*2, input_channels*2, 3, stride=1, padding=1),
            nn.BatchNorm3d(input_channels*2),
            nn.ReLU()
        )
        self.block5 = nn.Sequential(
            nn.Conv3d(input_channels*2, output_channels, 3, stride=1, padding=1),
            nn.BatchNorm3d(output_channels),
            nn.ReLU()
        )
        self.block6 = nn.Sequential(
            nn.Conv3d(output_channels, output_channels, 3, stride=1, padding=1),
            nn.BatchNorm3d(output_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block1(x)
        x2 = self.block2(x)
        x = self.block3(x2)

        x = self.block4(x)
        x = x + x2
        x = self.block5(x)
        x = self.block6(x)

        return x


class RFM(nn.Module):

    def __init__(self, input_channels, output_channels, r=2):
        super(RFM, self).__init__()

        # precontext
        self.block1 = nn.Sequential(
            nn.Conv3d(input_channels, input_channels, 3, stride=1, padding=1),
            nn.BatchNorm3d(input_channels),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(input_channels, input_channels*2, 3, stride=2, padding=1),
            nn.BatchNorm3d(input_channels*2),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(input_channels*2, input_channels*2, 3, stride=1, padding=1),
            nn.BatchNorm3d(input_channels*2),
            nn.ReLU()
        )

        self.squeeze_3D = nn.AdaptiveAvgPool3d(1)
        self.excitation_3D = nn.Sequential(
            nn.Linear(input_channels*2, input_channels*2 // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_channels*2 // r, input_channels*2, bias=False),
            nn.Sigmoid()
        )

        # postcontext
        self.block4 = nn.Sequential(
            nn.Conv3d(input_channels*2, input_channels*2, 3, stride=1, padding=1),
            nn.BatchNorm3d(input_channels*2),
            nn.ReLU()
        )
        self.block5 = nn.Sequential(
            nn.Conv3d(input_channels*2, output_channels, 3, stride=1, padding=1),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.block6 = nn.Sequential(
            nn.Conv3d(output_channels, output_channels, 3, stride=1, padding=1),
            nn.BatchNorm3d(output_channels),
            nn.ReLU()
        )

    def forward(self, r_prime_l, r_l):
        bs, *_ = r_prime_l.shape

        x = self.block1(r_prime_l)
        x = self.block2(x)
        r_e = self.block3(x)

        x = torch.abs(r_e - r_l)

        bs, c, _, _, _ = x.shape
        y = self.squeeze_3D(x)
        y = y.view(bs, c)
        y = self.excitation_3D(y)
        y = y.view(bs, c, 1, 1, 1)
        x = x * y.expand_as(x)

        x = x * r_e
        r_star_l = x + r_l

        x = self.block4(r_star_l)
        x = self.block5(x)
        x = self.block6(x)

        return x


class AttentionGuidedRegularization(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(AttentionGuidedRegularization, self).__init__()

        self.R_prime_0 = nn.Sequential(
            nn.Conv3d(input_channels, input_channels//2, 3, stride=1, padding=1),
            nn.BatchNorm3d(input_channels//2),
            nn.ReLU()
        )
        self.R_prime_1 = nn.Sequential(
            nn.Conv3d(input_channels, input_channels//4, 3, stride=2, padding=1),
            nn.BatchNorm3d(input_channels//4),
            nn.ReLU()
        )

        self.max_pool = nn.MaxPool3d(2, stride=2)

        self.rfm0 = RFM(input_channels//2, output_channels)
        self.rfm1 = RFM(input_channels//4, input_channels)
        self.rfm2 = RFM(input_channels//4, input_channels//2)
        self.simple_rfm = SimpleRFM(input_channels//4, input_channels//2)

    def forward(self, x):
        r_prime_0 = self.R_prime_0(x)
        r_prime_1 = self.R_prime_1(x)
        r_prime_2 = self.max_pool(r_prime_1)
        r_prime_3 = self.max_pool(r_prime_2)

        r_3 = self.simple_rfm(r_prime_3)
        r_2 = self.rfm2(r_prime_2, r_3)
        r_1 = self.rfm1(r_prime_1, r_2)
        r_0 = self.rfm0(r_prime_0, r_1)

        return r_0


# p: probability volume [B, D, H, W]
# depth_values: discrete depth values [B, D]
def depth_regression(p, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth


class AttMVSNet(nn.Module):

    def __init__(self, args):
        super(AttMVSNet, self).__init__()

        self.args = args

        self.feature_extractor = FeatureExtractor(3)
        self.global_avg_pool = nn.AvgPool2d(kernel_size=(32, 40))
        self.squeeze_and_excitation = SE_Block(16, 4)
        self.fagr = AttentionGuidedRegularization(32, 32)

        self.regression_head = nn.Sequential(
            nn.Upsample(scale_factor=(1, 4, 4)),
            nn.Conv3d(32, 1, 3, stride=1, padding=1),
        )

        self.encoder = nn.Sequential(
            nn.Upsample(scale_factor=(1, 4, 4)),
            nn.Conv3d(32, 1, 3, stride=1, padding=1),
        )

    def forward(self, ref_img, src_imgs, ref_ex, src_ex, depth_values):
        depth_est_list = []
        output = {}

        ref_feature = self.feature_extractor(ref_img)

        total_features = []
        for i in range(self.args.nsrc):
            total_features.append(self.feature_extractor(src_imgs[:, i, :, :, :]))

        total_features.append(ref_feature)

        gpooled_features = []
        for i in range(len(total_features)):
            feature = self.global_avg_pool(total_features[i])
            feature = feature.unsqueeze(1)
            gpooled_features.append(feature)

        gpooled_features = torch.concat(gpooled_features, dim=1) # b, N+1, 16, 1, 1
        v_avg = gpooled_features.mean(2, keepdim=True) # b, N+1, 1, 1, 1
        w_v = ((gpooled_features - v_avg) ** 2).sum(1) / len(total_features) # b, 16, 1, 1
        w_v_star = self.squeeze_and_excitation(w_v) # b, 16, 1, 1
        w_v_star = w_v_star.unsqueeze(2) # b, 16, 1, 1, 1

        M_stars = []
        for src_idx in range(self.args.nsrc):
            M = homography_warping(total_features[src_idx], src_proj=src_ex[:, src_idx], ref_proj=ref_ex, depth_values=depth_values)
            M_star = M * w_v_star # b, 16, Z, W, H
            M_stars.append(M_star)

        M_stars = torch.concat(M_stars, 1) # b, 32, Z, W, H

        regularized = self.fagr(M_stars)
        # print(regularized.shape) # torch.Size([b, 32, Z, 32, 40])

        prob_volume = self.regression_head(regularized)
        prob_volume = prob_volume[:, 0]
        prob_volume = F.softmax(prob_volume, dim=1)
        # print("regression_head", prob_volume.shape) # torch.Size([b, 1, Z, 128, 160])

        depth = depth_regression(prob_volume, depth_values=depth_values)
        # print("depth", depth.shape) # torch.Size([b, 128, 160])
        depth_est_list.append(depth)

        output["depth_est_list"] = depth_est_list

        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(self.args.ndepths, device=prob_volume.device, dtype=torch.float)).long()
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        output["photometric_confidence"] = photometric_confidence

        return output