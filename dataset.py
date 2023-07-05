from datasets import dtu_loader
from datasets.utils import read_cam_file, read_img, read_depth
from datasets.dataPaths import getImageFile, getCameraFile, getDepthFile
from PIL import Image
import numpy as np


class AttMVSDataset(dtu_loader.MVSDataset):

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views, light_idx = meta

        assert self.args.nsrc <= len(src_views)

        # print("Getting Item:\nscan:"+str(scan)+"\nref_view:"+str(ref_view)+"\nsrc_view:"+str(src_views)+"\nlight_idx"+str(light_idx))

        ref_img = []
        src_imgs = []
        ref_depths = []
        ref_depth_mask = []
        ref_extrinsics = []
        src_extrinsics = []
        depth_min = []
        depth_max = []

        ## 1. Read images
        # ref image
        ref_img_file = getImageFile(self.data_root,self.args.mode,scan,ref_view,light_idx)
        ref_img = read_img(ref_img_file)

        # src image(s)
        for i in range(self.args.nsrc):
            src_img_file = getImageFile(self.data_root,self.args.mode,scan,src_views[i],light_idx)
            src_img = read_img(src_img_file)

            src_imgs.append(src_img)

        ## 2. Read camera parameters
        cam_file = getCameraFile(self.data_root,self.args.mode,ref_view)
        ref_intrinsics, ref_extrinsics, depth_min, depth_max = read_cam_file(cam_file)

        depth_min, depth_max = self.args.depth_min, self.args.depth_max

        # depth_values = np.arange(depth_min, depth_max * self.args.ndepths + depth_min, depth_max, dtype=np.float32)
        depth_values = np.linspace(depth_min, depth_max, self.args.ndepths, endpoint=True, dtype=np.float32)

        for i in range(self.args.nsrc):
            cam_file = getCameraFile(self.data_root,self.args.mode,src_views[i])
            intrinsics, extrinsics, depth_min_tmp, depth_max_tmp = read_cam_file(cam_file)
            src_extrinsics.append(extrinsics)

        ## 3. Read Depth Maps
        if self.args.mode == "train":
            imgsize = self.args.imgsize
            nscale = self.args.nscale

            # Read depth map of same size as input image first.
            depth_file = getDepthFile(self.data_root,self.args.mode,scan,ref_view)
            ref_depth = read_depth(depth_file)
            depth_frame_size = (ref_depth.shape[0],ref_depth.shape[1])
            frame = np.zeros(depth_frame_size)
            frame[:ref_depth.shape[0],:ref_depth.shape[1]] = ref_depth
            ref_depths.append(frame)

            # Downsample the depth for each scale.
            ref_depth = Image.fromarray(ref_depth)
            original_size = np.array(ref_depth.size).astype(int)

            for scale in range(1, nscale):
                new_size = (original_size/(2**scale)).astype(int)
                down_depth = ref_depth.resize((new_size),Image.BICUBIC)
                frame = np.zeros(depth_frame_size)
                down_np_depth = np.array(down_depth)
                frame[:down_np_depth.shape[0],:down_np_depth.shape[1]] = down_np_depth
                ref_depths.append(frame)

        # Orgnize output and return
        sample = {}
        sample["filename"] = "rect_"+str(ref_view+1).zfill(3)+"_"+str(light_idx)+"_r5000"
        sample["ref_img"] = np.moveaxis(np.array(ref_img),2,0)
        sample["src_imgs"] = np.moveaxis(np.array(src_imgs),3,1)
        sample["ref_extrinsics"] = np.array(ref_extrinsics)
        sample["src_extrinsics"] = np.array(src_extrinsics)
        sample["depth_values"] = depth_values
        sample["depth_min"] = depth_min
        sample["depth_max"] = depth_max

        if self.args.mode == "train":
            sample["ref_depths"] = np.array(ref_depths,dtype=float)
            sample["ref_depth_mask"] = np.array(ref_depth_mask)
        elif self.args.mode == "test":
            sample["filename"] = scan + '/{}/' + '{:0>8}'.format(ref_view) + "{}"

        return sample