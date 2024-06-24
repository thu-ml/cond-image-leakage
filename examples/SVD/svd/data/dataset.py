import os, csv, random
import numpy as np
from decord import VideoReader
import jsonlines
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

class RandomHorizontalFlipVideo(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        if torch.rand(1) < self.p:
            return torch.flip(clip, [3])
        return clip

class WebVid10M(Dataset):
    def __init__(
            self,
            file_path, video_folder,
            sample_size=256, fps=6, sample_n_frames=16):
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            self.dataset = [video for video in reader]
        csvfile.close()
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

        self.video_folder = video_folder
        self.fps = fps
        self.sample_n_frames = sample_n_frames
        if isinstance(sample_size, int):
            sample_size = tuple([int(sample_size)] * 2)
        else:
            sample_size = tuple(map(int, sample_size.split(',')))

        self.pixel_transforms = transforms.Compose([
            transforms.Resize(sample_size[0], antialias=True),
            transforms.CenterCrop(sample_size),
            RandomHorizontalFlipVideo(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        videoid = video_dict['videoid']
        video_dir = os.path.join(self.video_folder, f"{videoid}.mp4")
        video_reader = VideoReader(video_dir)

        fps = video_reader.get_avg_fps()
        sample_stride = round(fps/self.fps)

        # sample sample_n_frames frames from videos with stride sample_stride
        video_length = len(video_reader)
        clip_length = min(video_length, (self.sample_n_frames - 1) * sample_stride + 1)
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255. #[T, C, H, W] with range [0, 1]
        del video_reader

        return pixel_values, self.fps, videoid

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, fps , videoid= self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values) #[T, C, H, W] with range [-1, 1]
        sample = dict(pixel_values=pixel_values, fps=fps, id=videoid)
        return sample

class ImageDataset(Dataset):
    def __init__(self, data_path):
        filenames = sorted(os.listdir(data_path))
        self.length = len(filenames)
        self.data_path = data_path
        self.filenames = filenames

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        path = os.path.join(self.data_path, filename)
        sample = dict(path=path, name=filename)
        return sample

class MultiImageDataset(Dataset):
    def __init__(self, data_paths):
        self.paths = []
        for data_path in data_paths:
            filenames = sorted(os.listdir(data_path))
            for filename in filenames:
                path = os.path.join(data_path, filename)
                self.paths.append(path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        dataset_name = path.split('/')[-2]
        filename = path.split('/')[-1]
        sample = dict(path=path, dataset_name=dataset_name, name=filename)
        return sample