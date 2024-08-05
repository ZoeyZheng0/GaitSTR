import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from glob import glob


class Skeleton:
    def __init__(self, fp):
        self.fp = fp
        self.read()
        self.neighbor_link = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
                              (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
                              (11, 13), (13, 15), (12, 14), (14, 16)]

    def read(self):
        print(f'start read {self.fp}')
        self.files = glob(f"{self.fp}/**/*.npy", recursive=True)

    def multi_run(self, video_p):
        _p = np.load(video_p)
        _o = self.compute_orientation(_p)
        # print(_p[:, 0, :], _o[:, 0, :])
        np.save(video_p[:-4] + '_orientation.npy', _o)

    def compute_orientation(self, _p):
        frames = []
        for frame in range(_p.shape[1]):
            orientations = []
            for link in self.neighbor_link:
                start, end = link
                direction = _p[:, frame, end] - _p[:, frame, start]
                orientations.append(direction)
            frames.append(np.stack(orientations, axis=-1))
        _o = np.stack(frames, axis=1)
        return _o


if __name__ == '__main__':
    fp = 'CASIA-B-mix'
    skl = Skeleton(fp)

    processing_num = 12
    max_ = len(skl.files)
    print(f'Total videos: {max_}')
    chunk = np.array_split(skl.files, max_ // 12)
    with tqdm(total=max_) as pbar:
        for c in chunk:
            with Pool(processes=processing_num) as p:
                for _ in p.imap_unordered(skl.multi_run, c):
                    pbar.update()
