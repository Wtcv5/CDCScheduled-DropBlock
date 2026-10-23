from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
REQUIRED_COLUMNS = {'image_path', 'group_id', 'grade_index', 'grade_text'}

class RockMassImageDataset(Dataset):

    def __init__(self, csv_path, root=None, transform=None, fold=None, train=False):
        self.df = pd.read_csv(csv_path)
        missing = REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(f'dataset CSV missing columns: {missing}')
        if fold is not None:
            if 'fold' not in self.df.columns:
                raise ValueError('fold requested but CSV has no fold column')
            if train:
                self.df = self.df[self.df['fold'] != fold].reset_index(drop=True)
            else:
                self.df = self.df[self.df['fold'] == fold].reset_index(drop=True)
        self.root = Path(root) if root else None
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _path(self, p):
        p = Path(str(p))
        if p.is_absolute():
            return p
        if self.root is None:
            return p
        return self.root / p

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self._path(row['image_path'])
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        out = {'image': image, 'label': int(row['grade_index']), 'group_id': str(row['group_id']), 'path': str(path)}
        if 'view' in row:
            out['view'] = str(row['view'])
        return out
