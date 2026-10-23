from PIL import Image
from torchvision import transforms
from dataset import RockMassImageDataset

class CachedResizeDataset(RockMassImageDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.transform, transforms.Compose)
        assert isinstance(self.transform.transforms[0], transforms.Resize)
        resize = self.transform.transforms[0]
        self.transform = transforms.Compose(self.transform.transforms[1:])
        self.cache = {}
        for value in self.df['image_path']:
            path = str(self._path(value))
            if path not in self.cache:
                with Image.open(path) as image:
                    self.cache[path] = resize(image.convert('RGB'))
        print(f'Cached {len(self.cache)} deterministic resized RGB images', flush=True)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = str(self._path(row['image_path']))
        image = self.transform(self.cache[path].copy())
        result = {'image': image, 'label': int(row['grade_index']), 'group_id': str(row['group_id']), 'path': path}
        if 'view' in row:
            result['view'] = str(row['view'])
        return result
if __name__ == '__main__':
    import train
    train.RockMassImageDataset = CachedResizeDataset
    train.main()
