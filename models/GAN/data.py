import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, List, Any
import torch
import config
from PIL import Image
import PIL


class PokemonData(Dataset):
    """A custom Dataset class for our Pokemon Images Data. Loads and augments our data

    Args:
        Dataset (torch.utils.data.Dataset): A Dataset object, conitaining all of our Pokemon images
    """
    def __init__(self,
                 targ_dir: str,
                 classes_df_dir: str,
                 transform=None):
        super().__init__()
        self.paths = list(Path(targ_dir).glob('*.png'))
        self.transform = transform
        self.classes, self.classes_to_idx = self.get_classes(classes_df_dir)
        
        
    def get_classes(self, df_dir: str):
        """Get classnames as a list and as a tuple of id: class_name

        Returns:
            List[str], Tuple[int: str]: A list of class_names and a tuple of id to class_name
        """
        # We Load up and read pokemon classnames from our csv
        df = pd.read_csv(df_dir)
        df = df.drop_duplicates(subset=["#"],keep="first")
        df = df.set_index(["#"], drop=True)
        classes = df["Name"].unique()
        classes_to_idx = {class_name: i for i, class_name in enumerate(classes)}
        return classes, classes_to_idx
        
        
    def load_image(self, index: int) -> PIL.Image:
        """Loads up an image by index, then fills in blank png spots with white

        Args:
            index (int): index of image to load and return

        Returns:
            PIL.Image: An pokemon image object, loaded by index
        """
        image_path = self.paths[index]
        image = Image.open(image_path).convert("RGBA")
        return Image.composite(image, Image.new('RGBA', image.size, 'white'), image).convert("RGB")
    
    
    def __len__(self) -> int:
        """Returns the lenght of our Pokemon image dataset

        Returns:
            int: lenght of a dataset
        """
        return len(self.paths)
    
    
    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """We get an item from our Dataset and assign it a class from image file's name

        Args:
            index (int): index of image to load

        Returns:
            Tuple[Any, int]: A tuple of a tranformed image and its class
        """
        img = self.load_image(index)
        # img_name = self.paths[index].name.split('.')[0]
        img_name = 3
        class_name = self.classes[min(720, img_name)]
        class_idx = self.classes_to_idx[class_name]
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx