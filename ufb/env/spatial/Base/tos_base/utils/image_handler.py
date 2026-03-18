"""
Simple image handler for spatial environments
"""
import os
import json
import re
import numpy as np
from typing import Dict, Union
from PIL import Image


class ImageHandler:
    """Handle images and data initialization based on position and orientation."""
    
    def __init__(self, base_dir: str, seed: int = None, image_size: tuple = (512, 512), preload_images: bool = True):
        """
        Initialize image handler with data loading.
        
        Args:
            base_dir: Base directory containing data subdirectories
            seed: Random seed for directory selection
            image_size: Target size for loaded images
            preload_images: Whether to load all images into memory
        """
        self.base_dir = base_dir
        self.image_size = image_size
        self.preload_images = preload_images
        self.image_dir, self.json_data = self._load_data(base_dir, seed)
        self.objects = {obj['object_id']: obj for obj in self.json_data.get('objects', [])}
        self._image_map, self._image_path_map = self._load_images()
        tmp = {obj['name']: obj['object_id'] for obj in self.objects.values()}
        tmp.update({k.replace('_', ' '): v for k, v in tmp.items()})
        tmp['agent'] = 'agent'
        self.name_2_cam_id = tmp
    

    def _load_data(self, base_dir: str, seed: int) -> tuple:
        """Load JSON data from a 'runNN' subdirectory (sorted by NN)."""
        target_run = f"run{seed:02d}"
        image_dir = os.path.join(base_dir, target_run)
        assert os.path.exists(image_dir), f"Image directory {image_dir} does not exist"
        with open(os.path.join(image_dir, "meta_data.json"), 'r') as f:
            json_data = json.load(f)
            
        return image_dir, json_data


    # def _load_data(self, base_dir: str, seed: int = None) -> tuple:
    #     """Load JSON data from a 'runNN' subdirectory (sorted by NN)."""
    #     subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    #     runs = [d for d in subdirs if re.fullmatch(r"run(\d+)", d)]
    #     assert runs, "No runNN folders found"
    #     runs.sort(key=lambda d: int(re.match(r"run(\d+)", d).group(1)))
    #     idx = (seed if seed is not None else np.random.randint(0, len(runs))) % len(runs)
    #     image_dir = os.path.join(base_dir, runs[idx])
        
    #     with open(os.path.join(image_dir, "meta_data.json"), 'r') as f:
    #         json_data = json.load(f)
            
    #     return image_dir, json_data
    
    def _load_images(self) -> Dict[str, Union[Image.Image, str]]:
        """Load images or paths based on preload setting."""
        image_map = {}
        image_path_map = {}
        for entry in self.json_data.get('images', []):
            key = f"{entry['cam_id']}_facing_{entry['direction']}"
            path = os.path.join(self.image_dir, entry['file'])
            assert os.path.exists(path)
            image_path_map[key] = path
            if self.preload_images:
                image_map[key] = Image.open(path).resize(self.image_size, Image.LANCZOS)
        topdown_path = os.path.join(self.image_dir, 'top_down_annotated.png')
        assert os.path.exists(topdown_path)
        image_path_map['topdown'] = topdown_path
        if self.preload_images:
            image_map['topdown'] = Image.open(topdown_path).resize(self.image_size, Image.LANCZOS)

        instruction_path = os.path.join(self.base_dir, 'instruction.png')
        assert os.path.exists(instruction_path)
        image_path_map['instruction'] = instruction_path
        if self.preload_images:
            image_map['instruction'] = Image.open(instruction_path).resize(self.image_size, Image.LANCZOS)            
        
        label_path = os.path.join(self.image_dir, 'orientation_instruction.png')
        assert os.path.exists(label_path)
        image_path_map['label'] = label_path
        if self.preload_images:
            image_map['label'] = Image.open(label_path).resize(self.image_size, Image.LANCZOS)

        return image_map, image_path_map
    
    def get_image(self, name: str = 'agent', direction: str = 'north') -> Image.Image:
        """
        Get image for given camera ID and direction.
        
        Args:
            name: Name of the object ('agent' or object_name or 'topdown' or 'instruction' as string)
            direction: Cardinal direction ('north', 'south', 'east', 'west')
            
        Returns:
            PIL Image
            
        Raises:
            KeyError: If image not found
        """
        # Handle special static images that don't need direction
        if name in ['topdown', 'instruction', 'label']:
            key = name
        else:
            key = f"{self.name_2_cam_id[name]}_facing_{direction}"
        
        if key not in self._image_map:
            raise KeyError(f"Image not found for name '{name}' facing '{direction}'")
        
        if self.preload_images:
            return self._image_map[key]
        else:
            path = self._image_path_map[key]
            return Image.open(path).resize(self.image_size, Image.LANCZOS)
        
    def get_image_path(self, name: str = 'agent', direction: str = 'north') -> str:
        """
        Get image path for given camera ID and direction.
        
        Args:
            name: Name of the object ('agent' or object_name or 'topdown' or 'instruction' as string)
            direction: Cardinal direction ('north', 'south', 'east', 'west')
            
        Returns:
            Image file path
            
        Raises:
            KeyError: If image path not found
        """
        # Handle special static images that don't need direction
        if name in ['topdown', 'instruction', 'label']:
            key = name
        else:
            key = f"{self.name_2_cam_id[name]}_facing_{direction}"
        
        if key not in self._image_path_map:
            raise KeyError(f"Image path not found for name '{name}' facing '{direction}'")
        
        return self._image_path_map[key]