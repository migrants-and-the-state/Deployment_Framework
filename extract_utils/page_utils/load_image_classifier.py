import requests 
from io import BytesIO
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import platform
import torch.nn as nn
import numpy as np
import sys
sys.path.append('./Model_Trainer/') # TODO: Fix Path issue later. Currently added model_trainer as submodule

class Pretrained_Image_Classifier(nn.Module):
    ''' 
    Image Classifier Class with a pretrained backbone
    In this case DinoV2 from image retrieval experiments as the backbone.
    A head is attached to this backbone and this is used for inferences on images
    '''

    def __init__(self, model_path, model_inf_map, device=None, pretrained=True):
        ''' 
        Sets up Device (cuda/mps/cpu)
        Loads up backbone and linear head
        '''

        super(Pretrained_Image_Classifier, self).__init__()
        self.model_inf_map = model_inf_map
        self.device = self.setup_device() if device is None else device
        self.pretrain = pretrained
        if self.pretrain:
            print("Loading a pretrained model + linear head")
            self.linear_head = torch.load(model_path, map_location=self.device)
            self.linear_head.eval()
            # hardcode device for now as attention not supported in mps
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg').to(self.device) 
            self.backbone.eval()
        else:
            self.model = torch.load(model_path, map_location=self.device)
        self.transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        

    def setup_device(self):
        ''' 
        Maps device to mps if macOS (to be fixed to make it put cpu if mps is not present)
        And in windows, cuda if present else cpu 
        '''

        system = platform.system()
        if system == 'Darwin':  # macOS
            try:
                # import MetalPerformanceShaders as mps
                device = 'mps'  # Set device to MPS
                print("Metal Performance Shaders (MPS) is available on mac")
            except ImportError:
                print("Metal Performance Shaders (MPS) is not available. Defaulting to CPU.")
                device = 'cpu'  # Default to CPU if MPS is not available
        elif torch.cuda.is_available():  # Check if CUDA is available
            device = 'cuda'
            print("CUDA is available.")
        else:
            device = 'cpu'
            print("Neither Metal Performance Shaders (MPS) nor CUDA is available. Defaulting to CPU.")
        return device

    def forward(self, x):
        ''' 
        Forward pass through the model
        '''
        if self.pretrain:
            features = self.backbone(x)
            output = self.linear_head(features.to(self.device))
            return output.cpu(), features.cpu()
        else:
            output = self.model(x)
            return output, None

    def download_preprocess_image(self, url):
        ''' 
        Downloads and opens a url
        '''
        image = None
        try:
            response = requests.get(url, timeout=5)  # Set timeout as per your requirement
            response.raise_for_status()  # This will raise an exception for HTTP errors
            # If the request was successful, proceed with processing the image
            image = Image.open(BytesIO(response.content))
        except requests.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")  # Handle HTTP errors like 502
            return None
        except Exception as err:
            print(f"An error occurred: {err}")  # Handle other errors like timeouts
            return None
        return image

    def inference(self, url):
        ''' 
        Model inference on a single image
        '''
        image = self.download_preprocess_image(url)
        if image is not None:
            image_tensor = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                output, features = self(image_tensor.to(self.device))
            return output, features
        else:
            print("Image download/preprocessing failed.")
            return np.array([]), np.array([])

    def get_class(self, idx):
        return self.model_inf_map[idx]


