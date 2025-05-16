from typing import Any, Dict, List, Optional, Tuple, Union
from libraries.MultiFTNet import MultiFTNet
from libraries.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from helpers import transform
import numpy as np
import torch.nn.functional as F
import cv2, os, torch

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE,
    'MultiFTNet': MultiFTNet
}

class FaceLiveness:
    def __init__(self):
        self.MODEL_STATE = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def parse_model_name(self, model_name: str) -> Tuple[int, int, str, Optional[float]]:
        """
        Parse a model name to extract information such as input height, input width, model type, and scale.

        Parameters:
        - model_name (str): The name of the model.

        Returns:
        - Tuple[int, int, str, Optional[float]]: A tuple containing the parsed information.
        - The first element is the input height (int).
        - The second element is the input width (int).
        - The third element is the model type (str).
        - The fourth element is the scale, which is optional and can be None or a float.
        """

        info = model_name.split('_')[0:-1]
        h_input, w_input = info[-1].split('x')
        model_type = model_name.split('.pth')[0].split('_')[-1]

        if info[0] == "org":
            scale = None
        else:
            scale = float(info[0])
        return int(h_input), int(w_input), model_type, scale


    def crop(self, org_img: np.ndarray, bbox: Tuple[int, int, int, int], scale: float, out_w: int, out_h: int, crop=True) -> np.ndarray:
        """
        Crop and resize a region of interest from the original image based on the specified bounding box,
        scale factor, and output dimensions.

        Parameters:
        - org_img (np.ndarray): The original image as a NumPy array.
        - bbox (Tuple[int, int, int, int]): Bounding box coordinates (left, top, width, height).
        - scale (float): The scale factor for cropping.
        - out_w (int): The desired output width.
        - out_h (int): The desired output height.

        Returns:
        - np.ndarray: The cropped and resized image.
        """

        if not crop:
            dst_img = cv2.resize(org_img, (out_w, out_h))
        else:
            src_h, src_w, _ = np.shape(org_img)
            left_top_x, left_top_y, right_bottom_x, right_bottom_y = self.get_new_box(src_w, src_h, bbox, scale)
            img = org_img[left_top_y: right_bottom_y+1, left_top_x: right_bottom_x+1]
            dst_img = cv2.resize(img, (out_w, out_h))

        return dst_img


    def get_new_box(self, src_w: int, src_h: int, bbox: Tuple[int, int, int, int], scale: float) -> Tuple[int, int, int, int]:
        """
        Calculate a new bounding box based on the original bounding box, image dimensions, and scale factor.

        Parameters:
        - src_w (int): The width of the source image.
        - src_h (int): The height of the source image.
        - bbox (Tuple[int, int, int, int]): Bounding box coordinates (left, top, width, height).
        - scale (float): The scale factor for adjusting the bounding box.

        Returns:
        - Tuple[int, int, int, int]: The updated bounding box coordinates (left, top, right, bottom).
        """

        x = bbox[0]
        y = bbox[1]
        box_w = bbox[2]
        box_h = bbox[3]

        scale = min((src_h-1)/box_h, min((src_w-1)/box_w, scale))

        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w/2+x, box_h/2+y

        left_top_x = center_x-new_width/2
        left_top_y = center_y-new_height/2
        right_bottom_x = center_x+new_width/2
        right_bottom_y = center_y+new_height/2

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0

        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0

        if right_bottom_x > src_w-1:
            left_top_x -= right_bottom_x-src_w+1
            right_bottom_x = src_w-1

        if right_bottom_y > src_h-1:
            left_top_y -= right_bottom_y-src_h+1
            right_bottom_y = src_h-1

        return int(left_top_x), int(left_top_y),\
                int(right_bottom_x), int(right_bottom_y)


    def predict_liveness(self, img: Any, model_path: str) -> Union[None, np.ndarray]:
        """
        Predict liveness using a trained model on the provided image.

        Parameters:
        - img (Any): The input image.
        - model_path (str): The file path to the trained model.

        Returns:
        - Union[None, np.ndarray]: The predicted liveness result as a NumPy array or None if the model is not loaded.
        """

        test_transform = transform.Compose([transform.ToTensor()])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        self.load_model(model_path)
        self.MODEL_STATE.eval()
        with torch.no_grad():
            result = self.MODEL_STATE.forward(img)
            result = F.softmax(result, dim=1).cpu().numpy()
        return result


    def load_model(self, model_path: str) -> None:
        """
        Load a model from the specified file path and update the MODEL_STATE.

        Parameters:
        - model_path (str): The file path to the trained model.

        Returns:
        - None
        """

        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = self.parse_model_name(model_name)
        kernel_size = self.get_kernel(h_input, w_input)
        self.MODEL_STATE = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.MODEL_STATE.load_state_dict(new_state_dict)
        else:
            self.MODEL_STATE.load_state_dict(state_dict)
        return None
    

    def get_kernel(self, height: int, width: int) -> Tuple[int, int]:
        """
        Calculate the kernel size based on the input height and width.

        Parameters:
        - height (int): The input height.
        - width (int): The input width.

        Returns:
        - Tuple[int, int]: The calculated kernel size.
        """

        kernel_size = ((height + 15) // 16, (width + 15) // 16)
        return kernel_size

    def resize_keep_aspect(self, img, bbox=None, max_size=192):
        h, w = img.shape[:2]

        if h > w:
            scale = max_size / h
        else:
            scale = max_size / w

        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        resized_bbox = None
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1 = int(x1 * scale)
            y1 = int(y1 * scale)
            x2 = int(x2 * scale)
            y2 = int(y2 * scale)
            resized_bbox = [x1, y1, x2, y2]

        return resized_img, resized_bbox, scale

    async def check(self, image: np.ndarray, bounding_box: List[int]) -> Dict[str, Union[int, str, Dict[str, Union[bool, float]]]]:
        """
        Perform liveness detection on a face region within an image using multiple models.

        Parameters:
        - image (np.ndarray): The input image as a NumPy array.
        - bounding_box (List[int]): Bounding box coordinates of the face region (x1, y1, x2, y2).

        Returns:
        - Dict[str, Union[int, str, Dict[str, Union[bool, float]]]]: A dictionary containing the result of the liveness check.
        - 'status' (int): The HTTP status code.
        - 'message' (str): A message describing the result.
        - 'data' (Dict[str, Union[bool, float]]): Additional data.
            - 'liveness' (bool): Indicates whether the detected face is live or not.
            - 'score' (float): The liveness score.
        """

        image, bounding_box, _ = self.resize_keep_aspect(image, bounding_box)            
        bbox = [bounding_box[0], bounding_box[1], bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1]]

        model_dir = "datasets"
        prediction = np.zeros((1, 3))
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = self.parse_model_name(model_name)
            params = {
                "org_img": image,
                "bbox": bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True
            }

            if scale is None:
                params["crop"] = False

            image_cropped = self.crop(**params)

            prediction += self.predict_liveness(image_cropped, os.path.join(model_dir, model_name))
            
        label = np.argmax(prediction)
        livenessResult = True if label == 1 else False
        value = prediction[0][label]/2
        livenessScore = round(value, 2)

        return { "status": 200, "message": "", "data": { "result": livenessResult, "score": livenessScore } }
