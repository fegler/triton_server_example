import sys 
import base64 

import numpy as np 
import cv2  

import albumentations as A

import triton_python_backend_utils as pb_utils 

def read_image_from_byte(raw_image):
    base64_image = raw_image.decode("utf-8")
    raw_image = base64.b64decode(base64_image)
    image = cv2.imdecode(np.frombuffer(raw_image, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image 

def transform_image(image):
    """
    trasnform cv2 image 
    1. image resize, normalization
    2. transpose image 
    """
    IM_SIZE=224
    transform = A.Compose([A.Resize(IM_SIZE, IM_SIZE), A.Normalize()])
    image = transform(image=image)["image"]
    image = np.transpose(image, (2,0,1))
    return image

def gen_input_image(raw_images):
    """
    generate input image batch array 
    """
    batch_input_image = []
    for i in range(raw_images.shape[0]):
        ## batch
        raw_image = raw_images[i]
        image = read_image_from_byte(raw_image)
        image = transform_image(image)
        batch_input_image.append(image)
    return np.stack(batch_input_image, axis=0)

class TritonPythonModel: 
    """
    preprocessing main logic 
    """

    def execute(self, requests):
        responses = [] 
        for request in requests:
            ## get request 
            raw_images = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()

            ## make response
            input_image = gen_input_image(raw_images)
            input_image_tensor = pb_utils.Tensor(
                "input_image", input_image.astype(np.float32)
            )
            response = pb_utils.InferenceResponse(
                output_tensors=[input_image_tensor]
            )
            responses.append(response)
        return responses