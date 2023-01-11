import base64 
import json 

import numpy as np 
import cv2 

import torch 
import torch.nn.functional as F 

import triton_python_backend_utils as pb_utils

def score_to_logit(score):
    score = torch.tensor(score)
    logit = F.softmax(score, dim=1)
    return logit

def wrap_json(logit):
    """
    1. tensor logit to list 
    2. make json output 
    """
    obj = {
        "result": np.array(logit).tolist()
    }
    return json.dumps(obj, ensure_ascii=False)

class TritonPythonModel:
    """
    post processing main logic
    """

    def execute(self, requests):
        responses = [] 
        for request in requests: 
            predict_score = pb_utils.get_input_tensor_by_name(
                request, "INPUT__0"
            ).as_numpy()
            
            logit = score_to_logit(predict_score)
            response = np.array(wrap_json(logit), dtype=np.object_)
            response = pb_utils.Tensor("result", response)
            response = pb_utils.InferenceResponse(output_tensors=[response])
            responses.append(response)
        return responses


