"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import os
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
# import draccus
import requests
from typing import Any, Dict, List, Union
from numpy.lib.format import descr_to_dtype, dtype_to_descr
from base64 import b64decode, b64encode

def numpy_serialize(o):
    if isinstance(o, (np.ndarray, np.generic)):
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {
            "__numpy__": b64encode(data).decode(),
            "dtype": dtype_to_descr(o.dtype),
            "shape": o.shape,
        }

    msg = f"Object of type {o.__class__.__name__} is not JSON serializable"
    raise TypeError(msg)


def numpy_deserialize(dct):
    if "__numpy__" in dct:
        np_obj = np.frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        return np_obj.reshape(shape) if (shape := dct["shape"]) else np_obj[0]
    return dct


def convert_numpy_in_dict(data, func):
    """
    Recursively processes a JSON-like dictionary, converting any NumPy arrays
    or lists of NumPy arrays into a serializable format using the provided function.

    Args:
        data: The JSON-like dictionary or object to process.
        func: A function to apply to each NumPy array to make it serializable.

    Returns:
        The processed dictionary or object with all NumPy arrays converted.
    """
    if isinstance(data, dict):
        if "__numpy__" in data:
            return func(data)
        return {key: convert_numpy_in_dict(value, func) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_in_dict(item, func) for item in data]
    elif isinstance(data, (np.ndarray, np.generic)):
        return func(data)
    else:
        return data
    
    
class Message(object):
    def __init__(self):
        pass
    
    def serialize(self):
        raise NotImplementedError
    
    @classmethod
    def deserialize(cls, response: Dict[str, Any]):
        raise NotImplementedError


class RequestMessage(Message):
    def __init__(self, image: Dict[str, Any], instruction: str, history: Dict[str, Any], state: Dict[str, Any], condition: Dict[str, Any], gt_action: Union[np.ndarray, List], dataset_name: str, timestamp: str):
        self.image, self.instruction, self.history, self.state, self.gt_action, self.dataset_name, self.timestamp = image, instruction, history, state, gt_action, dataset_name, timestamp
        self.condition = condition

    def serialize(self):
        msg = {
            "image": self.image,
            "instruction": self.instruction,
            "history": self.history,
            "state": self.state,
            "condition": self.condition,
            "gt_action": self.gt_action,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp
        }
        return convert_numpy_in_dict(msg, numpy_serialize)
    
    @classmethod
    def deserialize(cls, response: Dict[str, Any]):
        response = convert_numpy_in_dict(response, numpy_deserialize)
        return cls(
            image=response["image"],
            instruction=response["instruction"],
            history=response["history"],
            state=response["state"],
            condition=response["condition"],
            gt_action=response["gt_action"],
            dataset_name=response["dataset_name"],
            timestamp=response["timestamp"]
        )


class ResponseMessage(Message):
    def __init__(self, action: np.ndarray, err: float, traj_image: np.ndarray = None):
        self.action = action
        self.err = err
        self.traj_image = traj_image if traj_image is not None else np.zeros((1, 1, 3), dtype=np.uint8)
    
    def serialize(self):
        msg = {
            "action": self.action,
            "err": self.err,
            "traj_image": self.traj_image,
        }
        return convert_numpy_in_dict(msg, numpy_serialize)
    
    @classmethod
    def deserialize(cls, response: Dict[str, Any]):
        response = convert_numpy_in_dict(response, numpy_deserialize)
        err = response["err"] if "err" in response else 0.0
        traj_image = response["traj_image"] if "traj_image" in response else None
        if type(err) == str:
            print(f"[WARN] Server eror: {err}.")
        return cls(action=response["action"], err=err, traj_image=traj_image)



class HttpActionClient(object):
    def __init__(self, server_ip: str, server_port: int):
        self.server_ip, self.server_port = server_ip, server_port
    
    @property
    def timestamp(self):
        return str(datetime.now()).replace(" ", "_").replace(":", "-")
    
    def query_action(self, image_dict: Dict[str, Any], instruction: str, state_dict: Dict[str, Any], condition_dict: Dict[str, Any], history: Optional[Dict[str, Any]] = None, dataset="grasp", gt_action: Optional[np.ndarray] = None) -> np.ndarray:
        if history is None:
            history = {k: [] for k in image_dict.keys()}
        if gt_action is None:
            gt_action = []
        
        request = RequestMessage(image_dict, instruction, history, state_dict, condition_dict, gt_action, dataset, self.timestamp) 
        try:
            response = requests.post(
                f"http://{self.server_ip}:{self.server_port}/act",
                json=request.serialize()
            )
        except Exception:
            raise RuntimeError(f"Server is not up ?!")
        
        if response.status_code != 200:
            raise Exception(f"Server is not up ?! {response.status_code}: {response.text}")
        try:
            response = ResponseMessage.deserialize(response.json())
        except Exception as e:
            raise RuntimeError(response.text)
        
        # Validate that the server returned a valid trajectory image
        if not isinstance(response.traj_image, np.ndarray) or response.traj_image.ndim != 3:
            print("[WARN] Server did not return a valid trajectory image (traj_image).")
            return response.action, response.err, None
                            
        return response.action, response.err, response.traj_image
    

if __name__ == "__main__":
    server_ip = "localhost" #"172.17.0.1"
    server_port = 22085 #21000
    client = HttpActionClient(server_ip, server_port)
    
    from PIL import Image
    obs =  np.zeros((224, 224, 3), dtype=np.uint8) #np.array(Image.open("steore-left.png"), dtype=np.uint8)
    instruction = "Pick up red box."
    action = client.query_action(obs, instruction) # delta: xyz, rpy, openness
    print("unnormalized action: ", action)