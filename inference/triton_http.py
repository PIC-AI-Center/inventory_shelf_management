"""HTTP client wrapper for NVIDIA Triton Inference Server."""
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import requests

from config import get_settings


class TritonHTTPClient:
    """Thin wrapper around the Triton HTTP/REST API."""

    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        settings = get_settings()
        self.base_url = (
            f"http://{host or settings.triton_http_host}:{port or settings.triton_http_port}"
        )

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def health(self) -> bool:
        try:
            r = requests.get(self._url("/v2/health/ready"), timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def infer(
        self,
        model_name: str,
        inputs: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
        model_version: str = "1",
    ) -> Dict[str, Any]:
        """Generic HTTP inference call.

        inputs: list of dicts with keys: name, shape, datatype, data
        outputs: list of dicts with keys: name
        """
        payload = {
            "inputs": [
                {
                    "name": inp["name"],
                    "shape": inp["shape"],
                    "datatype": inp["datatype"],
                    "data": inp["data"],
                }
                for inp in inputs
            ],
            "outputs": [{"name": out["name"]} for out in outputs],
        }
        url = self._url(f"/v2/models/{model_name}/versions/{model_version}/infer")
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    def infer_numpy(
        self,
        model_name: str,
        input_name: str,
        array: np.ndarray,
        output_names: List[str],
        model_version: str = "1",
    ) -> Dict[str, np.ndarray]:
        """Convenience: send a single numpy array, return output arrays keyed by name."""
        inputs = [
            {
                "name": input_name,
                "shape": list(array.shape),
                "datatype": _numpy_to_triton_dtype(array.dtype),
                "data": array.flatten().tolist(),
            }
        ]
        outputs = [{"name": n} for n in output_names]
        raw = self.infer(model_name, inputs, outputs, model_version)
        result: Dict[str, np.ndarray] = {}
        for out in raw.get("outputs", []):
            name = out["name"]
            shape = out["shape"]
            dtype = _triton_to_numpy_dtype(out["datatype"])
            result[name] = np.array(out["data"], dtype=dtype).reshape(shape)
        return result


def _numpy_to_triton_dtype(dtype: np.dtype) -> str:
    mapping = {
        np.float32: "FP32",
        np.float16: "FP16",
        np.float64: "FP64",
        np.int32: "INT32",
        np.int64: "INT64",
        np.uint8: "UINT8",
        np.int8: "INT8",
    }
    for np_type, triton_type in mapping.items():
        if np.dtype(np_type) == dtype:
            return triton_type
    return "FP32"


def _triton_to_numpy_dtype(triton_dtype: str) -> np.dtype:
    mapping = {
        "FP32": np.float32,
        "FP16": np.float16,
        "FP64": np.float64,
        "INT32": np.int32,
        "INT64": np.int64,
        "UINT8": np.uint8,
        "INT8": np.int8,
    }
    return np.dtype(mapping.get(triton_dtype, np.float32))
