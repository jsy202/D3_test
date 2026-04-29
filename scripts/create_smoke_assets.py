from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnx
from onnx import TensorProto, helper


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    model_path = root / "models" / "smoke_identity.onnx"
    image_path = root / "samples" / "smoke_lane.jpg"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    create_identity_model(model_path)
    create_sample_image(image_path)
    print(f"model={model_path}")
    print(f"image={image_path}")
    return 0


def create_identity_model(path: Path) -> None:
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 320, 1600])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 320, 1600])
    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph([node], "smoke_identity", [input_tensor], [output_tensor])
    model = helper.make_model(graph, producer_name="ufldv2-d3g-benchmark")
    model.ir_version = 9
    model.opset_import[0].version = 13
    onnx.save(model, path)


def create_sample_image(path: Path) -> None:
    height, width = 720, 1280
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (35, 35, 35)
    cv2.line(image, (460, height), (560, 360), (255, 255, 255), 8)
    cv2.line(image, (820, height), (720, 360), (255, 255, 255), 8)
    cv2.line(image, (0, 520), (width, 520), (70, 70, 70), 2)
    cv2.imwrite(str(path), image)


if __name__ == "__main__":
    raise SystemExit(main())
