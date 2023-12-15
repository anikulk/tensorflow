/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "NgraphNodes.h"
void NgraphNodes::setOutputAtOperandIndex(int index,
                                          ov::Output<ov::Node> output) {
  outputAtOperandIndex[index] = output;
}

std::shared_ptr<ov::Node> NgraphNodes::getInterimNode(int index) {
  return std::shared_ptr<ov::Node>(outputAtOperandIndex.at(index).get_node_shared_ptr());
}

std::shared_ptr<ov::Node> NgraphNodes::getInputNode(const TfLiteTensor &tensor,
                                                    int tensor_index) {
  std::shared_ptr<ov::Node> input;
  if (tensor.type == kTfLiteFloat32) {
    ov::element::Type elementType = ov::element::f32;
    if (tensor.allocation_type == kTfLiteMmapRo) {
      ov::Shape tensor_shape;
      const void *data = (const void *) tensor.data.raw_const;
      for (int i = 0; i < tensor.dims->size; i++)
        tensor_shape.push_back(static_cast<size_t>(tensor.dims->data[i]));
      input = createConstNode(elementType, tensor_shape, data);
    } else {
      input = getInterimNode(tensor_index);
    }
  }

  return input;
}
