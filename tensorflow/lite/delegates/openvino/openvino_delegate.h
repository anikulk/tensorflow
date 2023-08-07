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

#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_H_

#include "tensorflow/lite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct {
  uint32_t devicePreference;
  uint32_t powerPreference;
} TfLiteOpenVINODelegateOptions;

// Returns a str1ucture with the default OpenVINO delegate options.
TfLiteOpenVINODelegateOptions TfLiteOpenVINODelegateOptionsDefault();

// Creates a new delegate instance that need to be destroyed with
// `TfLiteOpenVINODelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the following default values are used:
TfLiteDelegate* TfLiteOpenVINODelegateCreate(
    const TfLiteOpenVINODelegateOptions* options);

// Destroys a delegate created with `TfLiteOpenVINODelegateCreate` call.
void TfLiteOpenVINODelegateDelete(TfLiteDelegate* delegate);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_OPENVINO_DELEGATE_H_
