#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "openvino_delegate.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"

std::function<void (TfLiteOpaqueContext* opaque_context, TfLiteOpaqueNode* node)> test_function_;

class OpenVINODelegateTest : public testing::Test {
    protected:
    
    void setup_delegate(std::function<void (TfLiteOpaqueContext* opaque_context, TfLiteOpaqueNode* node)> test_function) {
    TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
    test_function_ = test_function; 
    opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* opaque_context,
                                       TfLiteOpaqueDelegate* opaque_delegate_,
                                       void* data) -> TfLiteStatus {
            // Test that an unnamed delegate kernel can be passed to the TF Lite
            // runtime.

            TfLiteIntArray* execution_plan;
            TF_LITE_ENSURE_STATUS(
                TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
            for (int i = 0; i < execution_plan->size; ++i) {
              TfLiteOpaqueNode* node = nullptr;
              TfLiteRegistrationExternal* registration = nullptr;
              TfLiteOpaqueContextGetNodeAndRegistration(opaque_context, i, &node,
                                                        &registration);
                bool is_supported = false;
                test_function_(opaque_context, node);
            }

            TfLiteRegistrationExternal* registration_external =
                TfLiteRegistrationExternalCreate(kTfLiteBuiltinDelegate,
                                                 /*name*/ nullptr,
                                                 /*version=*/1);
            return TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
                opaque_context, registration_external, execution_plan, opaque_delegate_);
            };
            opaque_delegate_ = TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);

            tflite::ops::builtin::BuiltinOpResolver resolver;
            tflite::InterpreterBuilder builder(*model, resolver);
            builder.AddDelegate(opaque_delegate_);
            EXPECT_EQ(kTfLiteOk, builder(&interpreter_));
            ASSERT_NE(interpreter_, nullptr);
        }

        void SetUp() override {
            model = tflite::FlatBufferModel::BuildFromFile(
                            "tensorflow/lite/testdata/add.bin");
            ASSERT_NE(model, nullptr);
}

        void TearDown() override {
            TfLiteOpaqueDelegateDelete(opaque_delegate_);
        }
        protected:
        std::unique_ptr<tflite::Interpreter> interpreter_;
        TfLiteOpaqueDelegate* opaque_delegate_ = nullptr;
        TfLiteOpaqueContext* test_graph_context = nullptr;
         std::unique_ptr<tflite::FlatBufferModel> model;
        // static std::function<void (TfLiteOpaqueContext* opaque_context, TfLiteOpaqueNode* node)> test_function_;
};

TEST_F(OpenVINODelegateTest, CheckInputsTypeTest) {
    
            auto test_func = [] (TfLiteOpaqueContext* opaque_context, TfLiteOpaqueNode* node)-> void {
                TfLiteOpenVINODelegateOptions options_del;
                bool is_supported = false;
            std::unique_ptr<tflite::openvinodelegate::OpenVINODelegate> ov_del_test =
                std::make_unique<tflite::openvinodelegate::OpenVINODelegate>(&options_del);
                EXPECT_EQ(kTfLiteOk, ov_del_test->CheckDataTypeSupported(opaque_context,
                                node, {{kTfLiteFloat32}, {kTfLiteFloat32}}, is_supported));
            };
            setup_delegate(test_func);

}