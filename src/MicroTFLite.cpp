#include "MicroTFLite.h"

// TensorFlow Lite components
tflite::AllOpsResolver    tflOpsResolver;
const tflite::Model      *tflModel           = nullptr;
tflite::MicroInterpreter *tflInterpreter     = nullptr;
TfLiteTensor             *tflInputTensor     = nullptr;
TfLiteTensor             *tflOutputTensor    = nullptr;
float                     tflInputScale      = 0.0f;
int32_t                   tflInputZeroPoint  = 0;
float                     tflOutputScale     = 0.0f;
int32_t                   tflOutputZeroPoint = 0;

// Initializes the TensorFlow Lite model and interpreter
bool ModelInit(const unsigned char *model, byte *tensorArena, int tensorArenaSize) {
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println(F("Model schema version mismatch!"));
    return false;
  }

  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize);
  tflInterpreter->AllocateTensors();
  tflInputTensor  = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  // Ensure tensors are initialized
  if (tflInputTensor == nullptr || tflOutputTensor == nullptr) {
    Serial.println(F("Tensors are not initialized."));
    return false;
  }

  // Set scales and zero points. We are using per-tensor quantization to apply the same scale and zero-point to all values in the input (tflInputTensor) and output (tflOutputTensor) tensors.
  tflInputScale      = tflInputTensor->params.scale;
  tflInputZeroPoint  = tflInputTensor->params.zero_point;
  tflOutputScale     = tflOutputTensor->params.scale;
  tflOutputZeroPoint = tflOutputTensor->params.zero_point;
  return true;
}

// Prints metadata about the model, such as description and version
void ModelPrintMetadata() {
  if (tflModel->description() != nullptr) {
    const flatbuffers::String *description = tflModel->description();
    Serial.print(F("Model Description: "));
    Serial.println(description->str().c_str());
  } else {
    Serial.println(F("No model description available."));
  }
  Serial.print(F("Model Version: "));
  Serial.println(tflModel->version());
  Serial.print(F("Arena Used Bytes: "));
  Serial.println(tflInterpreter->arena_used_bytes());
}

// Prints information about the tensors, including type and dimensions
void ModelPrintTensorInfo() {
  if (tflInputTensor == nullptr || tflOutputTensor == nullptr) {
    Serial.println(F("Tensors are not initialized."));
    return;
  }

  // Print input tensor info
  Serial.println(F("Input Tensor Information:"));
  Serial.print(F("Type: "));
  Serial.println(tflInputTensor->type == kTfLiteFloat32 ? "float32" : "int8");
  Serial.print(F("Dimensions: "));
  for (int i = 0; i < tflInputTensor->dims->size; ++i) {
    Serial.print(tflInputTensor->dims->data[i]);
    if (i < tflInputTensor->dims->size - 1)
      Serial.print(F(" x "));
  }
  Serial.println();

  // Print output tensor info
  Serial.println(F("Output Tensor Information:"));
  Serial.print(F("Type: "));
  Serial.println(tflOutputTensor->type == kTfLiteFloat32 ? "float32" : "int8");
  Serial.print(F("Dimensions: "));
  for (int i = 0; i < tflOutputTensor->dims->size; ++i) {
    Serial.print(tflOutputTensor->dims->data[i]);
    if (i < tflOutputTensor->dims->size - 1)
      Serial.print(F(" x "));
  }
  Serial.println();
}

// Prints the quantization parameters for the input and output tensors
void ModelPrintTensorQuantizationParams() {
  if (tflInputTensor == nullptr || tflOutputTensor == nullptr) {
    Serial.println(F("Tensors are not initialized."));
    return;
  }

  Serial.println(F("Input Tensor Quantization Parameters:"));
  Serial.print(F("Scale: "));
  Serial.println(tflInputScale, 10);
  Serial.print(F("Zero Point: "));
  Serial.println(tflInputZeroPoint);

  Serial.println(F("Output Tensor Quantization Parameters:"));
  Serial.print(F("Scale: "));
  Serial.println(tflOutputScale, 10);
  Serial.print(F("Zero Point: "));
  Serial.println(tflOutputZeroPoint);
}

// Quantizes a float value for int8 tensors
// In quantization, floating-point values (e.g., activations or weights in the model) are mapped to integer values (e.g., 8-bit integers) for efficiency.
// The scale and zero-point are used to convert between the floating-point and integer domains.
inline int8_t QuantizeInput(float x, float scale, float zeroPoint) {
  float  quantizedFloat = (x / scale) + zeroPoint;
  int8_t quantizedValue = static_cast<int8_t>(quantizedFloat);
  return quantizedValue;
}

// Dequantizes an int8 value to a float
inline float DequantizeOutput(int8_t quantizedValue, float scale, float zeroPoint) {
  return ((float)quantizedValue - zeroPoint) * scale;
}

// Sets the input tensor with a given value, handling quantization if needed
bool ModelSetInput(float inputValue, int index, bool showQuantizedValue) {
  int maxSize = 1;
  for (int i = 0; i < tflInputTensor->dims->size; i++)
    maxSize *= tflInputTensor->dims->data[i];
  //   if (tflInputTensor == nullptr || index >= tflInputTensor->dims->data[1]) {
  if (tflInputTensor == nullptr || index >= maxSize) {
    Serial.print(F("Input tensor index out of range!: "));
    Serial.print(index);
    Serial.print(F(" Range: "));
    Serial.println(tflInputTensor->dims->data[1]);
    Serial.println(maxSize);
    return false;
  }

  if (tflInputTensor->type == kTfLiteInt8) {
    int8_t quantizedValue            = QuantizeInput(inputValue, tflInputScale, tflInputZeroPoint);
    tflInputTensor->data.int8[index] = quantizedValue;
    if (showQuantizedValue) {
      Serial.print(F("Quantized value for index: "));
      Serial.print(index);
      Serial.print(F(" : "));
      Serial.print(quantizedValue);
      Serial.print(F(" , input : "));
      Serial.print(inputValue);
      Serial.print(F(" using scale: "));
      Serial.print(tflInputScale);
      Serial.print(F(" and zero-point: "));
      Serial.println(tflInputZeroPoint);
    }
  } else if (tflInputTensor->type == kTfLiteFloat32) {
    tflInputTensor->data.f[index] = inputValue;
  } else {
    Serial.println(F("Unsupported input tensor type!"));
    return false;
  }

  return true;
}

// Prints the dimensions of the output tensor
void ModelPrintOutputTensorDimensions() {
  if (tflOutputTensor == nullptr || tflOutputTensor->dims->size == 0) {
    Serial.println(F("Output tensor is null or has no dimensions!"));
    return;
  }

  Serial.print(F("Output tensor dimensions: "));
  Serial.println(tflOutputTensor->dims->size);
  for (int i = 0; i < tflOutputTensor->dims->size; ++i) {
    Serial.print(F("Output Dimension "));
    Serial.print(i);
    Serial.print(F(": "));
    Serial.println(tflOutputTensor->dims->data[i]);
  }
}

// Prints the dimensions of the input tensor
void ModelPrintInputTensorDimensions() {
  if (tflInputTensor == nullptr) {
    Serial.println(F("Input tensor is null!"));
    return;
  }

  Serial.print(F("Input tensor dimensions: "));
  Serial.println(tflInputTensor->dims->size);

  for (int i = 0; i < tflInputTensor->dims->size; ++i) {
    Serial.print(F("Dimension "));
    Serial.print(i);
    Serial.print(F(": "));
    Serial.println(tflInputTensor->dims->data[i]);
  }
}

// Runs inference on the model
bool ModelRunInference() {
  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  if (invokeStatus != kTfLiteOk) {
    Serial.println(F("Inference failed!"));
    return false;
  }
  return true;
}

// Retrieves the output value from the model
float ModelGetOutput(int index, bool showQuantizedValue) {
  if (tflOutputTensor == nullptr || index >= tflOutputTensor->dims->data[1]) {
    Serial.println(F("Output tensor index out of range!"));
    return -1;
  }

  if (tflOutputTensor->type == kTfLiteInt8) {
    int8_t quantizedValue = tflOutputTensor->data.int8[index];
    // Print the quantized value if requested
    // Print the quantized value, scale, and zero point if requested
    if (showQuantizedValue) {
      Serial.print(F("Quantized value for output index "));
      Serial.print(index);
      Serial.print(F(": "));
      Serial.print(quantizedValue);
      Serial.print(F(" using scale: "));
      Serial.print(tflOutputScale, 10);
      Serial.print(F(" and zero-point: "));
      Serial.println(tflOutputZeroPoint);
    }
    return DequantizeOutput(quantizedValue, tflOutputScale, tflOutputZeroPoint);
  } else if (tflOutputTensor->type == kTfLiteFloat32) {
    return tflOutputTensor->data.f[index];
  } else {
    Serial.println(F("Unsupported output tensor type!"));
    return -1;
  }
}
