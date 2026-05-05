import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import onnx
import tensorrt as trt
import torch

calset = torch.load("cs50.pt")

cnt = 0

def calibration_data_stream():
    for i in range(len(calset)):
        im_patches = calset[i][0].detach().numpy()
        train_feat = calset[i][1].detach().numpy()
        target_labels = calset[i][2].detach().numpy()
        train_ltrb = calset[i][3].detach().numpy()
        yield [im_patches, train_feat, target_labels, train_ltrb]

class EntropyCalibrator2(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_stream, cache_file):
        # input_layers: a list of dictionaries containing names and shapes of the input layers
        # cache_file: path to save calibration cache
        super(EntropyCalibrator2, self).__init__()
        self.calibration_stream = calibration_stream
        self.cache_file = cache_file
        self.batch_size = 1
        self.current_index = 0
        self.device_input_buffers = []  # To hold device input buffers
        self.allocate_buffers()

    def allocate_buffers(self):
        for tensors in next(iter(self.calibration_stream)):
            for tensor in tensors:
                volume = trt.volume(tensor.shape)
                print("allocate_buffers")
                print(volume)
                print(tensor.nbytes)
                # dtype = np.float32
                self.device_input_buffers.append(cuda.mem_alloc(tensor.nbytes))

    def get_batch_size(self):
        return 1
    
    def get_batch(self, names):
        try:
            for name in names:
                print(name)
            data = next(self.calibration_stream)
            for t in data:
                print(t.shape)
            global cnt
            print(cnt)
            cnt=cnt+1
            for input_tensor, b in zip(data, self.device_input_buffers):
                # if name not in self.device_input_buffers:
                    # raise ValueError(f"Buffer for {name} not allocated")
                    
                if not isinstance(input_tensor, np.ndarray) or input_tensor.dtype != np.float32:
                    raise TypeError("Input tensor must be a np.ndarray with dtype np.float32")
                
                # if np.prod(input_tensor.shape) * input_tensor.dtype.itemsize != b.size:
                    # raise ValueError("Input tensor size does not match the allocated buffer size")
                
                cuda.memcpy_htod(b, np.ascontiguousarray(input_tensor))
                print("get batch")
                print(type(b))
                print(b)
                print(int(b))
            return [int(b) for b in self.device_input_buffers]
        except StopIteration:
            return []
    
    def read_calibration_cache(self):
        try:
            with open(self.cache_file, "rb") as f:
                return f.read()
        except:
            return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

calibration_data_stream_gen = calibration_data_stream()
calibrator = EntropyCalibrator2(calibration_data_stream_gen, "calibration_cache.bin")

# input_layers = [
    # {'name': 'im_patches', 'shape': (1, 3, 288, 288)},
    # {'name': 'train_feat', 'shape': (1, 256, 18, 18)},
    # {'name': 'target_labels', 'shape': (1, 1, 18, 18)},
    # {'name': 'train_ltrb', 'shape': (1, 4, 18, 18)}
# ]

# Constants
# ONNX_MODEL_PATH = 'new_full_explicit_batch16_50.onnx'
# TENSORRT_ENGINE_PATH = 'new_full_explicit_batch16_50.engine'
# # ONNX_MODEL_PATH = 'new_full_implicit_batch_16.onnx'
# # TENSORRT_ENGINE_PATH = 'new_full_implicit_batch_16.engine'
# MIN_BATCH_SIZE = 1
# MAX_BATCH_SIZE = 16

# # Set up the logger
# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# # Create a TensorRT builder, runtime, and network
# builder = trt.Builder(TRT_LOGGER)
# network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
# config = builder.create_builder_config()
# parser = trt.OnnxParser(network, TRT_LOGGER)
# parser.set_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)

# # Parse the ONNX model file
# with open(ONNX_MODEL_PATH, 'rb') as model:
    # if not parser.parse(model.read()):
        # print('ERROR: Failed to parse the ONNX file.')
        # for error in range(parser.num_errors):
            # print(parser.get_error(error))
        # exit(1)

# # Define optimization profile for dynamic batch size
# config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
# # config.set_flag(trt.BuilderFlag.INT8)
# # config.int8_calibrator = calibrator
# profile = builder.create_optimization_profile()
# profile.set_shape('im_patches', (MIN_BATCH_SIZE, 3, 288, 288), (MAX_BATCH_SIZE, 3, 288, 288), (MAX_BATCH_SIZE, 3, 288, 288))
# profile.set_shape('train_feat', (MIN_BATCH_SIZE, 256, 18, 18), (MAX_BATCH_SIZE, 256, 18, 18), (MAX_BATCH_SIZE, 256, 18, 18))
# profile.set_shape('target_labels', (1, MIN_BATCH_SIZE, 18, 18), (1, MAX_BATCH_SIZE, 18, 18), (1, MAX_BATCH_SIZE, 18, 18))
# profile.set_shape('train_ltrb', (MIN_BATCH_SIZE, 4, 18, 18), (MAX_BATCH_SIZE, 4, 18, 18), (MAX_BATCH_SIZE, 4, 18, 18))
# config.add_optimization_profile(profile)

# # Build the engine
# engine = builder.build_serialized_network(network, config)

# # Save the engine
# with open(TENSORRT_ENGINE_PATH, 'wb') as f:
    # f.write(engine)

# Function to convert ONNX model to TensorRT engine and save it

logger = trt.Logger(trt.Logger.VERBOSE)

builder = trt.Builder(logger)

network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
# network.add_input("im_patches", trt.float32, (1,3,288,288))
# network.add_input("train_feat", trt.float32, (1,256,18,18))
# network.add_input("target_labels", trt.float32, (1,1,18,18))
# network.add_input("train_ltrb", trt.float32, (1,4,18,18))
parser = trt.OnnxParser(network, logger)
parser.set_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)
success = parser.parse_from_file("new_full_implicit_batch1_50_sanitized.onnx")

for i in range(network.num_layers):
    layer = network.get_layer(i)
    print(f"Layer {i}: {layer.type} - {layer.name}")

for idx in range(parser.num_errors):
    print(parser.get_error(idx))
if not success:
    print("bad")

config = builder.create_builder_config()
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = calibrator
# confin.set_flag
# config.max_aux_streams = 7

serialized_engine = builder.build_serialized_network(network, config)

with open("new_full_implicit_batch1_50_sanitized_calibrated.engine", "wb") as f:
    f.write(serialized_engine)