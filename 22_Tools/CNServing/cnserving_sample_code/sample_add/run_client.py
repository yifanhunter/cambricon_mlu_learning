from __future__ import division
import sys
import numpy as np
import time
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf
import grpc


def grpc_client_build_and_run(ip_port):
    channel = grpc.insecure_channel(ip_port)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'sample_add'

    input_shape = [1,4]
    input1 = np.random.random(input_shape).astype('float32')
    input2 = np.random.random(input_shape).astype('float32')
    expected_result = input1 + input2

    request.inputs['main/arg-0'].CopyFrom(tf.make_tensor_proto(input1, shape=input_shape))
    request.inputs['main/arg-1'].CopyFrom(tf.make_tensor_proto(input2, shape=input_shape))
    for i in range(100):
      start = time.time()
      output = stub.Predict(request, 100.0)  # timeout seconds
      end = time.time()
      result = np.reshape(output.outputs['main/mm.add:0'].float_val, [1, 4])
      print('{} latency: {}. result is {}: '.format(i, end - start, np.allclose(expected_result, result, rtol=0, atol=1e-6)))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python {} ip:port".format(sys.argv[0]))
        sys.exit(-1)
    ip_port = sys.argv[1]
    grpc_client_build_and_run(ip_port)
