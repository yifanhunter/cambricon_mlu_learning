import os
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2

def gen_warmup_file(predict_request_list, export_path):
  with tf.io.TFRecordWriter(os.path.join(export_path, 'warmup_requests_data')) as writer:
    for predict_request in predict_request_list:
      log = prediction_log_pb2.PredictionLog(
          predict_log=prediction_log_pb2.PredictLog(request=predict_request))
      print(log)
      writer.write(log.SerializeToString())
    print("generate {}/tf_serving_warmup_requests succeed.".format(export_path))


# modify inputs and shapes to what you need
if __name__ == '__main__':
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'sample_add'

  request_list = []
  for batch_size in [1]:
    input1 = np.random.rand(batch_size, 4)
    input2 = np.random.rand(batch_size, 4)

    request.inputs['main/arg-0'].CopyFrom(tf.make_tensor_proto(input1.astype(np.float32),
        shape=[batch_size, 4]))
    request.inputs['main/arg-1'].CopyFrom(tf.make_tensor_proto(input2.astype(np.float32),
        shape=[batch_size, 4]))
    request_list.append(request)

  gen_warmup_file(request_list, './')
