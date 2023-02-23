import magicmind.python.runtime as mm
import sys
import numpy as np
 
 
def infer_model(graph_path: str, logdir_name: str) -> bool:
    model = mm.Model()
    model.deserialize_from_file(graph_path)
    engine = model.create_i_engine()
    context = engine.create_i_context()
 
    dev = mm.Device()
    dev.id = 0
    dev.active()
    queue = dev.create_queue()
 
    # inputs = context.create_inputs()
    # assert isinstance(inputs, list)
    # for i, dim in enumerate(model.get_input_dimensions()):
        # shape = []
        # for v in dim.GetDims():
            # if v == -1:
                # shape.append(10)
            # else:
                # shape.append(v)
        # inputs[i].from_numpy(np.ones(shape))
    # outputs = context.create_outputs(inputs)
    # assert isinstance(outputs, list)
	
	# Prepare inputs outputs tensor
	
    profiler_options = mm.ProfilerOptions(mm.HostTracerLevel.kCritical, mm.DeviceTracerLevel.kOn)
    profiler = mm.Profiler(profiler_options, logdir_name)
    assert profiler.start(), 'profiler start failed!'
	
    inputs = context.create_inputs()
    input_dims = model.get_input_dimensions()
    img = np.load("./bin_file/test.npy")
    inputs[0].from_numpy(img)
    outputs = context.create_outputs(inputs)
    for i in range(0, 10):
        assert profiler.step_begin(1), 'profiler step_begin failed!'
        assert context.enqueue(inputs, outputs, queue).ok(), 'profiler enqueue failed!'
        assert queue.sync().ok(), 'profiler sync queue failed'
        assert profiler.step_end(), 'profiler step_end failed!'
    assert profiler.stop(), 'profiler stop failed'
 
    return True
 
 
def main():
    if len(sys.argv) != 3:
        print(
            'Arguments format error.\n example: profiling_test.py path/to/graph path/to/data path/to/profiler_output'
        )
    else:
        graph_name = sys.argv[1]
        logdir_name = sys.argv[2]
    assert infer_model(graph_name, logdir_name), 'infer model failed!'
    
 
if __name__ == "__main__":
    main()
