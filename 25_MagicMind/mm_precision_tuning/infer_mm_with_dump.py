import magicmind.python.runtime as mm
import numpy
import sys
 
from magicmind.python.runtime import Context, DumpMode, FileFormat
 
def infer_model(model, dev_id):
    # Switch device
    dev = mm.Device()
    dev.id = dev_id
    assert dev.active().ok()
 
    # Create engine, context and queue
    engine = model.create_i_engine()
    context = engine.create_i_context()
    queue = dev.create_queue()
 
    # Prepare inputs outputs tensor
    inputs = context.create_inputs()
    input_dims = model.get_input_dimensions()
    img = numpy.load("./bin_file/test.npy")
    inputs[0].from_numpy(img)
    outputs = context.create_outputs(inputs)
	
	# set dump_info 
    dump_info = Context.ContextDumpInfo(path='./', tensor_name=[], dump_mode=DumpMode.kOutputTensors, file_format=FileFormat.kBinary)
    dump_info.val.path = "./mlu_dump_dir"
    dump_info.val.file_format = FileFormat.kBinary
    context.set_context_dump_info(dump_info)
 
    # Launch model
    assert context.enqueue(inputs, outputs, queue).ok()
    assert queue.sync().ok()
    pred = numpy.array(outputs[0].asnumpy())
 
if __name__ == "__main__":
 
    # Load model from file
    graph_path = sys.argv[1]
    model = mm.Model()
    model.deserialize_from_file(graph_path)
 
    # Run on every device
    with mm.System() as mm_sys:
        for i in range(mm_sys.device_count()):
            infer_model(model, i)
