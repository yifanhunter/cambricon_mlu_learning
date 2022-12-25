import numpy.random

import magicmind.python.runtime as mm


def construct_conv_relu_network(input_dim: mm.Dims,
                                filter_data: numpy.ndarray,
                                bias_data: numpy.ndarray,
                                stride: list,
                                pad: list,
                                dialation: list,
                                grapth_name="graph") -> mm.Model:
    main_type = mm.DataType.FLOAT32

    # Create builder and network
    builder = mm.Builder()
    network = mm.Network()

    # Set input tensor
    input_tensor = network.add_input(main_type, input_dim)

    # Create filter node
    filter = network.add_i_const_node(main_type, mm.Dims(filter_data.shape), filter_data)

    # Create bias node
    bias = network.add_i_const_node(main_type, mm.Dims(bias_data.shape), bias_data)

    # Create conv node
    conv = network.add_i_conv_node(input_tensor, filter.get_output(0), bias.get_output(0))
    assert conv.set_stride(stride[0], stride[1]).ok()
    assert conv.set_pad(pad[0], pad[1], pad[2], pad[3]).ok()
    assert conv.set_dilation(dialation[0], dialation[1]).ok()
    assert conv.set_padding_mode(mm.IPaddingMode.EXPLICIT).ok()

    # Create relu node
    relu = network.add_i_activation_node(conv.get_output(0), mm.IActivation.RELU)

    # Mark output to make network valid
    assert network.mark_output(relu.get_output(0))

    # Build and save model to local storage
    model = builder.build_model("conv_relu_model", network)
    assert model.serialize_to_file(grapth_name).ok()

    return model


if __name__ == "__main__":
    # Set params
    input_dim = mm.Dims([1, 224, 224, 3])
    filter_data = numpy.random.rand(256, 3, 3, 3)
    bias_data = numpy.random.rand(256)

    stride = [2, 2]
    pad = [0, 0, 0, 0]
    dilation = [1, 1]
    construct_conv_relu_network(input_dim, filter_data, bias_data, stride, pad, dilation)
