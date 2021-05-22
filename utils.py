def conv_out_shape(win, padding, dilation, kernel_size, stride):
    return (win + 2*padding - dilation * (kernel_size-1) - 1)/stride + 1

def pool_out_shape(win, padding, dilation, kernel_size, stride):
    return (win + 2*padding - dilation * (kernel_size-1) - 1)/stride + 1