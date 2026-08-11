from torch import nn

def get_conv_kernel(conv: nn.Module):
    conv = nn.utils.remove_weight_norm(conv)
    return conv.weight.detach().clone().cpu()

def combine_conv1d(w1, w2):
    """
    Element-wise (true) convolution.
    Refer to https://stackoverflow.com/a/58357816/
    """
    return nn.functional.conv1d(
        w1.permute(1, 0, 2), 
        w2.flip(-1), # flip because this is actually correlation
        stride=1, 
        padding=w1.shape[-1] - 1
    ).permute(1, 0, 2)

def create_virtual_kernel(convs: list[nn.Module] | tuple[nn.Module], control=False):
    combined = get_conv_kernel(convs[0])
    for conv in convs[1:]:
        if control: # load up the dead salmon
            conv_kernel = nn.init.kaiming_uniform_(get_conv_kernel(conv))
        else: 
            conv_kernel = get_conv_kernel(conv)
        combined = combine_conv1d(combined, conv_kernel)
    return combined