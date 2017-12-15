from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024

input_size = 128

max_epochs = 100
batch_size = 2

# orig_width = 1918
# orig_height = 1280
orig_width = 735
orig_height = 500

threshold = 0.5

model_factory = get_unet_128
