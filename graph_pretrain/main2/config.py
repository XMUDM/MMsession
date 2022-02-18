import time

lr = 0.005
batch_size = 256

in_dim = 50  # word embedding size
in_dim2 = 512  # img embedding size
hidden_dim = 32
out_dim = 32
layer_size = [32 * 2, 32, 1]

save_path = '../save2/{}_lr{}_bs{}'.format(time.strftime("%Y%m%d"), lr, batch_size)


