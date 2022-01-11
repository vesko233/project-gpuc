import numpy as np 
import matplotlib.image as mpimg

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

# Meta data 
meta_file = r'cifar-10-batches-py/batches.meta'
meta_data = unpickle(meta_file)
label_names = meta_data['label_names']

# Save information
info_file = open("images_info.csv", "w")
for i in range(5):
    # getting data
    filename_source = 'cifar-10-batches-py/data_batch_' + str(i+1)
    data = unpickle(filename_source)

    # saving image
    for j in range(len(data['data'])):
        image = data['data'][j]
        image = image.reshape(3,32,32)
        image = image.transpose(1,2,0)
        image_name = "images_batch_" + str(i+1) + "/" + data['filenames'][j]
        # writing image and image info
        mpimg.imsave(image_name, image)
        lab = data['labels'][j]
        info_file.write(data['batch_label'] + "," + data['filenames'][j] + "," + label_names[lab] + "," + str(lab) + "\n")

info_file.close()