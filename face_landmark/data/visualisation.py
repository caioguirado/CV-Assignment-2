def visualize_image(image, landmarks):
    plt.figure(figsize = (5, 5))
    image = (image - image.min())/(image.max() - image.min())

    landmarks = landmarks.view(-1, 2)
    landmarks = (landmarks + 0.5) * augProcessor.dim

    plt.imshow(image[0], cmap = 'gray')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s = 25, c = 'dodgerblue')
    plt.axis('off')
    plt.show()
    
def visualize_batch(images_list, landmarks_list, size = 14, shape = (6, 6), title = None, save = None):
    fig = plt.figure(figsize = (size, size))
    grid = ImageGrid(fig, 111, nrows_ncols = shape, axes_pad = 0.08)
    for ax, image, landmarks in zip(grid, images_list, landmarks_list):
        image = (image - image.min())/(image.max() - image.min())

        landmarks = landmarks.view(-1, 2)
        landmarks = (landmarks + 0.5) * augProcessor.dim
        landmarks = landmarks.numpy().tolist()
        landmarks = np.array([(x, y) for (x, y) in landmarks if 0 <= x <= augProcessor.dim and 0 <= y <= augProcessor.dim])

        ax.imshow(image[0], cmap = 'gray')
        ax.scatter(landmarks[:, 0], landmarks[:, 1], s = 10, c = 'dodgerblue')
        ax.axis('off')

    if title:
        print(title)
    if save:
        plt.savefig(save)
    plt.show()

"""# Initialise preprocessor, train and test img"""

augProcessor = DataAugPreprocessor(
    dim = 128, 
    bright = 0.24,
    saturation = 0.3,
    contrast = 0.15,
    hue = 0.14,
    angle = 14,
    face_offset = 32,
    crop_offset = 16
)

train_imgs = DatasetLandmarks(augProcessor, train=True)
test_imgs = DatasetLandmarks(augProcessor, train=False)

"""# __Visualizing a sample image, with random augmentations__"""

image1, landmarks1 = train_imgs[36]
visualize_image(image1, landmarks1)

image2, landmarks2 = train_imgs[36]
visualize_image(image2, landmarks2)

image3, landmarks3 = train_imgs[36]
visualize_image(image3, landmarks3)

"""# Prepare Data for training and visulise intermediate"""

len_val = int(0.1*len(train_imgs))
len_train = len(train_imgs) - len_val

print(f'{len_train} images for training')
print(f'{len_val} images for validating')
print(f'{len(test_imgs)} images for testing')

train_imgs, val_imgs = random_split(train_imgs, [len_train, len_val])

batch_size = 32
train_data = DataLoader(train_imgs, batch_size=batch_size, shuffle=True)
test_data = DataLoader(test_imgs, batch_size=2*batch_size, shuffle=False)
val_data = torch.utils.data.DataLoader(val_imgs, batch_size=2*batch_size, shuffle=False)

type(train_data)

for x, y in train_data:
    break

print(x.shape, y.shape, x.max(), x.min(), y.max(), y.min())

for x, y in val_data:
    break

print(x.shape, y.shape, x.max(), x.min(), y.max(), y.min())

for x, y in test_data:
    break

print(x.shape, y.shape, x.max(), x.min(), y.max(), y.min())

visualize_batch(x[:16], y[:16], shape = (4, 4), size = 8, title = 'Training Batch Samples')
