from torchvision import transforms
from imgaug import augmenters as iaa
from imgaug import parameters as iap

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

aug_pipeline = iaa.Sequential([
        iaa.Sometimes(0.5,
            iaa.SomeOf((0, 5),
                        [
                            # iaa.OneOf([
                            #         iaa.GaussianBlur((0, 1.0)),
                            #         # blur images with a sigma between 0 and 3.0
                            #         iaa.AverageBlur(k=(3, 5)),
                            #         # blur image using local means with kernel sizes between 2 and 7
                            #         iaa.MedianBlur(k=(3, 5)),
                            #         # blur image using local medians with kernel sizes between 2 and 7
                            # ]),

                            # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),
                            # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                            # iaa.AdditiveGaussianNoise(loc=0,
                            #                             scale=(0.0, 0.01 * 255),
                            #                             per_channel=0.5),

                            # # add gaussian noise to images
                            # iaa.OneOf([
                            #         iaa.Dropout((0.01, 0.05), per_channel=0.5),

                            #         # randomly remove up to 10% of the pixels
                            #         iaa.CoarseDropout((0.01, 0.03),
                            #                             size_percent=(0.01, 0.02),
                            #                             per_channel=0.2),
                            # ]),
                            # iaa.Add((-2, 2), per_channel=0.5),
                            # iaa.OneOf([
                            #         iaa.Multiply((0.9, 1.1), per_channel=0.5),
                            #         iaa.FrequencyNoiseAlpha(
                            #                 exponent=(-1, 0),
                            #                 first=iaa.Multiply((0.9, 1.1),
                            #                                     per_channel=True),
                            #                 second=iaa.ContrastNormalization(
                            #                         (0.9, 1.1))
                            #         )
                            # ]),
                            # # move pixels locally around (with random strengths)
                            # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                            # # sometimes move parts of the image around
                            # sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                        ],
                    random_order=True
            )
        )
    ],
    random_order=False # apply the augmentations in random order
)

transformer_train = transforms.Compose([
    aug_pipeline.augment_image,
    transforms.ToTensor()
])

transformer_val = transforms.Compose([
    transforms.ToTensor()
])