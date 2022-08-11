from volumentations import *
import SimpleITK as sitk
from math import floor, ceil
from random import randint


# class for operating with sitk image
class ImageTranslation:

    @staticmethod
    def get_image_params(image):
        return image.GetOrigin(), image.GetDirection(), image.GetSpacing()

    @staticmethod
    def image_to_array(image):
        return sitk.GetArrayFromImage(image), ImageTranslation.get_image_params(image)

    @staticmethod
    def array_to_image(array, params):
        image = sitk.GetImageFromArray(array)
        image.SetOrigin(params[0])
        image.SetDirection(params[1])
        image.SetSpacing(params[2])
        return image


class ImageUtils:

    @staticmethod
    def get_target_shape(label):
        # label shape is in form (z, y, x):
        orig_shape = label.shape

        z_start = None
        z_finish = None
        y_start = None
        y_finish = None
        x_start = None
        x_finish = None
        for z in range(orig_shape[0]):
            is_boarder = np.where(label[z] != 0, True, False)
            if z_start is None and is_boarder.any():
                z_start = z
            if z_start is not None and is_boarder.any():
                z_finish = z

        for y in range(orig_shape[1]):
            is_boarder = np.where(label[:, y, :] != 0, True, False)
            if y_start is None and is_boarder.any():
                y_start = y
            if y_start is not None and is_boarder.any():
                y_finish = y

        for x in range(orig_shape[2]):
            is_boarder = np.where(label[..., x] != 0, True, False)
            if x_start is None and is_boarder.any():
                x_start = x
            if x_start is not None and is_boarder.any():
                x_finish = x

        target_shape = ((x_start, x_finish), (y_start, y_finish), (z_start, z_finish))
        return target_shape


def generate_data(original_image, original_label):
    """
    :param original_image: SimpleITK image to get augmentation from, must be >= target image shape (cropped img) +- 50 (along all axes)
    :param original_label: SimpleITK label for original image, same size as original image
    :return: tuple: generated_image, generated_label, cropped_original_image, cropped_original_label
    """

    target_shape = ImageUtils.get_target_shape(sitk.GetArrayFromImage(original_label))
    x_start = target_shape[0][0]
    x_finish = target_shape[0][1]
    y_start = target_shape[1][0]
    y_finish = target_shape[1][1]
    z_start = target_shape[2][0]
    z_finish = target_shape[2][1]
    thresh = 50

    original_image_np, image_params = ImageTranslation.image_to_array(original_image)
    original_label_np, label_params = ImageTranslation.image_to_array(original_label)

    cropped_image_np = original_image_np[z_start - thresh:z_finish + thresh, y_start - thresh:y_finish + thresh, x_start - thresh:x_finish + thresh]
    cropped_label_np = original_label_np[z_start - thresh:z_finish + thresh, y_start - thresh:y_finish + thresh, x_start - thresh:x_finish + thresh]


    window_min = floor(np.min(cropped_image_np))
    window_max = ceil(np.max(cropped_image_np))
    new_img = sitk.Cast(
        sitk.IntensityWindowing(ImageTranslation.array_to_image(cropped_image_np, image_params),
                                windowMinimum=int(window_min),
                                windowMaximum=int(window_max),
                                outputMinimum=0.0, outputMaximum=255.0), sitk.sitkFloat32)

    changed_intensity_img_np, img_params = ImageTranslation.image_to_array(new_img)

    image_gamma = randint(80, 120)
    roi_gamma = randint(image_gamma - 10, image_gamma)

    elastic_augmentation, gamma_augmentation, label_based_augmentation = (
        Compose([
            ElasticTransform((0.2, 1.7), interpolation=1, always_apply=True)
        ], p=1.0),

        Compose([
            RandomGamma(gamma_limit=(image_gamma, image_gamma + 1),
                        always_apply=True)
        ], p=1.0),

        Compose([
            RandomGamma(gamma_limit=(roi_gamma, roi_gamma + 1), always_apply=True),
            GaussianNoise(var_limit=(1, 5), p=0.8)
        ], p=1.0)
    )

    data = {'image': changed_intensity_img_np, 'mask': cropped_label_np}
    aug_data = elastic_augmentation(**data)
    elastic_transformed_image_np, elastic_transformed_label_np = aug_data['image'], aug_data['mask']

    elastic_transformed_roi_np = np.where(elastic_transformed_label_np == 2, elastic_transformed_image_np, 0)
    elastic_transformed_roi_np = np.where((elastic_transformed_roi_np < 0) | (np.isnan(elastic_transformed_roi_np)), 0, elastic_transformed_roi_np)
    elastic_transformed_image_np = np.where((elastic_transformed_image_np < 0) | (np.isnan(elastic_transformed_image_np)), 0, elastic_transformed_image_np)


    data = {'image': elastic_transformed_image_np}
    aug_data = gamma_augmentation(**data)
    overall_transformed_image_np = aug_data['image']

    data = {'image': elastic_transformed_roi_np}
    aug_data = label_based_augmentation(**data)
    augmented_roi_np = aug_data['image']


    augmented_image_np = np.where(elastic_transformed_label_np == 2, augmented_roi_np, overall_transformed_image_np)

    augmented_image_np = augmented_image_np[thresh:-thresh, thresh:-thresh, thresh:-thresh]
    elastic_transformed_label_np = elastic_transformed_label_np[thresh:-thresh, thresh:-thresh, thresh:-thresh]

    return augmented_image_np, elastic_transformed_label_np, cropped_image_np, cropped_label_np


if __name__ == '__main__':
    original_image = sitk.ReadImage('your/path')
    original_label = sitk.ReadImage('your/path')

    augmented_image, elastic_transformed_label, cropped_image, cropped_label = generate_data(original_image, original_label)

