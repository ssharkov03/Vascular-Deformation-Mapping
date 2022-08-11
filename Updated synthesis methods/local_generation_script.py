from volumentations import *

import SimpleITK as sitk
import matplotlib.pyplot as plt
from math import floor, ceil
import numpy as np
from random import randint
import os


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


def save_image_plot(image, path):
    int_roi = image.astype('int8')
    int_roi_min = floor(np.min(int_roi))
    int_roi_max = ceil(np.max(int_roi))
    hist, bin_edges = np.histogram(int_roi, bins=range(int_roi_min, int_roi_max))
    plt.bar(bin_edges[:-1], hist, width=1)
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.savefig(path)
    plt.close()


def save_image(image_np, image_params, path):
    sitk.WriteImage(ImageTranslation.array_to_image(image_np, image_params), path)


def pipeline(original_image, original_label, gammas_for_image, gammas_for_roi, results_path, num_iterations,
             img_number):
    """

    :param original_image: SimpleITK image to get augmentation from, must be >= target image size (cropped img) + 50 (along all axes)
    :param original_label: SimpleITK label for original image, same size as original image
    :param gammas_for_image: list of gammas for image generation
    :param gammas_for_roi: list of gammas for region of interest (here - thrombus masses)
    :param results_path: path to save data to
    :param num_iterations: num of generated images, must be equal to len(gammas_for_image) and len(gammas_for_roi)
    :param img_number: string, image identifier
    :return: returns nothing, only saves needed data
    """


    assert len(gammas_for_image) == len(gammas_for_roi) == num_iterations

    if 'intensity_hists' not in os.listdir(results_path):
        os.mkdir(os.path.join(results_path, 'intensity_hists'))

    image_results_path = os.path.join(results_path, 'intensity_hists')

    print(f"Image {img_number}:")
    print("\tInitializing params.")

    target_shape = ImageUtils.get_target_shape(sitk.GetArrayFromImage(original_label))
    roi_label = 2
    x_start = target_shape[0][0]
    x_finish = target_shape[0][1]
    y_start = target_shape[1][0]
    y_finish = target_shape[1][1]
    z_start = target_shape[2][0]
    z_finish = target_shape[2][1]
    slice_threshold = 50

    original_image_np, image_params = ImageTranslation.image_to_array(original_image)
    original_label_np, label_params = ImageTranslation.image_to_array(original_label)

    save_image_plot(original_image_np, os.path.join(image_results_path, f"original_image_{img_number}"))
    print("\tCropping images.")

    thresh = slice_threshold
    cropped_image_np = original_image_np[z_start - thresh:z_finish + thresh, y_start - thresh:y_finish + thresh,
                       x_start - thresh:x_finish + thresh]
    cropped_label_np = original_label_np[z_start - thresh:z_finish + thresh, y_start - thresh:y_finish + thresh,
                       x_start - thresh:x_finish + thresh]

    save_image(cropped_image_np[thresh:-thresh, thresh:-thresh, thresh:-thresh], image_params,
               os.path.join(results_path, f"original_image_{img_number}.nii.gz"))
    save_image(cropped_label_np[thresh:-thresh, thresh:-thresh, thresh:-thresh], image_params,
               os.path.join(results_path, f"original_label_{img_number}.nii.gz"))

    save_image_plot(cropped_image_np, os.path.join(image_results_path, f"original_cropped_image_{img_number}"))

    print("\tCorrecting image intensity.")

    window_min = floor(np.min(cropped_image_np))
    window_max = ceil(np.max(cropped_image_np))
    new_img = sitk.Cast(
        sitk.IntensityWindowing(ImageTranslation.array_to_image(cropped_image_np, image_params),
                                windowMinimum=int(window_min),
                                windowMaximum=int(window_max),
                                outputMinimum=0.0, outputMaximum=255.0), sitk.sitkFloat32)

    changed_intensity_img_np, img_params = ImageTranslation.image_to_array(new_img)

    save_image(changed_intensity_img_np[thresh:-thresh, thresh:-thresh, thresh:-thresh], image_params,
               os.path.join(results_path, f"original_image_changed_intensity_{img_number}.nii.gz"))
    save_image_plot(changed_intensity_img_np, os.path.join(image_results_path, f"original_image_changed_intensity_{img_number}"))

    for iteration in range(num_iterations):

        print(f"\tIteration {iteration}:")

        elastic_augmentation, gamma_augmentation, label_based_augmentation = (
            Compose([
                ElasticTransform((0.2, 1.7), interpolation=1, always_apply=True)
            ], p=1.0),

            Compose([
                RandomGamma(gamma_limit=(gammas_for_image[iteration], gammas_for_image[iteration] + 1),
                            always_apply=True)
            ], p=1.0),

            Compose([
                RandomGamma(gamma_limit=(gammas_for_roi[iteration], gammas_for_roi[iteration] + 1), always_apply=True),
                GaussianNoise(var_limit=(1, 5), p=0.8)
            ], p=1.0)
        )

        print("\t\tElastic augmentation started.")

        data = {'image': changed_intensity_img_np, 'mask': cropped_label_np}
        aug_data = elastic_augmentation(**data)
        elastic_transformed_image_np, elastic_transformed_label_np = aug_data['image'], aug_data['mask']

        save_image(elastic_transformed_image_np[thresh:-thresh, thresh:-thresh, thresh:-thresh], image_params,
                   os.path.join(results_path, f"elastic_transformed_image_{img_number}_{iteration}.nii.gz"))
        save_image_plot(elastic_transformed_image_np,
                        os.path.join(image_results_path, f"elastic_transformed_image_{img_number}_{iteration}"))

        print("\t\tChecking for NaN in roi and image...")

        elastic_transformed_roi_np = np.where(elastic_transformed_label_np == 2, elastic_transformed_image_np, 0)
        elastic_transformed_roi_np = np.where((elastic_transformed_roi_np < 0) | (np.isnan(elastic_transformed_roi_np)),
                                              0,
                                              elastic_transformed_roi_np)
        elastic_transformed_image_np = np.where(
            (elastic_transformed_image_np < 0) | (np.isnan(elastic_transformed_image_np)), 0,
            elastic_transformed_image_np)

        if not np.isnan(elastic_transformed_roi_np).any():
            print("\t\tNo NaN in roi.")
        else:
            raise ValueError("\t\tNaN in roi found!")

        if not np.isnan(elastic_transformed_image_np).any():
            print("\t\tNo NaN in image.")
        else:
            raise ValueError("\t\tNaN in roi found!")

        save_image(elastic_transformed_label_np[thresh:-thresh, thresh:-thresh, thresh:-thresh], image_params,
                   os.path.join(results_path, f"augmented_label_{img_number}_{iteration}.nii.gz"))
        save_image_plot(elastic_transformed_roi_np, os.path.join(image_results_path, f"elastic_transformed_roi_{img_number}_{iteration}"))

        print("\t\tImage gamma augmentation.")

        data = {'image': elastic_transformed_image_np}
        aug_data = gamma_augmentation(**data)
        overall_transformed_image_np = aug_data['image']

        save_image(overall_transformed_image_np[thresh:-thresh, thresh:-thresh, thresh:-thresh], image_params,
                   os.path.join(results_path, f"overall_transformed_image_{img_number}_{iteration}.nii.gz"))
        save_image_plot(overall_transformed_image_np,
                        os.path.join(image_results_path, f"overall_transformed_image_{img_number}_{iteration}"))

        print("\t\tROI based augmentation.")
        data = {'image': elastic_transformed_roi_np}
        aug_data = label_based_augmentation(**data)
        augmented_roi_np = aug_data['image']

    # save_image(augmented_roi_np[thresh:-thresh, thresh:-thresh, thresh:-thresh], image_params, results_path + f"augmented_roi_np_{iteration}.nii.gz")
        save_image_plot(augmented_roi_np, os.path.join(image_results_path, f"augmented_roi_{img_number}_{iteration}"))

        print("\t\tGetting resulting augmented image.")

        augmented_image_np = np.where(elastic_transformed_label_np == 2, augmented_roi_np, overall_transformed_image_np)

        save_image(augmented_image_np[thresh:-thresh, thresh:-thresh, thresh:-thresh], image_params,
                   os.path.join(results_path, f"augmented_image_{img_number}_{iteration}.nii.gz"))
        save_image_plot(augmented_image_np, os.path.join(image_results_path, f"augmented_image_{img_number}_{iteration}"))

        print("\t\tIteration Done.\n")

#         print("\tCropping images.")

#         augmented_image_np = augmented_image_np[thresh:-thresh, thresh:-thresh, thresh:-thresh]
#         overall_transformed_image_np = overall_transformed_image_np[thresh:-thresh, thresh:-thresh, thresh:-thresh]
#         elastic_transformed_label_np = elastic_transformed_label_np[thresh:-thresh, thresh:-thresh, thresh:-thresh]
    print("Image done.\n\n")


def show_image(image_np, image_params):
    sitk.Show(ImageTranslation.array_to_image(image_np, image_params))


if __name__ == '__main__':
    input_path = "D:\\MyProjects\\bmm2022\\data\\niiGZdata"
    result_path = "D:\\MyProjects\\bmm2022\\On github\\Generate images\\ResultsDir\\Synthesis"
    for folder in os.listdir(input_path):
        for state in os.listdir(os.path.join(input_path, folder)):
            image_folder_name = folder + "_" + state
            if image_folder_name in os.listdir(result_path):
                continue
            else:
                resulting_path_for_images = os.path.join(result_path, image_folder_name)
                os.mkdir(resulting_path_for_images)

            img_path = os.path.join(input_path, folder, state, "image.nii.gz")
            lbl_path = os.path.join(input_path, folder, state, "label.nii.gz")

            img = sitk.ReadImage(img_path)
            lbl = sitk.ReadImage(lbl_path)
            iterations = 5
            image_gamma = np.random.randint(low=80, high=120, size=(5,))
            roi_gamma = np.array([randint(x - 10, x) for x in image_gamma])

            pipeline(img, lbl, image_gamma, roi_gamma, resulting_path_for_images, iterations, image_folder_name)