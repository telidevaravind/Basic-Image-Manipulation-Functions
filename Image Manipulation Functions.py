import numpy as np



def convert_to_grayscale(im):
    
    grayscale_image = np.multiply(0.2989, im[:,:,0]) + np.multiply(0.5870, im[:,:,1]) + np.multiply(0.1140, im[:,:,2])
    
    return grayscale_image

'''

	Converts an (nxmx3) color image im into a (nxm) grayscale image.


'''
 
def crop_image(im, crop_bounds):
#    print(crop_bounds)
#    print(im.shape)
    im_cropped = im[crop_bounds[0]:im.shape[0] - crop_bounds[1],crop_bounds[2]:im.shape[1] - crop_bounds[3],:]
#    cv2.imshow('i',im_cropped)
#    cv2.waitKey(0)
    return im_cropped

'''

	Returns a cropped image, im_cropped. 

	im = numpy array representing a color or grayscale image.

	crops_bounds = 4 element long list containing the top, bottom, left, and right crops respectively. 



	e.g. if crop_bounds = [50, 60, 70, 80], the returned image should have 50 pixels removed from the top, 

	60 pixels removed from the bottom, and so on. 



	'''

def compute_range(im):
    if len(im.shape) == 2:
        ma = np.max(im)
        mi = np.min(im)
        image_range = ma-mi
    elif len(im.shape) == 3:
        imag = np.zeros((2, len(im.shape)), int)
        imag[0,:] = [np.max(im[:,:,0]),np.max(im[:,:,1]),np.max(im[:,:,2])]
        imag[1,:] = [np.min(im[:,:,0]),np.min(im[:,:,1]),np.min(im[:,:,2])]
        image_range = imag[0,:] - imag[1,:]
    return image_range

'''

	Returns the difference between the largest and smallest pixel values.

'''


def maximize_contrast(im, target_range = [0, 255]):
    image_adjusted = im
    
    ## IF WE HAVE TO CHANGE THE RANGE OF THE ENTIRE PIXELS
    
#    im = im.astype(float)
#    old_range = np.max(im) - np.min(im)
#    new_range = target_range[1] - target_range[0]
##    image_adjusted = ((im - np.min(im))/old_range)*new_range + target_range[0]
#    for i in range(0, im.shape[-1]):
#        old_range = np.max(im[:,:,i]) - np.min(im[:,:,i])
#        image_adjusted[:,:,i] = ((im[:,:,i] - np.min(im[:,:,i]))/old_range)*new_range + target_range[0]
#    image_adjusted = image_adjusted.astype('uint8')
    
    for i in range(0, im.shape[-1]):
        pixel_min = np.min(im[:,:,i])
        a = np.where(im[:,:,i]==pixel_min)
        coordinates = list(zip(a[0], a[1]))
        for cord in coordinates:
            image_adjusted[cord[0],cord[1],i] = target_range[0]
        pixel_max = np.max(im[:,:,i])
        b = np.where(im[:,:,i]==pixel_max)
        coordinates = list(zip(b[0], b[1]))
        for cord in coordinates:
            image_adjusted[cord[0],cord[1],i] = target_range[1]
    return image_adjusted
'''

	Return an image over same size as im that has been "contrast maximized" by rescaling the input image so 

	that the smallest pixel value is mapped to target_range[0], and the largest pixel value is mapped to target_range[1]. 

'''


def flip_image(im, direction = 'vertical'):
    flipped_image = np.flipud(im)
    return flipped_image

'''

	Flip image along direction indicated by the direction flag. 

	direction = vertical or horizontal.

'''


def count_pixels_above_threshold(im, threshold):
    i = im>threshold
    pixels_above_threshold = (np.count_nonzero(i))
    return pixels_above_threshold
'''

	Return the number of pixels with values above threshold.

'''


def normalize(im):
    normalized_image = im.astype(float)
    normalized_image = (normalized_image - np.mean(normalized_image)) / np.max(normalized_image)
    return normalized_image
'''

	Rescale all pixels value to make the mean pixel value equal zero and the standard deviation equal to one. 

	if im is of type uint8, convert to float.

'''


def resize_image(im, scale_factor):
    scaled_width = im.shape[0] * scale_factor
    scaled_height = im.shape[1] * scale_factor
    scaled_image = np.zeros((scaled_width, scaled_height, im.shape[2]), float)
    for  i in range(0, im.shape[2]):
        for j in range(0,scaled_width):
            for k in range(0,scaled_height):
                new_width = int(np.floor( j * im.shape[0] / scaled_width ))
                new_height = int(np.floor( k * im.shape[1] / scaled_height ))
                scaled_image[j][k][i] = im[new_width][new_height][i]
    return scaled_image
