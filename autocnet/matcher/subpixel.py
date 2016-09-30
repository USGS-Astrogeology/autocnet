import numpy as np

from autocnet.matcher import naive_template

# TODO: look into KeyPoint.size and perhaps use to determine an appropriately-sized search/template.


def clip_roi(img, center, img_size):
    """
    Given an input image, clip a square region of interest
    centered on some pixel at some size.

    Parameters
    ----------
    img : ndarray or object
          The input image to be clipped or an object
          with a read_array method that takes a pixels
          argument in the form [xstart, ystart, xstop, ystop]

    center : tuple
             (y,x) coordinates to center the roi

    img_size : int
               Odd, total image size

    Returns
    -------
    clipped_img : ndarray
                  The clipped image
    """
    if img_size % 2 == 0:
        raise ValueError('Image size must be odd.')

    i = int((img_size - 1) / 2)

    x, y = map(int, center)

    y_start = y - i
    x_start = x - i
    x_stop = (x + i) - x_start
    y_stop = (y + i) - y_start

    if isinstance(img, np.ndarray):
        clipped_img = img[y_start:y_start + y_stop + 1,
                          x_start:x_start + x_stop + 1]
    else:
        clipped_img = img.read_array(pixels=[x_start, y_start,
                                             x_stop + 1, y_stop + 1])
    return clipped_img


def subpixel_offset(template, search, **kwargs):
    """
    Uses a pattern-matcher on subsets of two images determined from the passed-in keypoints and optional sizes to
    compute an x and y offset from the search keypoint to the template keypoint and an associated strength.

    Parameters
    ----------
    template : ndarray
               The template used to search

    search : ndarray
             The search image

    Returns
    -------
    x_offset : float
               Shift in the x-dimension

    y_offset : float
               Shift in the y-dimension

    strength : float
               Strength of the correspondence in the range [-1, 1]
    """

    x_offset, y_offset, strength = naive_template.pattern_match(template, search, **kwargs)
    return x_offset, y_offset, strength

'''
Stub for an observable subpixel class

class PatternMatch(Observable):

    """
    Attributes
    ----------
    df : dataframe
         A dataframe of point to be subpixel registered

    img1 : object or ndarray
           A file handle object or ndarray to use to subpixel register

    img2 : object or ndarray
           A file handle object or ndarray to use to subpixel register

    destination : object
                  Destination node

    threshold_mask : series
                     A pandas series masking values < threshold

    shift_mask : series
                 A pandas series masking values with shifts larger
                 than the allowed x, y shifts

    subpixel_mask : series
                    A composite mask, threshold_mask & shift_mask
    """

    def __init__(self, img1, img2, df, min_x_shift=-1.0, max_x_shift=1.0,
                 min_y_shift=-1.0, max_y_shift=1.0, threshold=0.8):
        self.img1 = img1
        self.img2 = img2
        self.df = df

        self._min_x_shift = min_x_shift
        self._min_y_shift = min_y_shift
        self._max_x_shift = max_x_shift
        self._max_y_shift = max_y_shift
        self._threshold = threshold

        self.threshold_mask = pd.Series(True, index=self.df.index)
        self.shift_mask = pd.Series(True, index=self.df.index)
        self.subpixel_mask = self.threshold_mask & self.subpixel_mask

        self._action_stack = deque(maxlen=20)
        self._current_action_stack = 0
        self._observers = set()
        self.attrs = ['threshold', 'min_x_shift', 'max_x_shift',
                      'min_y_shift', 'm_y_shift', 'threshold_mask',
                      'shift_mask', 'subpixel_mask']

    def clip_roi(self, img, center):

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, v):
        if 0 <= v <= 1:
            self._threshold = v

            # Update the mask here
            self.threshold_mask = self.d

            current_state = self._action_stack[self._current_action_stack]
            current_state['threshold'] = self.threshold

            self._update_stack(current_state)

    @property
    def min_x_shift(self):
        return self._min_x_shift

    @min_x_shift.setter
    def min_x_shift(self, v):
        self._min_x_shift = v

        # Update mask here

        current_state = self._action_stack[self._current_action_stack]
        current_state['min_x_shift'] = self.min_x_shift
        self._update_stack()

    @property
    def min_y_shift(self):
        return self._min_y_shift

    @min_x_shift.setter
    def min_y_shift(self, v):
        self._min_y_shift = v

        # Update mask here

        current_state = self._action_stack[self._current_action_stack]
        current_state['min_y_shift'] = self.min_y_shift
        self._update_stack()

    @property
    def max_x_shift(self):
        return self._max_x_shift

    @max_x_shift.setter
    def max_x_shift(self, v):
        self._max_x_shift = v

        # Update mask here

        current_state = self._action_stack[self._current_action_stack]
        current_state['max_x_shift'] = self.max_x_shift
        self._update_stack()

    @property
    def max_y_shift(self):
        return self._max_y_shift

    @max_y_shift.setter
    def max_y_shift(self, v):
        self._max_y_shift = v

        # Update mask here

        current_state = self._action_stack[self._current_action_stack]
        current_state['max_y_shift'] = self.max_y_shift
        self._update_stack()
'''
