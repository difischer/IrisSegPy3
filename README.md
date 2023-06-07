# IrisSegPy3
Python 3 updated version of Banerjee and Mery (2015).  GAC &amp; GrabCut iris segmentation code.

The original code for this algorithm (available at: https://github.com/sbanerj1/IrisSeg ) was in python 2 and used the old ``cv`` library.
Keep in mind that the code still haves reminents from the previous verison so some variables may be unused, now the code can be imported and used as a function.
This code is in Python 3 and uses cv2, in order to update both codes:

* All the print statements were updated to print function.
* All the cv functions were updated to cv2 functions.
* Some division opeators ``/`` where updated to ``//`` when needed.
* All xrange were updated to range.

Also:
* Unused libraries were removed.
* ``contour_iterator()`` was removed.
* ``test_iris()`` and ``test_pupil()`` now recive an image and return coordinates instead of changing global variables.
* ``FitEllipse`` class was covnerted to a function ``image_fitElipse()``.
* ``silent`` clause to ``evolve_visual`` was added.
* the trackbar was removed.

Usage: 

```
import cv2
from IrisSeg import IrisSeg

image = cv2.imread(image_path)
ellipsis_params = IrisSeg(image)
```

``ellipsis_params`` is a list that contains ``(center, size, angle)`` of the fitted ellipsis
