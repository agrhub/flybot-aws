import cv2

from sentinel import require_opencv5


def test_opencv_is_5():
    assert require_opencv5() == cv2.__version__
    assert cv2.__version__.startswith("5.")


def test_opencv5_modules_present():
    # Modules the plan relies on; a failure here changes the design (docs/DECISIONS.md)
    assert hasattr(cv2, "dnn")
    assert hasattr(cv2, "SIFT_create")
