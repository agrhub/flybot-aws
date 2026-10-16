"""FlyBot Sentinel: OpenCV 5 perception + agent for fruit-fly trap monitoring."""

REQUIRED_OPENCV_MAJOR = 5


def require_opencv5() -> str:
    """Fail closed unless the imported OpenCV is 5.x (competition requirement)."""
    import cv2

    version = cv2.__version__
    if not version.startswith(f"{REQUIRED_OPENCV_MAJOR}."):
        raise RuntimeError(f"OpenCV {REQUIRED_OPENCV_MAJOR}.x required, found {version}")
    return version
