import numpy as np

from experiments import isaac_asset_validation as v


def image(blobs):
    img = np.full((60, 100, 3), 200, np.uint8)
    for side, col in blobs.items():
        x = {"left": 20, "right": 75}[col]
        img[20:30, x : x + 6] = v.MARKERS[side][2]
    return img


def test_marker_order_matches_each_camera_side():
    # camera on +x: the -y marker is on the left, +y on the right
    assert v.marker_check(image({"-y": "left", "+y": "right"}), "px") is True
    assert v.marker_check(image({"-y": "right", "+y": "left"}), "px") is False  # lagged view
    assert v.marker_check(image({"+x": "left", "-x": "right"}), "py") is True
    assert v.marker_check(image({"+y": "left", "-y": "right"}), "nx") is True
    assert v.marker_check(image({"-x": "left", "+x": "right"}), "ny") is True


def test_missing_marker_gives_no_verdict():
    assert v.marker_check(image({"-y": "left"}), "px") is None
