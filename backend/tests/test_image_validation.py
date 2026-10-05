import io
import pytest
from PIL import Image
from app.core.errors import InvalidImageError
from app.services.image_validation import validate_image_bytes, save_upload


def _img(fmt="JPEG", size=(100, 100)):
    b = io.BytesIO(); Image.new("RGB", size, (10, 120, 30)).save(b, fmt); return b.getvalue()

def test_valid_jpeg_and_png():
    assert validate_image_bytes(_img("JPEG"), 10**7)[1] == ".jpg"
    assert validate_image_bytes(_img("PNG"), 10**7)[1] == ".png"

def test_text_file_renamed_as_jpg_rejected():
    with pytest.raises(InvalidImageError): validate_image_bytes(b"print('hi')", 10**7)

def test_empty_rejected():
    with pytest.raises(InvalidImageError): validate_image_bytes(b"", 10**7)

def test_oversize_rejected():
    with pytest.raises(InvalidImageError): validate_image_bytes(_img(), 100)

def test_truncated_image_rejected():
    with pytest.raises(InvalidImageError): validate_image_bytes(_img()[:200], 10**7)

def test_wrong_format_gif_rejected():
    with pytest.raises(InvalidImageError): validate_image_bytes(_img("GIF"), 10**7)

def test_tiny_image_rejected():
    with pytest.raises(InvalidImageError): validate_image_bytes(_img(size=(20, 20)), 10**7)

def test_pixel_limit():
    with pytest.raises(InvalidImageError): validate_image_bytes(_img(size=(200, 200)), 10**7, max_pixels=1000)

def test_save_upload_uses_random_name(tmp_path):
    p = save_upload(b"x", tmp_path, ".jpg")
    assert p.suffix == ".jpg" and len(p.stem) == 32
