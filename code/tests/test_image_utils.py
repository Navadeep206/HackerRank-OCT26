"""Unit tests for safe image path handling and normalization."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from code.image_utils import ImageLoadError, normalize_dataset_image, normalize_image, resolve_image_path

class ImageUtilsTests(unittest.TestCase):
    def test_detects_png_with_jpg_extension_and_normalizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            source = root / "images" / "misleading.jpg"
            Image.new("RGBA", (40, 20), (255, 0, 0, 128)).save(source, format="PNG")
            result = normalize_dataset_image(root, "images/misleading.jpg")
            self.assertEqual(result.source_format, "PNG")
            self.assertEqual(result.mime_type, "image/jpeg")
            self.assertEqual((result.width, result.height), (40, 20))
            self.assertTrue(result.jpeg_bytes.startswith(bytes.fromhex("ffd8ff")))

    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            root.mkdir()
            outside = Path(tmp) / "outside.jpg"
            Image.new("RGB", (2, 2)).save(outside, format="JPEG")
            with self.assertRaises(ImageLoadError):
                resolve_image_path(root, "../outside.jpg")

    def test_rejects_oversized_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large.jpg"
            path.write_bytes(b"x" * 100)
            with self.assertRaises(ImageLoadError):
                normalize_image(path, max_bytes=10)

    def test_rejects_pixel_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.jpg"
            Image.new("RGB", (10, 10)).save(path, format="JPEG")
            with self.assertRaises(ImageLoadError):
                normalize_image(path, max_pixels=50)

if __name__ == "__main__":
    unittest.main()
