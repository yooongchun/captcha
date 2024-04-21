import io
import glob
from pathlib import Path

from tqdm import tqdm
from loguru import logger

from src.helper.util import ImageUtil


def convert(src: Path, dst: Path):
    assert src.is_dir(), "src should be a directory"
    if not dst.exists():
        dst.mkdir(parents=True, exist_ok=True)

    for file in tqdm(glob.glob("**/*.png", root_dir=str(src), recursive=True)):
        file = Path(src, file)
        try:
            img_util = ImageUtil(file)
        except Exception as e:
            logger.warning(f"error on {file}: {e}")
            file.unlink()
            continue
        for c in ["red", "blue", "black", "yellow"]:
            img = img_util.get_channel(c)
            # Save the image to a temporary in-memory file
            temp_file = io.BytesIO()
            img.save(temp_file, format="PNG")

            # Check the size of the temporary file
            if temp_file.getbuffer().nbytes > 300:
                # If the size is greater than 300 bytes, save the image to the destination file
                img.save(dst / f"{file.stem}_{c}.png")
