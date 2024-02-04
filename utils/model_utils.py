from urllib.parse import urlparse
from PIL import Image
from io import BytesIO
import requests

def open_image(path_or_url):
    # Check if the given path is a URL
    parsed_url = urlparse(path_or_url)
    if parsed_url.scheme and parsed_url.netloc:
        # It's a URL, so use requests to get the image content
        response = requests.get(path_or_url)
        if response.status_code == 200:
            # Open the image using PIL
            image = Image.open(BytesIO(response.content))
            return image
        else:
            return None
    else:
        # It's a local path, so directly open the image using PIL
        try:
            image = Image.open(path_or_url)
            return image
        except Exception as e:
            return None
