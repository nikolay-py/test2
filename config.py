"""Flask application initialization."""
import os

from dotenv import load_dotenv

load_dotenv()


class Config(object):
    """Base initialization config."""

    DEBUG = True
    PROPAGATE_EXCEPTIONS = True

    BASE_DIR = os.getcwd()

    WEIGHTS_PATH = str(os.path.join(BASE_DIR, 'artifacts'))
    NSFW = os.environ.get('NSFW')
    DEVICE = os.environ.get('DEVICE')


