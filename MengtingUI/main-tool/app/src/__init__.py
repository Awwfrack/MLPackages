from flask_caching import Cache
import yaml
import os

conf_path = os.path.join(os.path.dirname((os.path.dirname(__file__))), "conf.yaml")

with open(conf_path, "r") as file:
    config = yaml.safe_load(file)
    
CACHE_DIR = config["cache_directory"]

cache = Cache(config={
    'CACHE_TYPE': 'FileSystemCache',
    'CACHE_DIR': CACHE_DIR,
    "CACHE_THRESHOLD": 3000,
    })