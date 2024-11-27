import pathlib
import yaml

for path in pathlib.Path(".").glob("**/*.yml"):
    print(path)
    with open(path) as f:
        yaml.load(f, Loader=yaml.FullLoader)
