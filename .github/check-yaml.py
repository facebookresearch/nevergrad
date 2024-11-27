# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pathlib

import yaml

for path in pathlib.Path(".").glob("**/*.yml"):
    print(path)
    with open(path) as f:
        yaml.load(f, Loader=yaml.FullLoader)
