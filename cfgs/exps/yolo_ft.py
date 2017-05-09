import numpy as np
import yaml

# Read YAML file
try:
    cfg_file = open('voc0712_trainval_ft_debug2.yaml', 'r')
    cfg = yaml.load(cfg_file)
    print(cfg)
except Exception:
    print('Error: cannot parse cfg')
    raise Exception
