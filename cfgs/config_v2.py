import yaml


def add_cfg(cfg, yaml_file):
    # Read YAML file
    try:
        cfg.update(yaml.load(open(yaml_file, 'r')))
    except Exception:
        print('Error: cannot parse cfg', yaml_file)
        raise Exception


def load_cfg_yamls(yaml_files):
    cfg = dict()
    for yf in yaml_files:
        add_cfg(cfg, yf)
    return cfg
