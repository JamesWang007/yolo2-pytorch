import yaml


def add_cfg(cfg, yaml_file):
    # Read YAML file
    try:
        cfg.update(yaml.load(open(yaml_file, 'r')))
    except Exception:
        print('Error: cannot parse cfg', yaml_file)
        raise Exception
