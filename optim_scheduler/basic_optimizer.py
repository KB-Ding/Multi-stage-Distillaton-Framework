
class basic_optimizer(object):
    def __init__(self, *args, **kwargs):
        pass

    def step(self):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        raise NotImplementedError

    def lr_scheduler(self, config, *args, **kwargs):
        raise NotImplementedError
