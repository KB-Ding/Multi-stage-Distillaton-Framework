
class basic_formatter:
    def __init__(self, config, model, *args, **params):
        self.config = config
        self.model = model

    def process(self, data, config, mode, *args, **params):
        return data

