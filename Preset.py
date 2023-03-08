class Presets:
    def __init__(self, *args):
        self.presets = []
        for arg in args:
            self.presets.append(arg)
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.presets[key]
        elif isinstance(key, str):
            for preset in self.presets:
                if preset.name == key:
                    return preset
            raise KeyError(f'Preset with name {key} not found')
        else:
            raise TypeError(f'Key must be int or str, not {type(key)}')
    
    def __iter__(self):
        return iter(self.presets)
    
    def __len__(self):
        return len(self.presets)

class Preset:
    def __init__(self, name, size, freq):
        self.name = name
        self.size = size
        self.freq = freq

preset1 = Preset('verysmall', (62,-1), 1)
preset2 = Preset('small', (86,-1), 2)
preset4 = Preset('medium', (117,-1), 4)
preset8 = Preset('large', (152,-1), 8)