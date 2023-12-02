
class Entity(object):
    def __init__(self, _id, _text, _mask, _interactive, _type, _start_idx, _end_idx, _image=None):
        self.id = _id
        self.text = _text
        self.mask = _mask
        self.interactive = _interactive
        self.type = _type
        self.start_idx = _start_idx
        self.end_idx = _end_idx

        self.image = _image