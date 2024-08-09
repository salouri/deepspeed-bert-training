import os
import json

class NamedPipeManager:
    def __init__(self, pipe_path):
        self.pipe_path = pipe_path
        self.create_named_pipe()

    def create_named_pipe(self):
        if not os.path.exists(self.pipe_path):
            os.mkfifo(self.pipe_path)

    def write_to_pipe(self, data):
        with open(self.pipe_path, 'w') as pipe:
            json.dump(data, pipe)

    def read_from_pipe(self):
        with open(self.pipe_path, 'r') as pipe:
            data = json.load(pipe)
        return data

    def remove_named_pipe(self):
        try:
            os.remove(self.pipe_path)
        except OSError as oe:
            if oe.errno != errno.ENOENT:
                raise
