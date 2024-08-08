import os
import errno
import json

class NamedPipeManager:
    def __init__(self, pipe_name):
        self.pipe_name = pipe_name
        self.create_named_pipe()

    def create_named_pipe(self):
        try:
            os.mkfifo(self.pipe_name)
        except OSError as oe:
            if oe.errno != errno.EEXIST:
                raise

    def write_data_to_pipe(self, data):
        with open(self.pipe_name, 'w') as pipe:
            json.dump(data, pipe)

    def read_data_from_pipe(self):
        with open(self.pipe_name, 'r') as pipe:
            data = json.load(pipe)
        return data

    def remove_named_pipe(self):
        try:
            os.remove(self.pipe_name)
        except OSError as oe:
            if oe.errno != errno.ENOENT:
                raise

# Example usage
# pipe_manager = NamedPipeManager('/tmp/my_pipe')
# pipe_manager.write_data_to_pipe({'key': 'value'})
# data = pipe_manager.read_data_from_pipe()
# print(data)
# pipe_manager.remove_named_pipe()
