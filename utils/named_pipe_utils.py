import os
import json

class NamedPipeManager:
    def __init__(self, pipe_path):
        self.pipe_path = pipe_path
        self.create_named_pipe()

    def create_named_pipe(self):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.pipe_path), exist_ok=True)
        # Create the named pipe if it doesn't exist
        if not os.path.exists(self.pipe_path):
            os.mkfifo(self.pipe_path)

    def write_to_pipe(self, data):
         rint(f"Reading from pipe: {self.pipe_path}")
        try:
            with open(self.pipe_path, 'w') as pipe:
                while True:
                    # Use select.select to check if the pipe is writable
                    _, wlist, _ = select.select([], [pipe], [], 1)
                    if wlist:
                        # If writable, write the data to the pipe
                        json.dump(data, pipe)
                        pipe.flush()
                        break
        except Exception as e:
            print(f"Error writing to pipe: {e}")

    def read_from_pipe(self):
        print(f"Reading from pipe: {self.pipe_path}")
        with open(self.pipe_path, 'r') as pipe:  # Open in read mode
            while True:
                rlist, _, _ = select.select([pipe], [], [], 1)  # Wait for data
                if rlist:
                    data = pipe.read()
                    if data:
                        return data
                    else:
                        break
                else:
                    print("No data available yet...")
    def remove_named_pipe(self):
        try:
            os.remove(self.pipe_path)
        except OSError as oe:
            if oe.errno != errno.ENOENT:
                raise
