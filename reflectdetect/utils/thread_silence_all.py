import threading
import queue
import os
import sys

# Custom context manager to suppress stdout and stderr
class SuppressOutput:
    def __enter__(self):
        # Save the original file descriptors
        self.stdout_fd = sys.stdout.fileno()
        self.stderr_fd = sys.stderr.fileno()

        # Save a copy of the original file descriptors
        self.saved_stdout = os.dup(self.stdout_fd)
        self.saved_stderr = os.dup(self.stderr_fd)

        # Open null device
        self.null_fd = os.open(os.devnull, os.O_RDWR)

        # Redirect stdout and stderr to the null device
        os.dup2(self.null_fd, self.stdout_fd)
        os.dup2(self.null_fd, self.stderr_fd)

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore stdout and stderr
        os.dup2(self.saved_stdout, self.stdout_fd)
        os.dup2(self.saved_stderr, self.stderr_fd)

        # Close the copies and the null device
        #os.close(self.saved_stdout)
        #os.close(self.saved_stderr)
        os.close(self.null_fd)

# Suppress any output from threading and unraisable exceptions
def silent_excepthook(args):
    pass  # Do nothing, completely silence the exceptions

def silent_unraisablehook(unraisable):
    pass  # Do nothing, completely silence the unraisable exceptions

# Set the custom hooks
threading.excepthook = silent_excepthook
#sys.unraisablehook = silent_unraisablehook

def run_in_thread(func, *args, **kwargs):
    """
    Runs a function in a separate thread, suppresses its output, and ignores any exceptions.

    :param func: The function to run in a separate thread.
    :param args: Positional arguments to pass to the function.
    :param kwargs: Keyword arguments to pass to the function.
    :return: The result of the function, or None if an exception occurred.
    """
    def wrapper(q, *args, **kwargs):
        with SuppressOutput():
            try:
                result = func(*args, **kwargs)
                q.put(result)
            except Exception:
                # Ignore the exception and put None in the queue
                q.put(None)

    # Create a queue to store the result
    q = queue.Queue()

    # Create and start the thread
    thread = threading.Thread(target=wrapper, args=(q, *args), kwargs=kwargs)
    thread.start()

    # Wait for the thread to finish
    thread.join()

    # Retrieve and return the result from the queue
    return q.get()
