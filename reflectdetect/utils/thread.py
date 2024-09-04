import queue
import threading
from typing import Callable, Any, Optional


def run_in_thread(
    func: Callable[..., Any], throw_exceptions: bool = True, *args: Any, **kwargs: Any
) -> Optional[Any]:
    """
    Runs a function in a separate thread, suppresses its output, and ignores any exceptions.

    :param func: The function to run in a separate thread.
    :param throw_exceptions:  whether to throw an exception that might occur in the thread
    :param args: Positional arguments to pass to the function.
    :param kwargs: Keyword arguments to pass to the function.
    :return: The result of the function, or None if an exception occurred.
    """

    def wrapper(q: queue.Queue[Optional[Any]], *args: Any, **kwargs: Any) -> None:
        try:
            result = func(*args, **kwargs)
            q.put(result)
        except Exception as e:
            # Ignore the exception and put None in the queue
            if throw_exceptions:
                raise e
            q.put(None)

    # Create a queue to store the result
    q: queue.Queue[Optional[Any]] = queue.Queue()

    # Create and start the thread
    thread = threading.Thread(target=wrapper, args=(q, *args), kwargs=kwargs)
    thread.start()

    # Wait for the thread to finish
    thread.join()

    # Retrieve and return the result from the queue
    return q.get()
