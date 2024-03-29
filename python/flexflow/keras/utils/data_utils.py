"""Utilities for file download and caching."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import multiprocessing as mp
import os
import random
import shutil
import sys
import tarfile
import threading
import time
import warnings
import zipfile
import uuid
import atexit
from abc import abstractmethod
from contextlib import closing
from multiprocessing.pool import ThreadPool

import numpy as np
import six
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlopen

try:
    import queue
except ImportError:
    import Queue as queue

from .generic_utils import Progbar

if sys.version_info[0] == 2:
    def urlretrieve(url, filename, reporthook=None, data=None):
        """Replacement for `urlretrieve` for Python 2.

        Under Python 2, `urlretrieve` relies on `FancyURLopener` from legacy
        `urllib` module, known to have issues with proxy management.

        # Arguments
            url: url to retrieve.
            filename: where to store the retrieved data locally.
            reporthook: a hook function that will be called once
                on establishment of the network connection and once
                after each block read thereafter.
                The hook will be passed three arguments;
                a count of blocks transferred so far,
                a block size in bytes, and the total size of the file.
            data: `data` argument passed to `urlopen`.
        """

        def chunk_read(response, chunk_size=8192, reporthook=None):
            content_type = response.info().get('Content-Length')
            total_size = -1
            if content_type is not None:
                total_size = int(content_type.strip())
            count = 0
            while True:
                chunk = response.read(chunk_size)
                count += 1
                if reporthook is not None:
                    reporthook(count, chunk_size, total_size)
                if chunk:
                    yield chunk
                else:
                    break

        with closing(urlopen(url, data)) as response, open(filename, 'wb') as fd:
            for chunk in chunk_read(response, reporthook=reporthook):
                fd.write(chunk)
else:
    from six.moves.urllib.request import urlretrieve


def _extract_archive(file_path, path='.', archive_format='auto'):
    """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.

    # Arguments
        file_path: path to the archive file
        path: path to extract the archive file
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.

    # Returns
        True if a match was found and an archive extraction was completed,
        False otherwise.
    """
    if archive_format is None:
        return False
    if archive_format == 'auto':
        archive_format = ['tar', 'zip']
    if isinstance(archive_format, six.string_types):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type == 'tar':
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type == 'zip':
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError,
                        KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False

def cleanup_keras_folder(fpath=os.path.join('/tmp', '.keras')):
    """Deletes Keras temporary folder used for downloading datasets
    """
    shutil.rmtree(fpath, ignore_errors=True)

def get_file(fname,
             origin,
             untar=False,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
    """Downloads a file from a URL if it not already in the cache.

    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `~/.keras/datasets/example.txt`.

    Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
    Passing a hash will verify the file after download. The command line
    programs `shasum` and `sha256sum` can compute the hash.

    # Arguments
        fname: Name of the file. If an absolute path `/path/to/file.txt` is
            specified the file will be saved at that location.
        origin: Original URL of the file.
        untar: Deprecated in favor of 'extract'.
            boolean, whether the file should be decompressed
        md5_hash: Deprecated in favor of 'file_hash'.
            md5 hash of the file for verification
        file_hash: The expected hash string of the file after download.
            The sha256 and md5 hash algorithms are both supported.
        cache_subdir: Subdirectory under the Keras cache dir where the file is
            saved. If an absolute path `/path/to/folder` is
            specified the file will be saved at that location.
        hash_algorithm: Select the hash algorithm to verify the file.
            options are 'md5', 'sha256', and 'auto'.
            The default 'auto' detects the hash algorithm in use.
        extract: True tries extracting the file as an Archive, like tar or zip.
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.
        cache_dir: Location to store cached files, when None it
            defaults to the [Keras Directory](/faq/#where-is-the-keras-configuration-filed-stored).

    # Returns
        Path to the downloaded file
    """  # noqa
    if cache_dir is None:
        if 'KERAS_HOME' in os.environ:
            cache_dir = os.environ.get('KERAS_HOME')
        else:
            cache_dir = os.path.join(os.path.expanduser('~'), '.keras')
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.keras', str(uuid.uuid4()))
        atexit.register(cleanup_keras_folder, datadir_base)
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print('A local file was found, but it seems to be incomplete'
                      ' or outdated because the {} file hash does not match '
                      'the original value of {} so we will re-download the '
                      'data.'.format(hash_algorithm, file_hash))
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)

        class ProgressTracker(object):
            # Maintain progbar for the lifetime of download.
            # This design was chosen for Python 2.7 compatibility.
            progbar = None

        def dl_progress(count, block_size, total_size):
            if ProgressTracker.progbar is None:
                if total_size == -1:
                    total_size = None
                ProgressTracker.progbar = Progbar(total_size)
            else:
                ProgressTracker.progbar.update(count * block_size)

        error_msg = 'URL fetch failure on {} : {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath, dl_progress)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        ProgressTracker.progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            _extract_archive(fpath, datadir, archive_format='tar')
        return untar_fpath

    if extract:
        _extract_archive(fpath, datadir, archive_format)

    return fpath


def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
    """Calculates a file sha256 or md5 hash.

    # Example

    ```python
        >>> from keras.utils.data_utils import _hash_file
        >>> _hash_file('/path/to/file.zip')
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ```

    # Arguments
        fpath: path to the file being validated
        algorithm: hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.

    # Returns
        The file hash
    """
    if (algorithm == 'sha256') or (algorithm == 'auto' and len(hash) == 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    """Validates a file against a sha256 or md5 hash.

    # Arguments
        fpath: path to the file being validated
        file_hash:  The expected hash string of the file.
            The sha256 and md5 hash algorithms are both supported.
        algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.

    # Returns
        Whether the file is valid
    """
    if ((algorithm == 'sha256') or
            (algorithm == 'auto' and len(file_hash) == 64)):
        hasher = 'sha256'
    else:
        hasher = 'md5'

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False


class Sequence(object):
    """Base object for fitting to a sequence of data, such as a dataset.

    Every `Sequence` must implement the `__getitem__` and the `__len__` methods.
    If you want to modify your dataset between epochs you may implement
    `on_epoch_end`. The method `__getitem__` should return a complete batch.

    # Notes

    `Sequence` are a safer way to do multiprocessing. This structure guarantees
    that the network will only train once on each sample per epoch which is not
    the case with generators.

    # Examples

    ```python
        from skimage.io import imread
        from skimage.transform import resize
        import numpy as np

        # Here, `x_set` is list of path to the images
        # and `y_set` are the associated classes.

        class CIFAR10Sequence(Sequence):

            def __init__(self, x_set, y_set, batch_size):
                self.x, self.y = x_set, y_set
                self.batch_size = batch_size

            def __len__(self):
                return int(np.ceil(len(self.x) / float(self.batch_size)))

            def __getitem__(self, idx):
                batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

                return np.array([
                    resize(imread(file_name), (200, 200))
                       for file_name in batch_x]), np.array(batch_y)
    ```
    """

    use_sequence_api = True

    @abstractmethod
    def __getitem__(self, index):
        """Gets batch at position `index`.

        # Arguments
            index: position of the batch in the Sequence.

        # Returns
            A batch
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Number of batch in the Sequence.

        # Returns
            The number of batches in the Sequence.
        """
        raise NotImplementedError

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        pass

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item


# Global variables to be shared across processes
_SHARED_SEQUENCES = {}
# We use a Value to provide unique id to different processes.
_SEQUENCE_COUNTER = None


def init_pool(seqs):
    global _SHARED_SEQUENCES
    _SHARED_SEQUENCES = seqs


def get_index(uid, i):
    """Get the value from the Sequence `uid` at index `i`.

    To allow multiple Sequences to be used at the same time, we use `uid` to
    get a specific one. A single Sequence would cause the validation to
    overwrite the training Sequence.

    # Arguments
        uid: int, Sequence identifier
        i: index

    # Returns
        The value at index `i`.
    """
    return _SHARED_SEQUENCES[uid][i]


class SequenceEnqueuer(object):
    """Base class to enqueue inputs.

    The task of an Enqueuer is to use parallelism to speed up preprocessing.
    This is done with processes or threads.

    # Examples

    ```python
        enqueuer = SequenceEnqueuer(...)
        enqueuer.start()
        datas = enqueuer.get()
        for data in datas:
            # Use the inputs; training, evaluating, predicting.
            # ... stop sometime.
        enqueuer.close()
    ```

    The `enqueuer.get()` should be an infinite stream of datas.

    """
    def __init__(self, sequence,
                 use_multiprocessing=False):
        self.sequence = sequence
        self.use_multiprocessing = use_multiprocessing

        global _SEQUENCE_COUNTER
        if _SEQUENCE_COUNTER is None:
            try:
                _SEQUENCE_COUNTER = mp.Value('i', 0)
            except OSError:
                # In this case the OS does not allow us to use
                # multiprocessing. We resort to an int
                # for enqueuer indexing.
                _SEQUENCE_COUNTER = 0

        if isinstance(_SEQUENCE_COUNTER, int):
            self.uid = _SEQUENCE_COUNTER
            _SEQUENCE_COUNTER += 1
        else:
            # Doing Multiprocessing.Value += x is not process-safe.
            with _SEQUENCE_COUNTER.get_lock():
                self.uid = _SEQUENCE_COUNTER.value
                _SEQUENCE_COUNTER.value += 1

        self.workers = 0
        self.executor_fn = None
        self.queue = None
        self.run_thread = None
        self.stop_signal = None

    def is_running(self):
        return self.stop_signal is not None and not self.stop_signal.is_set()

    def start(self, workers=1, max_queue_size=10):
        """Start the handler's workers.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, workers could block on `put()`)
        """
        if self.use_multiprocessing:
            self.executor_fn = self._get_executor_init(workers)
        else:
            # We do not need the init since it's threads.
            self.executor_fn = lambda _: ThreadPool(workers)
        self.workers = workers
        self.queue = queue.Queue(max_queue_size)
        self.stop_signal = threading.Event()
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.daemon = True
        self.run_thread.start()

    def _send_sequence(self):
        """Send current Iterable to all workers."""
        # For new processes that may spawn
        _SHARED_SEQUENCES[self.uid] = self.sequence

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.

        # Arguments
            timeout: maximum time to wait on `thread.join()`
        """
        self.stop_signal.set()
        with self.queue.mutex:
            self.queue.queue.clear()
            self.queue.unfinished_tasks = 0
            self.queue.not_full.notify()
        self.run_thread.join(timeout)
        _SHARED_SEQUENCES[self.uid] = None

    @abstractmethod
    def _run(self):
        """Submits request to the executor and queue the `Future` objects."""
        raise NotImplementedError

    @abstractmethod
    def _get_executor_init(self, workers):
        """Get the Pool initializer for multiprocessing.

        # Returns
            Function, a Function to initialize the pool
        """
        raise NotImplementedError

    @abstractmethod
    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Returns
            Generator yielding tuples `(inputs, targets)`
                or `(inputs, targets, sample_weights)`.
        """
        raise NotImplementedError


class OrderedEnqueuer(SequenceEnqueuer):
    """Builds a Enqueuer from a Sequence.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        sequence: A `keras.utils.data_utils.Sequence` object.
        use_multiprocessing: use multiprocessing if True, otherwise threading
        shuffle: whether to shuffle the data at the beginning of each epoch
    """
    def __init__(self, sequence, use_multiprocessing=False, shuffle=False):
        super(OrderedEnqueuer, self).__init__(sequence, use_multiprocessing)
        self.shuffle = shuffle
        self.end_of_epoch_signal = threading.Event()

    def _get_executor_init(self, workers):
        """Get the Pool initializer for multiprocessing.

        # Returns
            Function, a Function to initialize the pool
        """
        return lambda seqs: mp.Pool(workers,
                                    initializer=init_pool,
                                    initargs=(seqs,))

    def _wait_queue(self):
        """Wait for the queue to be empty."""
        while True:
            time.sleep(0.1)
            if self.queue.unfinished_tasks == 0 or self.stop_signal.is_set():
                return

    def _run(self):
        """Submits request to the executor and queue the `Future` objects."""
        while True:
            sequence = list(range(len(self.sequence)))
            self._send_sequence()  # Share the initial sequence

            if self.shuffle:
                random.shuffle(sequence)

            with closing(self.executor_fn(_SHARED_SEQUENCES)) as executor:
                for i in sequence:
                    if self.stop_signal.is_set():
                        return
                    future = executor.apply_async(get_index, (self.uid, i))
                    future.idx = i
                    self.queue.put(future, block=True)

                # Done with the current epoch, waiting for the final batches
                self._wait_queue()

                if self.stop_signal.is_set():
                    # We're done
                    return

            # Call the internal on epoch end.
            self.sequence.on_epoch_end()
            # communicate on_epoch_end to the main thread
            self.end_of_epoch_signal.set()

    def join_end_of_epoch(self):
        self.end_of_epoch_signal.wait(timeout=30)
        self.end_of_epoch_signal.clear()

    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Yields
            The next element in the queue, i.e. a tuple
            `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
        """
        try:
            while self.is_running():
                try:
                    future = self.queue.get(block=True)
                    inputs = future.get(timeout=30)
                except mp.TimeoutError:
                    idx = future.idx
                    warnings.warn(
                        'The input {} could not be retrieved.'
                        ' It could be because a worker has died.'.format(idx),
                        UserWarning)
                    inputs = self.sequence[idx]
                finally:
                    self.queue.task_done()

                if inputs is not None:
                    yield inputs
        except Exception:
            self.stop()
            six.reraise(*sys.exc_info())


def init_pool_generator(gens, random_seed=None):
    global _SHARED_SEQUENCES
    _SHARED_SEQUENCES = gens

    if random_seed is not None:
        ident = mp.current_process().ident
        np.random.seed(random_seed + ident)


def next_sample(uid):
    """Get the next value from the generator `uid`.

    To allow multiple generators to be used at the same time, we use `uid` to
    get a specific one. A single generator would cause the validation to
    overwrite the training generator.

    # Arguments
        uid: int, generator identifier

    # Returns
        The next value of generator `uid`.
    """
    return six.next(_SHARED_SEQUENCES[uid])


class GeneratorEnqueuer(SequenceEnqueuer):
    """Builds a queue out of a data generator.

    The provided generator can be finite in which case the class will throw
    a `StopIteration` exception.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        sequence: a sequence function which yields data
        use_multiprocessing: use multiprocessing if True, otherwise threading
        wait_time: time to sleep in-between calls to `put()`
        random_seed: Initial seed for workers,
            will be incremented by one for each worker.
    """

    def __init__(self, sequence, use_multiprocessing=False, wait_time=None,
                 random_seed=None):
        super(GeneratorEnqueuer, self).__init__(sequence, use_multiprocessing)
        self.random_seed = random_seed
        if wait_time is not None:
            warnings.warn('`wait_time` is not used anymore.',
                          DeprecationWarning)

    def _get_executor_init(self, workers):
        """Get the Pool initializer for multiprocessing.

        # Returns
            Function, a Function to initialize the pool
        """
        return lambda seqs: mp.Pool(workers,
                                    initializer=init_pool_generator,
                                    initargs=(seqs, self.random_seed))

    def _run(self):
        """Submits request to the executor and queue the `Future` objects."""
        self._send_sequence()  # Share the initial generator
        with closing(self.executor_fn(_SHARED_SEQUENCES)) as executor:
            while True:
                if self.stop_signal.is_set():
                    return
                self.queue.put(
                    executor.apply_async(next_sample, (self.uid,)), block=True)

    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Yields
            The next element in the queue, i.e. a tuple
            `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
        """
        try:
            while self.is_running():
                try:
                    future = self.queue.get(block=True)
                    inputs = future.get(timeout=30)
                    self.queue.task_done()
                except mp.TimeoutError:
                    warnings.warn(
                        'An input could not be retrieved.'
                        ' It could be because a worker has died.'
                        'We do not have any information on the lost sample.',
                        UserWarning)
                    continue
                if inputs is not None:
                    yield inputs
        except StopIteration:
            # Special case for finite generators
            last_ones = []
            while self.queue.qsize() > 0:
                last_ones.append(self.queue.get(block=True))
            # Wait for them to complete
            [f.wait() for f in last_ones]
            # Keep the good ones
            last_ones = (future.get() for future in last_ones if future.successful())
            for inputs in last_ones:
                if inputs is not None:
                    yield inputs
        except Exception as e:
            self.stop()
            if 'generator already executing' in str(e):
                raise RuntimeError(
                    "Your generator is NOT thread-safe."
                    "Keras requires a thread-safe generator when"
                    "`use_multiprocessing=False, workers > 1`."
                    "For more information see issue #1638.")
            six.reraise(*sys.exc_info())