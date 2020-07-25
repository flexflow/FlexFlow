from __future__ import absolute_import
from __future__ import print_function

import tarfile
import os
import sys
import shutil
import hashlib
from six.moves.urllib.request import urlopen
from six.moves.urllib.error import URLError, HTTPError

import requests
from generic_utils import Progbar


# Under Python 2, 'urlretrieve' relies on FancyURLopener from legacy
# urllib module, known to have issues with proxy management
if sys.version_info[0] == 2:
    def urlretrieve(url, filename, reporthook=None, data=None):
        def chunk_read(response, chunk_size=8192, reporthook=None):
            total_size = response.info().get('Content-Length').strip()
            total_size = int(total_size)
            count = 0
            while 1:
                chunk = response.read(chunk_size)
                count += 1
                if not chunk:
                    reporthook(count, total_size, total_size)
                    break
                if reporthook:
                    reporthook(count, chunk_size, total_size)
                yield chunk

        response = urlopen(url, data)
        with open(filename, 'wb') as fd:
            for chunk in chunk_read(response, reporthook=reporthook):
                fd.write(chunk)
else:
    from six.moves.urllib.request import urlretrieve


def get_file(fname, origin, untar=False,
             #md5_hash=None, datadir='../Data/common'):
             #md5_hash=None, cache_subdir='common', datadir='../Data/common'):
             md5_hash=None, cache_subdir='common', datadir=None): # datadir argument was never actually used so changing it to None
    """ Downloads a file from a URL if it not already in the cache.
        Passing the MD5 hash will verify the file after download as well
        as if it is already present in the cache.

        Parameters
        ----------
        fname : string
            name of the file
        origin : string
            original URL of the file
        untar : boolean
            whether the file should be decompressed
        md5_hash : string
            MD5 hash of the file for verification
        cache_subdir : string
            directory being used as the cache
        datadir : string
            if set, datadir becomes its setting (which could be e.g. an absolute path) and cache_subdir no longer matters

        Returns
        ----------
        Path to the downloaded file
    """

    if datadir is None:
        file_path = os.path.dirname(os.path.realpath(__file__))
        datadir_base = os.path.expanduser(os.path.join(file_path, '..', 'Data'))
        datadir = os.path.join(datadir_base, cache_subdir)

    if not os.path.exists(datadir):
        os.makedirs(datadir)

    #if untar:
    #    fnamesplit = fname.split('.tar.gz')
    #    untar_fpath = os.path.join(datadir, fnamesplit[0])

    if fname.endswith('.tar.gz'):
        fnamesplit = fname.split('.tar.gz')
        untar_fpath = os.path.join(datadir, fnamesplit[0])
        untar = True
    elif fname.endswith('.tgz'):
        fnamesplit = fname.split('.tgz')
        untar_fpath = os.path.join(datadir, fnamesplit[0])
        untar = True
    else:
        untar_fpath = None

    fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath) or (untar_fpath is not None and os.path.exists(untar_fpath)):
        # file found; verify integrity if a hash was provided
        if md5_hash is not None:
            if not validate_file(fpath, md5_hash):
                print('A local file was found, but it seems to be '
                      'incomplete or outdated.')
                download = True
    else:
        download = True

    # fix ftp protocol if needed
    '''
    if origin.startswith('ftp://'):
        new_url = origin.replace('ftp://','http://')
        origin = new_url
    print('Origin = ', origin)
    '''

    if download:
        print('Downloading data from', origin)
        global progbar
        progbar = None

        def dl_progress(count, block_size, total_size):
            global progbar
            if progbar is None:
                progbar = Progbar(total_size)
            else:
                progbar.update(count * block_size)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath, dl_progress)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        progbar = None
        print()

    if untar:
        if not os.path.exists(untar_fpath):
            print('Untarring file...')
            tfile = tarfile.open(fpath, 'r:gz')
            try:
                tfile.extractall(path=datadir)
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(untar_fpath):
                    if os.path.isfile(untar_fpath):
                        os.remove(untar_fpath)
                    else:
                        shutil.rmtree(untar_fpath)
                raise
            tfile.close()
        return untar_fpath
        print()

    return fpath


def validate_file(fpath, md5_hash):
    """ Validates a file against a MD5 hash

        Parameters
        ----------
        fpath : string
            path to the file being validated
        md5_hash : string
            the MD5 hash being validated against

        Returns
        ----------
        boolean
            Whether the file is valid
    """
    hasher = hashlib.md5()
    with open(fpath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    if str(hasher.hexdigest()) == str(md5_hash):
        return True
    else:
        return False
