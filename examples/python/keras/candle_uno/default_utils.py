from __future__ import absolute_import

import numpy as np
import random
from pprint import pprint
import inspect

import logging
import warnings

import os
import sys
import gzip
import argparse
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path)

work_path = os.path.dirname(os.path.realpath(__file__))

from file_utils import get_file

# Seed for random generation -- default value
DEFAULT_SEED = 7102
DEFAULT_TIMEOUT = -1 # no timeout
DEFAULT_DATATYPE = np.float32


PARAMETERS_CANDLE = [
                  'config_file', 
                  # neon parser
                  'verbose', 'logfile', 'save_path', 'model_name', 'data_type', 'dense', 'rng_seed', 'epochs', 'batch_size', 
                  # general behavior
                  'train_bool', 'eval_bool', 'timeout', 
                  # logging
                  'home_dir', 'train_data', 'test_data', 'output_dir', 'data_url', 'experiment_id', 'run_id', 
                  # model architecture
                  'conv', 'locally_connected', 'activation', 'out_activation', 'lstm_size', 'recurrent_dropout', 
                  # processing between layers
                  'dropout', 'pool', 'batch_normalization', 
                  # model evaluation
                  'loss', 'optimizer', 'metrics', 
                  # data preprocessing
                  'scaling', 'shuffle', 'feature_subsample', 
                  # training
                  'learning_rate', 'early_stop', 'momentum', 'initialization', 
                  'val_split', 'train_steps', 'val_steps', 'test_steps', 'train_samples', 'val_samples', 
                  # backend
                  'gpus', 
                  # profiling
                  'profiling',
                  # cyclic learning rate
                  'clr_flag', 'clr_mode', 'clr_base_lr', 'clr_max_lr', 'clr_gamma'
                  ]

CONFLICT_LIST = [
    ['clr_flag','warmup_lr'],
    ['clr_flag','reduce_lr']
]

def check_flag_conflicts(params):
    key_set = set(params.keys())
    # check for conflicts
    #conflict_flag = False
    # loop over each set of mutually exclusive flags
    # if any set conflicts exit program 
    for flag_list in CONFLICT_LIST:
        flag_count = 0
        for i in flag_list:
            if i in key_set:
                if params[i] is True:
                    flag_count +=1
        if flag_count > 1 :
            raise Exception('ERROR ! Conflict in flag specification. ' \
                        'These flags should not be used together: ' + str(sorted(flag_list)) + \
                            '... Exiting')
            #print("Warning: conflicting flags in ", flag_list)
            #exit()

#### IO UTILS

def fetch_file(link, subdir, untar=False, md5_hash=None):
    """ Convert URL to file path and download the file
        if it is not already present in spedified cache.

        Parameters
        ----------
        link : link path
            URL of the file to download
        subdir : directory path
            Local path to check for cached file.
        untar : boolean
            Flag to specify if the file to download should
            be decompressed too.
            (default: False, no decompression)
        md5_hash : MD5 hash
            Hash used as a checksum to verify data integrity.
            Verification is carried out if a hash is provided.
            (default: None, no verification)

        Return
        ----------
        local path to the downloaded, or cached, file.
    """

    fname = os.path.basename(link)
    return get_file(fname, origin=link, untar=untar, md5_hash=md5_hash, cache_subdir=subdir)

def verify_path(path):
    """ Verify if a directory path exists locally. If the path
        does not exist, but is a valid path, it recursivelly creates
        the specified directory path structure.

        Parameters
        ----------
        path : directory path
            Description of local directory path
    """
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


def set_up_logger(logfile, logger, verbose):
    """ Set up the event logging system. Two handlers are created.
        One to send log records to a specified file and
        one to send log records to the (defaulf) sys.stderr stream.
        The logger and the file handler are set to DEBUG logging level.
        The stream handler is set to INFO logging level, or to DEBUG
        logging level if the verbose flag is specified.
        Logging messages which are less severe than the level set will
        be ignored.

        Parameters
        ----------
        logfile : filename
            File to store the log records
        logger : logger object
            Python object for the logging interface
        verbose : boolean
            Flag to increase the logging level from INFO to DEBUG. It 
            only applies to the stream handler.
    """
    verify_path(logfile)
    fh = logging.FileHandler(logfile)
    fh.setFormatter(logging.Formatter("[%(asctime)s %(process)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(''))
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(sh)


#### REFORMATING UTILS


def eval_string_as_list(str_read, separator, dtype):
    """ Parse a string and convert it into a list of lists.

        Parameters
        ----------
        str_read : string
            String read (from configuration file or command line, for example)
        separator : character
            Character that specifies the separation between the lists
        dtype : data type
            Data type to decode the elements of the list

        Return
        ----------
        decoded_list : list
            List extracted from string and with elements of the
            specified type.
    """

    # Figure out desired type
    ldtype = dtype
    if ldtype is None:
        ldtype = np.int32

    # Split list
    decoded_list = []
    out_list = str_read.split(separator)

    # Convert to desired type
    for el in out_list:
        decoded_list.append( ldtype( el ) )

    return decoded_list



def eval_string_as_list_of_lists(str_read, separator_out, separator_in, dtype):
    """ Parse a string and convert it into a list of lists.

        Parameters
        ----------
        str_read : string
            String read (from configuration file or command line, for example)
        separator_out : character
            Character that specifies the separation between the outer level lists
        separator_in : character
            Character that specifies the separation between the inner level lists
        dtype : data type
            Data type to decode the elements of the lists

        Return
        ----------
        decoded_list : list
            List of lists extracted from string and with elements of the specified type.
    """

    # Figure out desired type
    ldtype = dtype
    if ldtype is None:
        ldtype = np.int32

    # Split outer list
    decoded_list = []
    out_list = str_read.split(separator_out)
    # Split each internal list
    for l in out_list:
        in_list = []
        elem = l.split(separator_in)
        # Convert to desired type
        for el in elem:
            in_list.append( ldtype( el ) )
        decoded_list.append( in_list )

    return decoded_list


def str2bool(v):
    """This is taken from:
        https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
        Because type=bool is not interpreted as a bool and action='store_true' cannot be
        undone.

        Parameters
        ----------
        v : string
            String to interpret

        Return
        ----------
        Boolean value. It raises and exception if the provided string cannot \
        be interpreted as a boolean type.
        Strings recognized as boolean True : 
            'yes', 'true', 't', 'y', '1' and uppercase versions (where applicable).
        Strings recognized as boolean False : 
            'no', 'false', 'f', 'n', '0' and uppercase versions (where applicable).
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#### CLASS DEFINITIONS

class ArgumentStruct:
    """Class that converts a python dictionary into an object with
       named entries given by the dictionary keys.
       This structure simplifies the calling convention for accessing
       the dictionary values (corresponding to problem parameters).
       After the object instantiation both modes of access (dictionary
       or object entries) can be used.
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)



class ListOfListsAction(argparse.Action):
    """This class extends the argparse.Action class by instantiating
        an argparser that constructs a list-of-lists from an input
        (command-line option or argument) given as a string.
    """
    def __init__(self, option_strings, dest, type, **kwargs):
        """Initialize a ListOfListsAction object. If no type is specified,
           an integer is assumed by default as the type for the elements
           of the list-of-lists.

           Parameters
           ----------
           option_strings : string
               String to parse
           dest : object
               Object to store the output (in this case the parsed list-of-lists).
           type : data type
               Data type to decode the elements of the lists.
               Defaults to np.int32.
           kwargs : object
               Python object containing other argparse.Action parameters.

        """

        super(ListOfListsAction, self).__init__(option_strings, dest, **kwargs)
        self.dtype = type
        if self.dtype is None:
            self.dtype = np.int32



    def __call__(self, parser, namespace, values, option_string=None):
        """This function overrides the __call__ method of the base
           argparse.Action class.

           This function implements the action of the ListOfListAction
           class by parsing an input string (command-line option or argument)
           and maping it into a list-of-lists. The resulting list-of-lists is
           added to the namespace of parsed arguments. The parsing assumes that
           the separator between lists is a colon ':' and the separator inside
           the list is a comma ','. The values of the list are casted to the
           type specified at the object initialization.

           Parameters
           ----------
           parser : ArgumentParser object
               Object that contains this action
           namespace : Namespace object
               Namespace object that will be returned by the parse_args()
               function.
           values : string
               The associated command-line arguments converted to string type
               (i.e. input).
           option_string : string
               The option string that was used to invoke this action. (optional)

        """

        decoded_list = []
        removed1 = values.replace('[', '')
        removed2 = removed1.replace(']', '')
        out_list = removed2.split(':')

        for l in out_list:
            in_list = []
            elem = l.split(',')
            for el in elem:
                in_list.append( self.dtype( el ) )
            decoded_list.append( in_list )

        setattr(namespace, self.dest, decoded_list)

#### INITIALIZATION UTILS


def set_seed(seed):
    """Set the seed of the pseudo-random generator to the specified value.

        Parameters
        ----------
        seed : int
            Value to intialize or re-seed the generator.
    """
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)

    random.seed(seed)

def check_file_parameters_exists(params_parser, params_benchmark, params_file):
    """Functionality to verify that the parameters defined in the configuraion file are recognizable by the command line parser (i.e. no uknown keywords are used in the configuration file).
 
    Parameters
    ----------
    params_parser : python dictionary
        Includes parameters set via the command line.
    params_benchmark : python list
        Includes additional parameters defined in the benchmark.
    params_file : python dictionary
        Includes parameters read from the configuration file.

        Global:
        PARAMETERS_CANDLE : python list
            Includes all the core keywords that are specified in CANDLE.
    """
    # Get keywords from arguments coming via command line (and CANDLE supervisor)
    args_dict = vars(params_parser)
    args_set = set(args_dict.keys())
    # Get keywords from benchmark definition
    bmk_keys = []
    for item in params_benchmark:
        bmk_keys.append( item['name'] )
    bmk_set = set(bmk_keys)
    # Get core CANDLE keywords
    candle_set = set(PARAMETERS_CANDLE)
    # Consolidate keywords from CANDLE core, command line, CANDLE supervisor and benchmark
    candle_set = candle_set.union(args_set)
    candle_set = candle_set.union(bmk_set)
    # Get keywords used in config_file
    file_set = set(params_file.keys())
    # Compute keywords that come from the config_file that are not in the CANDLE specs
    diff_set = file_set.difference(candle_set)

    if ( len(diff_set) > 0 ):
        message = 'These keywords used in the configuration file are not defined in CANDLE: ' + str(sorted(diff_set))
        warnings.warn(message, RuntimeWarning)


def finalize_parameters(bmk):
    """Utility to parse parameters in common as well as parameters
        particular to each benchmark.

        Parameters
        ----------
        bmk : benchmark object
            Object that has benchmark filepaths and specifications

        Return
        ----------
        gParameters : python dictionary
            Dictionary with all the parameters necessary to run the benchmark.
            Command line overwrites config file specifications
    """

    # Parse common parameters
    bmk.parse_from_common()
    # Parse parameters that are applicable just to benchmark
    bmk.parse_from_benchmark()

    #print('Args:', args)
    # Get parameters from configuration file
    # Reads parameter subset, just checking if a config_file has been set
    # by comand line (the parse_known_args() function allows a partial
    # parsing)
    aux = bmk.parser.parse_known_args()
    try : # Try to get the 'config_file' option
        conffile_txt = aux[0].config_file
    except AttributeError: # The 'config_file' option was not set by command-line
        conffile = bmk.conffile # use default file
    else: # a 'config_file' has been set --> use this file
        conffile = os.path.join(bmk.file_path, conffile_txt)

    #print("Configuration file: ", conffile)
    fileParameters = bmk.read_config_file(conffile)#aux.config_file)#args.config_file)
    # Get command-line parameters
    args, unkown = bmk.parser.parse_known_args()
    print(unkown)
    #print ('Params:', fileParameters)
    # Check keywords from file against CANDLE common and module definitions
    bmk_dict = bmk.additional_definitions
    check_file_parameters_exists(args, bmk_dict, fileParameters)
    # Consolidate parameter set. Command-line parameters overwrite file configuration
    gParameters = args_overwrite_config(args, fileParameters)
    # Check that required set of parameters has been defined
    bmk.check_required_exists(gParameters)
    print ('Params:')
    pprint(gParameters)
    # Check that no keywords conflict
    check_flag_conflicts(gParameters)

    return gParameters


def get_default_neon_parser(parser):
    """Parse command-line arguments that are default in neon parser (and are common to all frameworks). 
        Ignore if not present.

        Parameters
        ----------
        parser : ArgumentParser object
            Parser for neon default command-line options
    """
    # Logging Level
    parser.add_argument("-v", "--verbose", type=str2bool,
                        help="increase output verbosity")
    parser.add_argument("-l", "--log", dest='logfile',
                        default=None,
                        help="log file")

    # Logging utilities
    parser.add_argument("-s", "--save_path", dest='save_path',
                        default=argparse.SUPPRESS, type=str,
                        help="file path to save model snapshots")

    # General behavior
    parser.add_argument("--model_name", dest='model_name', type=str,
                        default=argparse.SUPPRESS,
                        help="specify model name to use when building filenames for saving")
    parser.add_argument("-d", "--data_type", dest='data_type',
                        default=argparse.SUPPRESS,
                        choices=['f16', 'f32', 'f64'],
                        help="default floating point")

    # Model definition
    # Model Architecture
    parser.add_argument("--dense", nargs='+', type=int,
                        default=argparse.SUPPRESS,
                        help="number of units in fully connected layers in an integer array")

    # Data preprocessing
    #parser.add_argument("--shuffle", type=str2bool,
    #                    default=True,
    #                    help="randomly shuffle data set (produces different training and testing partitions each run depending on the seed)")

    # Training configuration
    parser.add_argument("-r", "--rng_seed", dest='rng_seed', type=int,
                        default=argparse.SUPPRESS,
                        help="random number generator seed")
    parser.add_argument("-e", "--epochs", type=int,
                        default=argparse.SUPPRESS,
                        help="number of training epochs")
    parser.add_argument("-z", "--batch_size", type=int,
                        default=argparse.SUPPRESS,
                        help="batch size")

    return parser


def get_common_parser(parser):
    """Parse command-line arguments. Ignore if not present.

        Parameters
        ----------
        parser : ArgumentParser object
            Parser for command-line options
    """

    # Configuration file
    parser.add_argument("--config_file", dest='config_file', default=argparse.SUPPRESS,
        help="specify model configuration file")

    # General behavior
    parser.add_argument("--train_bool", dest='train_bool', type=str2bool,
                        default=True,
                        help="train model")
    parser.add_argument("--eval_bool", dest='eval_bool', type=str2bool,
                        default=argparse.SUPPRESS,
                        help="evaluate model (use it for inference)")

    parser.add_argument("--timeout", dest='timeout', type=int, action="store",
                    default=argparse.SUPPRESS,
                    help="seconds allowed to train model (default: no timeout)")


    # Logging utilities
    parser.add_argument("--home_dir", dest='home_dir',
                        default=argparse.SUPPRESS, type=str,
                        help="set home directory")

    parser.add_argument("--train_data", action="store",
                        default=argparse.SUPPRESS,
                        help="training data filename")

    parser.add_argument("--test_data", action="store",
                        default=argparse.SUPPRESS,
                        help="testing data filename")

    parser.add_argument("--output_dir", dest='output_dir',
                        default=argparse.SUPPRESS, type=str,
                        help="output directory")

    parser.add_argument("--data_url", dest='data_url',
                        default=argparse.SUPPRESS, type=str,
                        help="set data source url")

    parser.add_argument("--experiment_id", default="EXP000", type=str, help="set the experiment unique identifier")

    parser.add_argument("--run_id", default="RUN000", type=str, help="set the run unique identifier")



    # Model definition
    # Model Architecture
    parser.add_argument("--conv", nargs='+', type=int,
                        default=argparse.SUPPRESS,
                        help="integer array describing convolution layers: conv1_filters, conv1_filter_len, conv1_stride, conv2_filters, conv2_filter_len, conv2_stride ...")
    parser.add_argument("--locally_connected", type=str2bool,
                        default=argparse.SUPPRESS,
                        help="use locally connected layers instead of convolution layers")
    parser.add_argument("-a", "--activation",
                        default=argparse.SUPPRESS,
                        help="keras activation function to use in inner layers: relu, tanh, sigmoid...")
    parser.add_argument("--out_activation",
                        default=argparse.SUPPRESS,
                        help="keras activation function to use in out layer: softmax, linear, ...")


    parser.add_argument("--lstm_size", nargs='+', type=int,
                        default= argparse.SUPPRESS,
                        help="integer array describing size of LSTM internal state per layer")
    parser.add_argument("--recurrent_dropout", action="store",
                        default=argparse.SUPPRESS, type=float,
                        help="ratio of recurrent dropout")


    # Processing between layers
    parser.add_argument("--dropout", type=float,
                        default=argparse.SUPPRESS,
                        help="ratio of dropout used in fully connected layers")
    parser.add_argument("--pool", type=int,
                        default=argparse.SUPPRESS,
                        help="pooling layer length")
    parser.add_argument("--batch_normalization", type=str2bool,
                        default=argparse.SUPPRESS,
                        help="use batch normalization")

    # Model Evaluation
    parser.add_argument("--loss",
                        default=argparse.SUPPRESS,
                        help="keras loss function to use: mse, ...")
    parser.add_argument("--optimizer",
                        default=argparse.SUPPRESS,
                        help="keras optimizer to use: sgd, rmsprop, ...")

    parser.add_argument("--metrics",
                        default=argparse.SUPPRESS,
                        help="metrics to evaluate performance: accuracy, ...")

    # Data preprocessing
    parser.add_argument("--scaling",
                        default=argparse.SUPPRESS,
                        choices=['minabs', 'minmax', 'std', 'none'],
                        help="type of feature scaling; 'minabs': to [-1,1]; 'minmax': to [0,1], 'std': standard unit normalization; 'none': no normalization")

    parser.add_argument("--shuffle", type=str2bool, default=False,
                        help="randomly shuffle data set (produces different training and testing partitions each run depending on the seed)")

    # Feature selection
    parser.add_argument("--feature_subsample", type=int,
                        default=argparse.SUPPRESS,
                        help="number of features to randomly sample from each category (cellline expression, drug descriptors, etc), 0 means using all features")

    # Training configuration
    parser.add_argument("--learning_rate",
                        default= argparse.SUPPRESS, type=float,
                        help="overrides the learning rate for training")
    parser.add_argument("--early_stop", type=str2bool,
                        default= argparse.SUPPRESS,
                        help="activates keras callback for early stopping of training in function of the monitored variable specified")
    parser.add_argument("--momentum",
                        default= argparse.SUPPRESS, type=float,
                        help="overrides the momentum to use in the SGD optimizer when training")

    parser.add_argument("--initialization",
                        default=argparse.SUPPRESS,
                        choices=['constant', 'uniform', 'normal', 'glorot_uniform', 'lecun_uniform', 'he_normal'],
                        help="type of weight initialization; 'constant': to 0; 'uniform': to [-0.05,0.05], 'normal': mean 0, stddev 0.05; 'glorot_uniform': [-lim,lim] with lim = sqrt(6/(fan_in+fan_out)); 'lecun_uniform' : [-lim,lim] with lim = sqrt(3/fan_in); 'he_normal' : mean 0, stddev sqrt(2/fan_in)")
    parser.add_argument("--val_split", type=float,
                        default=argparse.SUPPRESS,
                        help="fraction of data to use in validation")
    parser.add_argument("--train_steps", type=int,
                        default=argparse.SUPPRESS,
                        help="overrides the number of training batches per epoch if set to nonzero")
    parser.add_argument("--val_steps", type=int,
                        default=argparse.SUPPRESS,
                        help="overrides the number of validation batches per epoch if set to nonzero")
    parser.add_argument("--test_steps", type=int,
                        default=argparse.SUPPRESS,
                        help="overrides the number of test batches per epoch if set to nonzero")
    parser.add_argument("--train_samples", action="store",
                        default=argparse.SUPPRESS, type=int,
                        help="overrides the number of training samples if set to nonzero")
    parser.add_argument("--val_samples", action="store",
                        default=argparse.SUPPRESS, type=int,
                        help="overrides the number of validation samples if set to nonzero")


    # Backend configuration
    parser.add_argument("--gpus", nargs="*",
                        default=argparse.SUPPRESS,
                        #default=[0],
                        type=int,
                        help="set IDs of GPUs to use")

    # profiling flags
    parser.add_argument("-p", "--profiling", type=str2bool,
                        default = 'false',
                        help="Turn profiling on or off")

    # cyclic learning rate
    parser.add_argument("--clr_flag", 
                        default=argparse.SUPPRESS,
                        #default=None, 
                        type=str2bool, 
                        help="CLR flag (boolean)")
    parser.add_argument("--clr_mode", 
                        default=argparse.SUPPRESS,
                        #default=None, 
                        type=str, choices=['trng1', 'trng2', 'exp'], 
                        help="CLR mode (default: trng1)")
    parser.add_argument("--clr_base_lr", type=float, 
                        default=argparse.SUPPRESS,
                        #default=1e-4, 
                        help="Base lr for cycle lr.")
    parser.add_argument("--clr_max_lr", type=float, 
                        default=argparse.SUPPRESS,
                        #default=1e-3, 
                        help="Max lr for cycle lr.")
    parser.add_argument("--clr_gamma", type=float, 
                        default=argparse.SUPPRESS,
                        #default=0.999994, 
                        help="Gamma parameter for learning cycle LR.")

    return parser



def args_overwrite_config(args, config):
    """Overwrite configuration parameters with 
        parameters specified via command-line.

        Parameters
        ----------
        args : ArgumentParser object
            Parameters specified via command-line
        config : python dictionary
            Parameters read from configuration file
    """

    params = config

    args_dict = vars(args)

    for key in args_dict.keys():
        params[key] = args_dict[key]


    if 'data_type' not in params:
        params['data_type'] = DEFAULT_DATATYPE
    else:
        if params['data_type'] in set(['f16', 'f32', 'f64']):
            params['data_type'] = get_choice(params['datatype'])

    if 'output_dir' not in params:
        params['output_dir'] = directory_from_parameters(params)
    else:
        params['output_dir'] = directory_from_parameters(params, params['output_dir'])

    if 'rng_seed' not in params:
        params['rng_seed'] = DEFAULT_SEED

    if 'timeout' not in params:
        params['timeout'] = DEFAULT_TIMEOUT


    return params



def get_choice(name):
    """ Maps name string to the right type of argument
    """
    mapping = {}

    # dtype
    mapping['f16'] = np.float16
    mapping['f32'] = np.float32
    mapping['f64'] = np.float64

    mapped = mapping.get(name)
    if not mapped:
        raise Exception('No mapping found for "{}"'.format(name))

    return mapped


def directory_from_parameters(params, commonroot='Output'):
    """ Construct output directory path with unique IDs from parameters

        Parameters
        ----------
        params : python dictionary
            Dictionary of parameters read
        commonroot : string
            String to specify the common folder to store results.

    """

    if commonroot in set(['.', './']): # Same directory --> convert to absolute path
        outdir = os.path.abspath('.')
    else: # Create path specified
        outdir = os.path.abspath(os.path.join('.', commonroot))
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outdir = os.path.abspath(os.path.join(outdir, params['experiment_id']))
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outdir = os.path.abspath(os.path.join(outdir, params['run_id']))
        if not os.path.exists(outdir):
            os.makedirs(outdir)


    return outdir



class Benchmark:
    """ Class that implements an interface to handle configuration options for the
        different CANDLE benchmarks.
        It provides access to all the common configuration
        options and configuration options particular to each individual benchmark.
        It describes what minimum requirements should be specified to instantiate
        the corresponding benchmark.
        It interacts with the argparser to extract command-line options and arguments
        from the benchmark's configuration files.
    """

    def __init__(self, filepath, defmodel, framework, prog=None, desc=None, parser=None):
        """ Initialize Benchmark object.

            Parameters
            ----------
            filepath : ./
                os.path.dirname where the benchmark is located. Necessary to locate utils and
                establish input/ouput paths
            defmodel : 'p*b*_default_model.txt'
                string corresponding to the default model of the benchmark
            framework : 'keras', 'neon', 'mxnet', 'pytorch'
                framework used to run the benchmark
            prog : 'p*b*_baseline_*'
                string for program name (usually associated to benchmark and framework)
            desc : ' '
                string describing benchmark (usually a description of the neural network model built)
            parser : argparser (default None)
                if 'neon' framework a NeonArgparser is passed. Otherwise an argparser is constructed.
        """

        if parser is None:
            parser = argparse.ArgumentParser(prog=prog, formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=desc, conflict_handler='resolve')

        self.parser = parser
        self.file_path = filepath
        self.default_model = defmodel
        self.framework = framework

        self.required = set([])
        self.additional_definitions = []
        self.set_locals()



    def parse_from_common(self):
        """Functionality to parse options common
           for all benchmarks.
           This functionality is based on methods 'get_default_neon_parser' and
           'get_common_parser' which are defined previously(above). If the order changes
           or they are moved, the calling has to be updated.
        """


        # Parse has been split between arguments that are common with the default neon parser
        # and all the other options
        parser = self.parser
        if self.framework is not 'neon':
            parser = get_default_neon_parser(parser)
        parser = get_common_parser(parser)

        self.parser = parser

        # Set default configuration file
        self.conffile = os.path.join(self.file_path, self.default_model)


    def parse_from_benchmark(self):
        """Functionality to parse options specific
           specific for each benchmark.
        """

        for d in self.additional_definitions:
            if 'type' not in d:
                d['type'] = None
            if 'default' not in d:
                d['default'] = argparse.SUPPRESS
            if 'help' not in d:
                d['help'] = ''
            if 'action' in d: # Actions
                if d['action'] == 'list-of-lists': # Non standard. Specific functionallity has been added
                    d['action'] = ListOfListsAction
                    self.parser.add_argument('--' + d['name'], dest=d['name'], action=d['action'], type=d['type'], default=d['default'], help=d['help'])
                elif (d['action'] == 'store_true') or (d['action'] == 'store_false'):
                    raise Exception ('The usage of store_true or store_false cannot be undone in the command line. Use type=str2bool instead.')
                else:
                    self.parser.add_argument('--' + d['name'], action=d['action'], default=d['default'], help=d['help'])
            else: # Non actions
                if 'nargs' in d: # variable parameters
                    if 'choices' in d: # choices with variable parameters
                        self.parser.add_argument('--' + d['name'], nargs=d['nargs'], choices=d['choices'], default=d['default'], help=d['help'])
                    else: # Variable parameters (free, no limited choices)
                        self.parser.add_argument('--' + d['name'], nargs=d['nargs'], type=d['type'], default=d['default'], help=d['help'])
                elif 'choices' in d: # Select from choice (fixed number of parameters)
                    self.parser.add_argument('--' + d['name'], choices=d['choices'], default=d['default'], help=d['help'])
                else: # Non an action, one parameter, no choices
                    self.parser.add_argument('--' + d['name'], type=d['type'], default=d['default'], help=d['help'])



    def format_benchmark_config_arguments(self, dictfileparam):
        """ Functionality to format the particular parameters of
            the benchmark.

            Parameters
            ----------
            dictfileparam : python dictionary
                parameters read from configuration file
            args : python dictionary
                parameters read from command-line
                Most of the time command-line overwrites configuration file
                except when the command-line is using default values and
                config file defines those values

        """

        configOut = dictfileparam.copy()

        for d in self.additional_definitions:
            if d['name'] in configOut.keys():
                if 'type' in d:
                    dtype = d['type']
                else:
                    dtype = None

                if 'action' in d:
                    if inspect.isclass(d['action']):
                        str_read = dictfileparam[d['name']]
                        configOut[d['name']] = eval_string_as_list_of_lists(str_read, ':', ',', dtype)
                elif d['default'] != argparse.SUPPRESS:
                    # default value on benchmark definition cannot overwrite config file
                    self.parser.add_argument('--' + d['name'], type=d['type'], default=configOut[d['name']], help=d['help'])

        return configOut



    def read_config_file(self, file):
        """Functionality to read the configue file
           specific for each benchmark.
        """

        config=configparser.ConfigParser()
        config.read(file)
        section=config.sections()
        fileParams={}

        # parse specified arguments (minimal validation: if arguments
        # are written several times in the file, just the first time
        # will be used)
        for sec in section:
            for k,v in config.items(sec):
                if not k in fileParams:
                    fileParams[k] = eval(v)
        fileParams = self.format_benchmark_config_arguments(fileParams)
        #pprint(fileParams)

        return fileParams



    def set_locals(self):
        """ Functionality to set variables specific for the benchmark
            - required: set of required parameters for the benchmark.
            - additional_definitions: list of dictionaries describing \
                the additional parameters for the benchmark.
        """

        pass



    def check_required_exists(self, gparam):
        """Functionality to verify that the required 
           model parameters have been specified.
        """

        key_set = set(gparam.keys())
        intersect_set = key_set.intersection(self.required)
        diff_set = self.required.difference(intersect_set)

        if ( len(diff_set) > 0 ):
            raise Exception('ERROR ! Required parameters are not specified. ' \
            'These required parameters have not been initialized: ' + str(sorted(diff_set)) + \
            '... Exiting')



def keras_default_config():
    """Defines parameters that intervine in different functions using the keras defaults.
        This helps to keep consistency in parameters between frameworks.
    """

    kerasDefaults = {}

    # Optimizers
    #kerasDefaults['clipnorm']=?            # Maximum norm to clip all parameter gradients
    #kerasDefaults['clipvalue']=?          # Maximum (minimum=-max) value to clip all parameter gradients
    kerasDefaults['decay_lr']=0.            # Learning rate decay over each update
    kerasDefaults['epsilon']=1e-8           # Factor to avoid divide by zero (fuzz factor)
    kerasDefaults['rho']=0.9                # Decay parameter in some optmizer updates (rmsprop, adadelta)
    kerasDefaults['momentum_sgd']=0.        # Momentum for parameter update in sgd optimizer
    kerasDefaults['nesterov_sgd']=False     # Whether to apply Nesterov momentum in sgd optimizer
    kerasDefaults['beta_1']=0.9             # Parameter in some optmizer updates (adam, adamax, nadam)
    kerasDefaults['beta_2']=0.999           # Parameter in some optmizer updates (adam, adamax, nadam)
    kerasDefaults['decay_schedule_lr']=0.004# Parameter for nadam optmizer

    # Initializers
    kerasDefaults['minval_uniform']=-0.05   #  Lower bound of the range of random values to generate
    kerasDefaults['maxval_uniform']=0.05    #  Upper bound of the range of random values to generate
    kerasDefaults['mean_normal']=0.         #  Mean of the random values to generate
    kerasDefaults['stddev_normal']=0.05     #  Standard deviation of the random values to generate


    return kerasDefaults

