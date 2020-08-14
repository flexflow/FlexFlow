from __future__ import print_function

import os
import sys
import logging
import argparse
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

from default_utils import Benchmark, str2bool

logger = logging.getLogger(__name__)


class BenchmarkUno(Benchmark):

    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


additional_definitions = [
    # Feature selection
    {'name': 'agg_dose',
        'type': str,
        'default': None,
        'choices': ['AUC', 'IC50', 'EC50', 'HS', 'AAC1', 'AUC1', 'DSS1'],
        'help': 'use dose-independent response data with the specified aggregation metric'},
    {'name': 'cell_features',
        'nargs': '+',
        'choices': ['rnaseq', 'none'],
        'help': 'use rnaseq cell line feature set or none at all'},
    {'name': 'drug_features',
        'nargs': '+',
        'choices': ['descriptors', 'fingerprints', 'none', 'mordred'],
        'help': 'use dragon7 descriptors or fingerprint descriptors for drug features or none at all'},
    {'name': 'by_cell',
        'type': str,
        'default': None,
        'help': 'sample ID for building a by-cell model'},
    {'name': 'by_drug',
        'type': str,
        'default': None,
        'help': 'drug ID or name for building a by-drug model'},
    # Data set selection
    {'name': 'train_sources',
        'nargs': '+',
        'choices': ['all', 'CCLE', 'CTRP', 'gCSI', 'GDSC', 'NCI60', 'SCL', 'SCLC', 'ALMANAC'],
        'help': 'use one or more sources of drug response data for training'},
    {'name': 'test_sources',
        'nargs': '+',
        'choices': ['train', 'all', 'CCLE', 'CTRP', 'gCSI', 'GDSC', 'NCI60', 'SCL', 'SCLC', 'ALMANAC'],
        'help': 'use one or more sources of drug response data for testing'},
    # Sample selection
    {'name': 'cell_types',
        'nargs': '+',
        'help': 'limit training and test data to one or more tissue types'},
    {'name': 'cell_subset_path',
        'type': str,
        'default': '',
        'help': 'path for file with space delimited molecular sample IDs to keep'},
    {'name': 'drug_subset_path',
        'type': str,
        'default': '',
        'help': 'path for file with space delimited drug IDs to keep'},
    {'name': 'drug_median_response_min',
        'type': float,
        'default': -1,
        'help': 'keep drugs whose median response is greater than the threshold'},
    {'name': 'drug_median_response_max',
        'type': float,
        'default': 1,
        'help': 'keep drugs whose median response is less than the threshold'},
    # Training
    {'name': 'no_feature_source',
        'type': str2bool,
        'default': False,
        'help': 'do not embed cell or drug feature source as part of input'},
    {'name': 'no_response_source',
        'type': str2bool,
        'default': False,
        'help': 'do not encode response data source as an input feature'},
    {'name': 'dense_feature_layers',
        'nargs': '+',
        'type': int,
        'help': 'number of neurons in intermediate dense layers in the feature encoding submodels'},
    {'name': 'dense_cell_feature_layers',
        'nargs': '+',
        'type': int,
        'default': None,
        'help': 'number of neurons in intermediate dense layers in the cell feature encoding submodels'},
    {'name': 'dense_drug_feature_layers',
        'nargs': '+',
        'type': int,
        'default': None,
        'help': 'number of neurons in intermediate dense layers in the drug feature encoding submodels'},
    {'name': 'use_landmark_genes',
        'type': str2bool,
        'default': False,
        'help': 'use the 978 landmark genes from LINCS (L1000) as expression features'},
    {'name': 'use_filtered_genes',
        'type': str2bool,
        'default': False,
        'help': 'use the variance filtered genes as expression features'},
    {'name': 'feature_subset_path',
        'type': str,
        'default': '',
        'help': 'path for file with space delimited features to keep'},
    {'name': 'cell_feature_subset_path',
        'type': str,
        'default': '',
        'help': 'path for file with space delimited molecular features to keep'},
    {'name': 'drug_feature_subset_path',
        'type': str,
        'default': '',
        'help': 'path for file with space delimited drug features to keep'},
    {'name': 'preprocess_rnaseq',
        'choices': ['source_scale', 'combat', 'none'],
        'default': 'none',
        'help': 'preprocessing method for RNAseq data; none for global normalization'},
    {'name': 'residual',
        'type': str2bool,
        'default': False,
        'help': 'add skip connections to the layers'},
    {'name': 'reduce_lr',
        'type': str2bool,
        'default': False,
        'help': 'reduce learning rate on plateau'},
    {'name': 'warmup_lr',
        'type': str2bool,
        'default': False,
        'help': 'gradually increase learning rate on start'},
    {'name': 'base_lr',
        'type': float,
        'default': None,
        'help': 'base learning rate'},
    {'name': 'es',
        'type': str2bool,
        'default': False,
        'help': 'early stopping on val_loss'},
    {'name': 'cp',
        'type': str2bool,
        'default': False,
        'help': 'checkpoint models with best val_loss'},
    {'name': 'tb',
        'type': str2bool,
        'default': False,
        'help': 'use tensorboard'},
    {'name': 'tb_prefix',
        'type': str,
        'default': 'tb',
        'help': 'prefix name for tb log'},
    {'name': 'max_val_loss',
        'type': float,
        'default': argparse.SUPPRESS,
        'help': 'retrain if val_loss is greater than the threshold'},
    {'name': 'partition_by',
        'choices': ['index', 'drug_pair', 'cell'],
        'default': None,
        'help': 'cross validation paritioning scheme'},
    {'name': 'cv',
        'type': int,
        'default': argparse.SUPPRESS,
        'help': 'cross validation folds'},
    {'name': 'no_gen',
        'type': str2bool,
        'default': False,
        'help': 'do not use generator for training and validation data'},
    {'name': 'cache',
        'type': str,
        'default': None,
        'help': 'prefix of data cache files to use'},
    {'name': 'single',
        'type': str2bool,
        'default': False,
        'help': 'do not use drug pair representation'},
    {'name': 'export_csv',
        'type': str,
        'default': None,
        'help': 'output csv file name'},
    {'name': 'export_data',
        'type': str,
        'default': None,
        'help': 'output dataframe file name'},
    {'name': 'use_exported_data',
        'type': str,
        'default': None,
        'help': 'exported file name'},
    {'name': 'growth_bins',
        'type': int,
        'default': 0,
        'help': 'number of bins to use when discretizing growth response'},
    {'name': 'initial_weights',
        'type': str,
        'default': None,
        'help': 'file name of initial weights'},
    {'name': 'save_weights',
        'type': str,
        'default': None,
        'help': 'name of file to save weights to'}
]

required = [
    'activation',
    'batch_size',
    'dense',
    'dense_feature_layers',
    'dropout',
    'epochs',
    'feature_subsample',
    'learning_rate',
    'loss',
    'optimizer',
    'residual',
    'rng_seed',
    'save_path',
    'scaling',
    'val_split',
    'timeout'
]
