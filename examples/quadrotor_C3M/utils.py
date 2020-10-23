import gzip
import pickle
def savepklz(data_to_dump, dump_file_full_name):
    ''' Saves a pickle object and gzip it '''

    with gzip.open(dump_file_full_name, 'wb') as out_file:
        pickle.dump(data_to_dump, out_file)


def loadpklz(dump_file_full_name):
    ''' Loads a gziped pickle object '''

    with gzip.open(dump_file_full_name, 'rb') as in_file:
        dump_data = pickle.load(in_file)

    return dump_data