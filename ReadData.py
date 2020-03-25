import random
from typing import Any, List
import numpy as np
from Bio import SeqIO
from sklearn import svm, metrics
from sklearn.preprocessing import LabelBinarizer


def main():
    read_vector, ID_vector = read_data()  # type: (List[List[Any]], List[List[Any]])
    train_set, train_ID, test_set, test_ID = seperate_sets(read_vector, ID_vector)
    train_set, train_ID, test_set, test_ID = binary_labeler(train_set, train_ID, test_set, test_ID)
    learn(train_set, train_ID, test_set, test_ID)


def read_data(File_path: str = '/home/theox/Documents/School/Course10a/DataScience/train_set.fasta') -> (
        List[List[Any]], List[List[Any]]):
    """

    :type File_path: str
    """
    print("Reading data...")
    read_vector = []  # type: List[List[Any]]
    ID_vector = []  # type: List[List[Any]]

    for read in SeqIO.parse(File_path, "fasta"):
        if read.id.split("|")[2] == "NO_SP" or read.id.split("|")[2] == "SP":
            read_vector.append(list(read.seq[0:30]))
            ID_vector.append(read.id.split("|")[2])

    return read_vector, ID_vector


def seperate_sets(read_vector: List[List[Any]], ID_vector: List[List[Any]]):
    """

    :param read_vector: List[List[Any]]
    :param ID_vector: List[List[Any]]
    :return:
    """
    test_select = random.sample(range(len(read_vector)), int(len(read_vector) * 0.20))
    test_set = []
    test_IDs = []

    test_select.sort(reverse=True)

    for place in test_select:
        test_set.append(read_vector[place])
        read_vector.remove(read_vector[place])
        test_IDs.append(ID_vector[place])
        ID_vector.remove(ID_vector[place])

    return read_vector, ID_vector, test_set, test_IDs


def binary_labeler(train_set: List[List[Any]], train_ID: List[List[Any]], test_set: List[List[Any]],
                   test_ID: List[List[Any]]):
    """

    :param train_set:
    :param train_ID:
    :param test_set:
    :param test_ID:
    :return:
    """
    enc_acids = LabelBinarizer()
    enc_acids.fit(['A', "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"])
    print("Converting test set to binary...")
    test_set, test_ID = binary_converter(test_set, test_ID, enc_acids)
    print("Converting train set to binary...")
    train_set, train_ID = binary_converter(train_set, train_ID, enc_acids)

    return train_set, train_ID, test_set, test_ID


def binary_converter(amino_set: List[List[Any]], ID_set: List[List[Any]], binerizer: LabelBinarizer) -> object:
    """

    :param amino_set:
    :param ID_set:
    :param binerizer:
    :return:
    """
    new_amino_set = [[]]
    for position, read in enumerate(amino_set):
        new_amino_set.append([])
        temp_list = binerizer.transform(read).tolist()
        final_list = []
        for sublist in temp_list:
            for item in sublist:
                final_list.append(item)
        new_amino_set[position].extend(np.array(final_list, object))
        if ID_set[position] == "SP":
            ID_set[position] = 1
        elif ID_set[position] == "NO_SP":
            ID_set[position] = 0
        else:
            print("{} is not SP or NO_SP".format(position))
    new_amino_set.remove([])
    return np.array(new_amino_set, object), ID_set


def learn(train_set: List[List[Any]], train_ID: List[List[Any]], test_set: List[List[Any]], test_ID: List[List[Any]]):
    """

    :rtype: object
    :param test_set: List[List[Any]]
    :param test_ID: List[List[Any]]
    :param train_ID: List[List[Any]]
    :type train_set: List[List[Any]]
    """
    svc = svm.SVC(kernel='linear')  # constructor
    print("Start to train set.")
    svc.fit(train_set, train_ID)  # fit classifier data
    print("Predicting...")
    predicted = svc.predict(test_set)  # predicts class_id of test set <- "test"
    print("Evaluating...")
    score = svc.score(test_set, test_ID)  # evaluate prediction quality/performance <- "evaluation"

    print('============================================')
    print('\nScore ', score)
    print('\nResult Overview\n', metrics.classification_report(test_ID, predicted))
    print('\nConfusion matrix:\n', metrics.confusion_matrix(test_ID, predicted))


if __name__ == '__main__':
    main()
