import os
import numpy as np
import pandas as pd

def initialize_dataset(subject_list):
    """
    Initializes dataset from subject_list and files found in data folder.
    
    Parameters
    ----------
    subject_list: list
        List of subject_ids to include in dataset.

    Returns
    -------
    data_list: numpy.ndarray
        Numpy array of data with shape (subjects, sliding window, fNIRS samples, and fNIRS features).
    label_list: numpy.ndarray
        Numpy array of labels with shape (subjects, sliding window label).
    """
    data_list, label_list = [], []
    filenames = os.listdir(os.getcwd()+"/data/")
    for filename in filenames:
        if not filename.startswith("."):
            subject_id = os.path.basename(filename).strip("sub_.csv")
            if int(subject_id) in subject_list:
                df = pd.read_csv(os.getcwd()+"/data/"+filename)
                grouped = df.groupby(["chunk","label"])
                grouped_list, labels = [], []
                for tuple, group in grouped:
                    data = group.drop(columns=["chunk","label"]).to_numpy(dtype=np.float32)
                    grouped_list.append(data)
                    labels.append(tuple[1])
                data_list.append(grouped_list)
                label_list.append(labels)
    return np.array(data_list), np.array(label_list)

def plot(data):
    """
    Plots 
    
    Parameters
    ----------
    data: numpy.ndarray
        Numpy array of data with shape (fNIRS features, and fNIRS samples).

    Returns
    -------
    fig: matplotlib.figure.Figure
        A visualization of the data.

    """
    # initialize figure
    fig, axs = plt.subplots(9, 1, figsize=(12, 3), sharey=True, sharex=True, gridspec_kw={"wspace":0, "hspace":0}, squeeze=True)
    cmaps = ["Reds","Reds","Blues","Blues"]
    for i in range(8):
        index = i if i<4 else i+1
        axs[index].imshow(data[:][i].reshape(1,len(data[:][i])), cmap=cmaps[i%4], aspect="auto")
    axs[4].imshow(np.zeros((1,len(data[:][i]))), vmax=1, vmin=0, cmap="binary", aspect="auto")
    # set graph axis
    axs[8].set_xlabel("Time (seconds)")
    # divide axis by 5
    a = axs[8].get_xticks().tolist()
    a = [int(i/5) for i in a]
    axs[8].set_xticklabels(a)
    yaxis = ["AB_I_O","AB_PHI_O","AB_I_DO","AB_PHI_DO","","CD_I_O","CD_PHI_O","CD_I_DO","CD_PHI_DO"]
    for i in range(9):
        axs[i].grid(False)
        axs[i].set_yticks([])
        axs[i].set_ylabel(yaxis[i], rotation=0, labelpad=40, loc="center")
    # return figure
    return fig

def round_predictions(predictions, lower, upper):
    predictions[predictions<=lower] = lower
    predictions[predictions>=upper] = upper
    return predictions

def subject_train_test_split(X, y):
    n_samples, n_features = X.shape
    split = int(n_samples*10/16)
    X_train = X[0:split]
    X_test = X[split:len(X)]
    y_train = y[0:split]
    y_test = y[split:len(X)] 
    return X_train, X_test, y_train, y_test