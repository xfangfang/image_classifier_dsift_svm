
# image classifier powered by dsift+svm

# How to Use

    1. unzip the dataset
    2. run ``` python random_file.py ``` to split the dataset to train set and test set.
        current dir

        new_data/
            bird/
                img001.jpg
                ...
            puppy/
                ...
            ...
        train/
            bird/
                img034.jpg
                ...
            puppy/
                ...
            ...
        test/
            bird/
                img011.jpg
                ...
            puppy/
                ...
            ...
    3. run ``` python main.py ```

# result

    this algorithm got 91% top5-right in a 21 classes dataset(about 4500 pictures, not the dataset in this project)


# notice

    the pictures in the dataset are from network, if you don't want me to use it, please let me know.

# tips

    change number in load.loaddata.standarizeImage method to a large number can get more better score.
