import csv
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

# Row 1: State; Row 2: Year; Row 3: Enroll; Row 4: Total Revenue; Row 5: Federal Revenue; Row 6: State Revenue; Row 7:
# Local Revenue; Row 8: Total Expenditure; Row 9: Instruction Expenditure; Row 10: Support Services Expenditure;  Row 11:
# Other Expenditure; Row 12: Capital Outlay Expenditure; Row 15: Grades 4 G; Row 16: Grades 8 G; Row 21: Avg Math 4 score
# Row 22: Avg Math 8 Score; In extended, Row 189?: Avg Math 4 Score; Row 190?: Avg Math 8 Score
def main():
    states_train = []   # Training data
    data_train = []
    targetOrig_train = []

    states_test = []    # Testing data
    data_test = []
    targetOrig_test = []

    i = 0
    with open('states_all_extended.csv', 'r') as csvfile:    # Read all the relevant data and save it
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            if row[1] != '' and row[2] != '' and row[3] != '' and row[4] != '' and row[6] != '' and row[7] != '' and row[8] \
            != '' and row[9] != '' and row[10] != '' and row[11] != '' and row[12] != '' and row[15] != '' and row[16] != '' \
            and row[189] != '' and row[190] != '':
                if i < 430:
                    datalist = []
                    states_train.append(row[1])
                    datalist.extend((row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[15], row[16], row[189]))
                    targetOrig_train.append(row[190])
                    data_train.append(datalist)
                    i += 1
                else:
                    datalist = []
                    states_test.append(row[1])
                    datalist.extend((row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[15], row[16], row[189]))
                    targetOrig_test.append(row[190])
                    data_test.append(datalist)

    del data_train[0]
    del targetOrig_train[0]
    del data_test[0]
    del targetOrig_test[0]
    del states_train[0]
    del states_test[0]

    df = pd.DataFrame(data_train, columns=["Year", "Enroll", "Total Revenue", "Federal Revenue", "State Revenue", "Local Revenue", \
    "Total Expenditure", "Instruction Expenditure", "Support Services Expenditure", "Other Expenditure", \
    "Capital Outlay Expenditure", "Grades 4 G", "Grades 8 G", "Avg Math 4 Score"])
    dt = pd.DataFrame(data_test, columns=["Year", "Enroll", "Total Revenue", "Federal Revenue", "State Revenue", "Local Revenue", \
    "Total Expenditure", "Instruction Expenditure", "Support Services Expenditure", "Other Expenditure", \
    "Capital Outlay Expenditure", "Grades 4 G", "Grades 8 G", "Avg Math 4 Score"])

    target = pd.DataFrame(targetOrig_train, columns=["Average Math 8 Score"])
    target_test = pd.DataFrame(targetOrig_test, columns=["Average Math 8 Score"])

    X = df
    X_test = dt
    y = target["Average Math 8 Score"]

    lm = linear_model.LinearRegression()
    model = lm.fit(X,y)

    predictions = lm.predict(X)
    predictions_test = lm.predict(X_test)

    print("Results:")
    print("Score of in-sample error: " + str(lm.score(X, predictions)))
    print("Score of out-of-sample error: " + str(lm.score(X_test, predictions_test)))
    print("Mean Squared Error in-sample: " + str(mean_squared_error(target, predictions)))
    print("Mean Squared Error out-of-sample: " + str(mean_squared_error(target_test, predictions_test)))

    y = []
    x = np.array(range(0, 500))
    for value in x:
        total = 0
        for coef in lm.coef_:
            total += value*coef
        y.append(total)
    y = np.array(y)

    # Create the plot and show
    plt.plot(x,y)
    # plt.plot(np.array(range())), targetOrig_train, 'ro')
    plt.plot(np.arange(len(targetOrig_train)), targetOrig_train, 'ro')
    plt.show()

    #
    # plt.plot(np.arange(len(states_train)), predictions)
    # plt.grid(True)
    #
    # plt.show()
    #
    # y_pos = np.arange(len(states_train))
    # plt.bar(y_pos, predictions, align='center', alpha=1)
    # # plt.bar(y_pos, target, align='center', alpha=1)
    # plt.xticks(y_pos, states_train)
    # plt.ylabel('Score')
    # plt.title('Average Math 8 Score')
    #
    # plt.show()

if __name__ == "__main__":
    main()
