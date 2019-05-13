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
    # Training data
    states_train = []   # Holds the state of the corresponding score
    data_train = [] # Holds the data for that state
    targetOrig_train = []   # Holds the average math 8 score for that state

    states_test = []    # Testing data
    data_test = []
    targetOrig_test = []

    i = 0
    with open('states_all_extended.csv', 'r') as csvfile:    # Read all the relevant data and save it
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            if i != 0:
                if row[1] != '' and row[2] != '' and row[3] != '' and row[4] != '' and row[6] != '' and row[7] != '' and row[8] \
                != '' and row[9] != '' and row[10] != '' and row[11] != '' and row[12] != '' and row[15] != '' and row[16] != '' \
                and row[189] != '' and row[190] != '':
                    if i < 300:
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
            else:
                i += 1

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
    model = lm.fit(X,y) # Create our model

    predictions = lm.predict(X) # Make predictions on our training data
    predictions_test = lm.predict(X_test)   # Make predictions on our testing data

    # Print Results
    print("Results:")
    print("Mean Squared Error in-sample: " + str(mean_squared_error(target, predictions)))
    print("Mean Squared Error out-of-sample: " + str(mean_squared_error(target_test, predictions_test)))

    # Dot Plot Training Data
    plt.title("Average Math 8 Scores, Training Data")
    plt.ylabel("Scores")
    plt.yticks(np.arange(0,500,50))
    p1 = plt.plot(np.arange(len(predictions)), [float(x) for x in predictions], 'blue')
    p2 = plt.plot(np.arange(len(targetOrig_train)), [float(x) for x in targetOrig_train], 'ro')
    plt.legend((p1[0], p2[0]), ('Predictions', 'Target Values'))
    plt.show()

    # Dot Plot Testing Data
    plt.title("Average Math 8 Scores, Testing Data")
    plt.ylabel("Scores")
    plt.yticks(np.arange(0,500,50))
    p1 = plt.plot(np.arange(len(predictions_test)), [float(x) for x in predictions_test], 'blue')
    p2 = plt.plot(np.arange(len(targetOrig_test)), [float(x) for x in targetOrig_test], 'ro')
    plt.legend((p1[0], p2[0]), ('Predictions', 'Target Values'))
    plt.show()

    # Bar graph
    while True:
        x1 = int(input("X1: "))
        x2 = int(input("X2: "))
        N = x2 - x1
        targetBar = [float(targetOrig_train[i]) for i in range(x1,x2)]
        predictionsBar = [float(predictions[i]) for i in range(x1,x2)]
        print(predictionsBar)
        ind = np.arange(N)
        width = 0.20

        p2 = plt.bar(ind, predictionsBar, width)
        p1 = plt.bar(ind, targetBar, width)

        plt.ylabel('Scores')
        plt.title('Average Math 8 Score, Training Data')
        plt.xticks(ind, [states_train[i] + '_' + data_train[i][0] for i in range(x1,x2)])
        plt.yticks(np.arange(0,500,10))
        plt.legend((p1[0], p2[0]), ('Target Values', 'Predictions'))

        plt.show()



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
    # plt.show()


if __name__ == "__main__":
    main()
