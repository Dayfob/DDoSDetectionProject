import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import RFE

data_set = pd.read_csv('datasets/dataset-preprocessed.csv')

var = data_set.loc[(data_set['syn_flag_number'] == 1.0) & (data_set['label'] == "BenignTraffic")]

print(data_set["syn_flag_number"].value_counts())
print(var['syn_flag_number'])
print(var['label'])

# Random Forest Feature Importances ====================================================================================

# data_set_without_syn_flag = data_set.drop(["syn_flag_number"], axis=1)
# data_set_without_syn_flag.to_csv('datasets/dimensionality_reduction/random_forest_feature_importances/dataset-syn-flag.csv', index=False)

# Missing Value Ratio ==================================================================================================

# data_set = pd.read_csv('datasets/dimensionality_reduction/random_forest_feature_importances/dataset-syn-flag.csv')
# print(data_set.isnull().sum() / len(data_set) * 100)

# Low Variance Filter ==================================================================================================

# data_set = pd.read_csv('datasets/dimensionality_reduction/random_forest_feature_importances/dataset-syn-flag.csv')
# data_set_without_labels = data_set.iloc[:, :-1]
#
# print(data_set_without_labels.var())

# data_set_red = data_set.drop(
#     ["Drate", "fin_flag_number", "ece_flag_number", "cwr_flag_number", "Telnet", "SMTP",
#      "SSH", "IRC", "DHCP", "ICMP"], axis=1)
#
# data_set_red.to_csv('datasets/dimensionality_reduction/low_variance_filter/dataset-red.csv', index=False)
#
# data_set_orange = data_set.drop(
#     ["Drate", "fin_flag_number", "ece_flag_number", "cwr_flag_number", "Telnet", "SMTP",
#      "SSH", "IRC", "DHCP", "ICMP",
#      "Protocol Type", "Duration", "ack_flag_number", "syn_count", "HTTPS", "UDP", "IPv", "LLC",
#      "Tot sum", "Max", "AVG", "Std", "Tot size", "IAT", "Magnitue", "Variance"], axis=1)
#
#
# data_set_orange.to_csv('datasets/dimensionality_reduction/low_variance_filter/dataset-orange.csv', index=False)
#
# data_set_yellow = data_set.drop(
#     ["Drate", "fin_flag_number", "ece_flag_number", "cwr_flag_number", "Telnet", "SMTP",
#      "SSH", "IRC", "DHCP", "ICMP",
#      "Protocol Type", "Duration", "ack_flag_number", "syn_count", "HTTPS", "UDP", "IPv", "LLC",
#      "Tot sum", "Max", "AVG", "Std", "Tot size", "IAT", "Magnitue", "Variance",
#      "rst_flag_number", "psh_flag_number", "DNS", "TCP", "Min", "Number", "Radius", "Weight"], axis=1)
#
#
# data_set_yellow.to_csv('datasets/dimensionality_reduction/low_variance_filter/dataset-yellow.csv', index=False)


# High Correlation filter ==============================================================================================

# data_set_orange = pd.read_csv('datasets/dimensionality_reduction/low_variance_filter/dataset-orange.csv')

# corr_data_set_orange = data_set_orange.drop(["label"], axis=1)
#
# corr_data_set_orange.corr().to_csv('datasets/dimensionality_reduction/high_correlation_filter/correlation-orange.csv',
#                                    index=False)


# corr_data_set_orange_0 = data_set_orange.drop(["Srate", "rst_count", "Weight"], axis=1)
# corr_data_set_orange_0.to_csv('datasets/dimensionality_reduction/high_correlation_filter/data-set-correlation-orange-0.csv',
#                                    index=False)
# corr_data_set_orange_1 = data_set_orange.drop(["Srate"], axis=1)
# corr_data_set_orange_1.to_csv('datasets/dimensionality_reduction/high_correlation_filter/data-set-correlation-orange-1.csv',
#                                    index=False)
# corr_data_set_orange_2 = data_set_orange.drop(["Srate", "Weight"], axis=1)
# corr_data_set_orange_2.to_csv('datasets/dimensionality_reduction/high_correlation_filter/data-set-correlation-orange-2.csv',
#                                    index=False)


# Backward Feature Elimination =========================================================================================
# [ True  True False False False False False  True  True False False  True False  True  True  True  True  True]
# [1 1 2 8 9 3 4 1 1 5 7 1 6 1 1 1 1 1]

# array = ["flow_duration", "Header_Length", "Rate", "rst_flag_number", "psh_flag_number", "ack_count", "fin_count",
#          "urg_count", "rst_count", "HTTP", "DNS", "TCP", "ARP", "Min", "Number", "Radius", "Covariance", "Weight"]

# [ True  True  True  True False False False  True False False]
# [1 1 1 1 6 3 5 1 2 4]

# array = ["flow_duration", "Header_Length", "urg_count", "rst_count", "TCP", "Min", "Number", "Radius", "Covariance", "Weight"]

# array = ["psh_flag_number", "rst_flag_number", "DNS", "ARP", "HTTP", "fin_count",
#          "ack_count", "Rate", "TCP"]
#
# data_set = pd.read_csv(
#     'datasets/dimensionality_reduction/high_correlation_filter/data-set-correlation-orange-1.csv')
#
# new_data_set = data_set.drop(array, axis=1)
#
# new_data_set.to_csv('datasets/dimensionality_reduction/backward_feature_elimination/data-set-correlation-orange-1-backward.csv',
#                                    index=False)

# features = new_data_set.iloc[:, :-1].values
# labels = pd.factorize(new_data_set.iloc[:, -1].values)[0]
#
# features_train, features_test, labels_train, labels_test = train_test_split(
#     features, labels, test_size=0.2, random_state=1234
# )
#
# skit_model = RandomForestClassifier()
# # rfe = RFE(skit_model, n_features_to_select=5, step=1)
# # rfe.fit(features_train, labels_train)
# skit_model.fit(features_train, labels_train)
#
#
# # print("Backward Feature Elimination")
# # print('\n\nFEATUERS SELECTED\n\n')
# # print(rfe.support_)
# #
# # print('\n\nRANKING OF FEATURES\n\n')
# # print(rfe.ranking_)
#
# predictions = skit_model.predict(features_test)
# loss_function = metrics.log_loss(labels_test, predictions)
# f1_score = metrics.f1_score(labels_test, predictions)
# accuracy_rate = metrics.accuracy_score(labels_test, predictions)
# precision = metrics.precision_score(labels_test, predictions)
# recall = metrics.recall_score(labels_test, predictions)
# roc_auc = metrics.roc_auc_score(labels_test, predictions)
#
# print(f"loss_function = {loss_function}")
# print(f"f1_score = {f1_score}")
# print(f"accuracy_rate = {accuracy_rate}")
# print(f"precision = {precision}")
# print(f"recall = {recall}")
# print(f"roc_auc = {roc_auc}")
#
# # array = ["psh_flag_number", "rst_flag_number", "DNS", "ARP", "HTTP", "fin_count", "ack_count", "Rate", "TCP"]
# # loss_function = 0.0006878428539365429
# # f1_score = 0.999954944807389
# # accuracy_rate = 0.9999809163947253
# # precision = 0.999909893674536
# # recall = 1.0
# # roc_auc = 0.9999878946349021
#
# # array = ["psh_flag_number", "rst_flag_number", "DNS", "ARP", "HTTP", "fin_count", "ack_count", "Rate", "TCP", "Number"]
# # loss_function = 0.0013756857078728638
# # f1_score = 0.999909893674536
# # accuracy_rate = 0.9999618327894506
# # precision = 0.9998198035859086
# # recall = 1.0
# # roc_auc = 0.9999757892698045