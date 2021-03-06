function plot_auc()
load('dataset_model_neural.mat');

[X,Y,T,AUC] = perfcurve(Y_test_neural, predicted_labels_neural,2); % perfcurve(labels,predicted_label,positive_class);

AUC
plot(X,Y)
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification by Logistic Regression')
