%fnorm Calculates the aucscore.
% usage auc = aucscore(y, ypred, plot)
% y is the actual values and ypred the predicted ones.
% if plot is true a graph with auc will be ploted. default is 0
% positive labels are 1 and negative -1.
function auc = auc_compute(y, y_pred, plot)
if (~exist('plot', 'var') || isempty(plot))
plot = 0;
end

if (size(y, 2) ~= 1)
    y = y';
    y_pred = y_pred';
end

[~,ind] = sort(y_pred,'descend'); 
roc_y = y(ind);
stack_x = cumsum(roc_y == -1)/sum(roc_y == -1);
stack_y = cumsum(roc_y == 1)/sum(roc_y == 1);
auc = sum((stack_x(2:length(roc_y),1)-stack_x(1:length(roc_y)-1,1)).*stack_y(2:length(roc_y),1));


#{
if (plot)
    plot(stack_x, stack_y);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(['ROC curve of (AUC = ' num2str(auc) ' )']);
end
#}


end
