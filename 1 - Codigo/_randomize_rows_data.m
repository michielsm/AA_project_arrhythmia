function [data_randomized] = _randomize_rows_data(data)


rows_shuffled = randperm(rows(data));
data_randomized = data(rows_shuffled,:);

endfunction