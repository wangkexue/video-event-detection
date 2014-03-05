%function train_classifer
clear; clc;
addpath('~/Downloads/libsvm-3.17/matlab');
load('~/tracklet_descriptor_code/data/feat_data/kth/labels/histograms_pref_6629_scale_1.5_nClusters_891.mat');
%len = 0;
%{
train_data = zeros(0.5*size(hist{1}), 4*891);
train_label = zeros(size(train_data, 1), 1);
test_data = zeros(0.5*size(hist{1}), 4*891);
test_label = zeros(size(test_data, 1), 1);
%}
train_data = [];
train_label = [];
test_data = [];
test_label = [];

%tr_start = 0;
%te_start = 0;
for i = 1:6
     for j = 1:floor(size(hist{i}, 2)/2)
         %train_label(j+tr_start, :) = tmp;
         %train_label(tr_start+j) = i;
         train_data = [train_data; reshape(hist{i}{j}, 1, 4*891)];
         train_label = [train_label; i];
         %tmp = [];
     end
     %tr_start = tr_start + j;
     for j = floor(0.5*size(hist{i}, 2)+1):size(hist{i}, 2)
         %test_label(te_start+j, :) = tmp;
         %test_label(te_start+j) = i;
         test_data = [test_data; reshape(hist{i}{j}, 1, 4*891)];
         test_label = [test_label; i];
         %tmp = [];
     end
     %te_start = te_start + j;
end

model = cell(6, 1);
for k = 1:6
    model{k} = svmtrain(double(train_label==k), train_data, '-c 1 -b 1');
end
prob = zeros(size(test_data, 1), 6);
for k = 1:6
        [~, ~, p] = svmpredict(double(test_label==k), test_data, model{k}, '-b 1');
        prob(:, k) = p(:, model{k}.Label==1);
end

[~, pred] = max(prob, [], 2);
acc = sum(pred == test_label) ./ numel(test_label)
C = confusionmat(test_label, pred)
%end