clear all  
close all  
clc;

%% 1. 加载MNIST数据
train_x_file = 'train-images.idx3-ubyte'; % 60000个训练集图片
test_x_file = 't10k-images.idx3-ubyte';   % 10000个测试集图片
train_y_file = 'train-labels.idx1-ubyte'; % 60000个训练集图片对应的数字 
test_y_file = 't10k-labels.idx1-ubyte';   % 10000个测试集图片对应的数字 

fprintf('正在加载MNIST数据...\n');
train_x = decodefile(train_x_file, 'image');  
test_x = decodefile(test_x_file, 'image');  
train_y = decodefile(train_y_file, 'label');  
test_y = decodefile(test_y_file, 'label');  

% 重塑图像数据
train_x_matrix = reshape(train_x, 28, 28, 60000);
train_x_matrix = permute(train_x_matrix, [2 1 3]);
test_x_matrix = reshape(test_x, 28, 28, 10000);
test_x_matrix = permute(test_x_matrix, [2 1 3]);

% 将图像数据转换为二维矩阵（样本×特征）
train_data = double(reshape(train_x_matrix, [], 60000))';
test_data = double(reshape(test_x_matrix, [], 10000))';
train_labels = double(train_y);
test_labels = double(test_y);

fprintf('训练集: %d个样本, 测试集: %d个样本\n', size(train_data,1), size(test_data,1));

%% 2. 任务1: K-means聚类

% 2.1 确定最佳K值 - 肘部法则
fprintf('正在使用肘部法则确定最佳K值...\n');
max_k = 15;
within_cluster_sum = zeros(max_k, 1);

for k = 1:max_k
    [~, ~, sumd] = kmeans(train_data, k, 'MaxIter', 100, 'Replicates', 3);
    within_cluster_sum(k) = sum(sumd);
    fprintf('K=%d, 簇内平方和: %.2f\n', k, within_cluster_sum(k));
end

% 绘制肘部法则图
figure('Position', [100, 100, 1200, 800]);
subplot(2, 3, 1);
plot(1:max_k, within_cluster_sum, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('簇数量 K');
ylabel('簇内平方和');
title('肘部法则 - 确定最佳K值');
grid on;

% 计算斜率变化来确定肘部点
slopes = diff(within_cluster_sum);
slope_ratios = slopes(1:end-1) ./ slopes(2:end);
[~, optimal_k_idx] = max(slope_ratios);
optimal_k = optimal_k_idx + 2; % 因为diff会减少一个元素

hold on;
plot(optimal_k, within_cluster_sum(optimal_k), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
legend('簇内平方和', sprintf('建议K值: %d', optimal_k));

% 2.2 使用轮廓系数评估聚类质量
fprintf('\n正在计算轮廓系数评估聚类质量...\n');
k_values = [5, 8, 10, 12, 15];
silhouette_scores = zeros(length(k_values), 1);

% 使用部分数据计算轮廓系数（计算量较大）
sample_size = 2000;
sample_idx = randperm(size(train_data, 1), sample_size);
sample_data = train_data(sample_idx, :);

for i = 1:length(k_values)
    k = k_values(i);
    [idx, ~] = kmeans(sample_data, k, 'MaxIter', 100);
    silhouette_vals = silhouette(sample_data, idx);
    silhouette_scores(i) = mean(silhouette_vals);
    fprintf('K=%d, 平均轮廓系数: %.4f\n', k, silhouette_scores(i));
end

subplot(2, 3, 2);
bar(k_values, silhouette_scores, 'FaceColor', [0.2, 0.6, 0.8]);
xlabel('簇数量 K');
ylabel('平均轮廓系数');
title('轮廓系数分析');
grid on;

% 2.3 执行最终K-means聚类
final_k = 10; % MNIST有10个数字类别
fprintf('\n执行最终K-means聚类 (K=%d)...\n', final_k);
[idx, centers, sumd, distances] = kmeans(train_data, final_k, ...
    'MaxIter', 200, 'Replicates', 5, 'Display', 'final');

% 计算聚类准确率
accuracy = calculate_clustering_accuracy(idx, train_labels, final_k);
fprintf('K-means聚类准确率: %.2f%%\n', accuracy * 100);

% 2.4 可视化聚类结果
% 显示聚类中心
subplot(2, 3, 3);
montage_array = zeros(28, 28, 1, final_k);
for i = 1:final_k
    center_img = reshape(centers(i, :), 28, 28);
    montage_array(:, :, 1, i) = center_img;
end
montage(montage_array, 'Size', [2, 5]);
title('K-means聚类中心');

% 显示每个聚类的样本分布
subplot(2, 3, 4);
cluster_counts = zeros(final_k, 1);
for i = 1:final_k
    cluster_counts(i) = sum(idx == i);
end
bar(1:final_k, cluster_counts, 'FaceColor', [0.8, 0.4, 0.2]);
xlabel('聚类编号');
ylabel('样本数量');
title('各聚类样本数量分布');
grid on;

% 显示聚类纯度（每个聚类中主要数字的占比）
subplot(2, 3, 5);
purity_scores = zeros(final_k, 1);
for i = 1:final_k
    cluster_indices = (idx == i);
    true_labels_in_cluster = train_labels(cluster_indices);
    if ~isempty(true_labels_in_cluster)
        most_common_label = mode(true_labels_in_cluster);
        purity_scores(i) = sum(true_labels_in_cluster == most_common_label) / length(true_labels_in_cluster);
    end
end
bar(1:final_k, purity_scores * 100, 'FaceColor', [0.4, 0.8, 0.4]);
xlabel('聚类编号');
ylabel('纯度 (%)');
title('各聚类纯度');
ylim([0, 100]);
grid on;

% 显示距离热图
subplot(2, 3, 6);
center_distances = pdist2(centers, centers);
imagesc(center_distances);
colorbar;
title('聚类中心距离热图');
xlabel('聚类编号');
ylabel('聚类编号');

%% 3. 任务2: t-SNE可视化

% 使用PCA先降维到50维，再t-SNE（提高速度和稳定性）
fprintf('使用PCA预处理后进行t-SNE...\n');
[coeff, score_pca, ~] = pca(test_data, 'NumComponents', 50);
test_data_pca = test_data * coeff(:, 1:50);

% 执行t-SNE
fprintf('正在进行t-SNE降维...\n');
Y = tsne(test_data_pca, 'NumDimensions', 2, 'Perplexity', 30, 'Verbose', 1);

% 可视化t-SNE结果
figure('Position', [100, 100, 1400, 600]);

% 子图1: 按真实标签着色
subplot(1, 2, 1);
gscatter(Y(:,1), Y(:,2), test_labels, [], 'o', 6, 'on', 'northeastoutside');
title('t-SNE可视化 - 按真实标签着色');
xlabel('t-SNE 1'); ylabel('t-SNE 2');

% 子图2: 按数字类别显示不同的标记
subplot(1, 2, 2);
colors = lines(10);
for digit = 0:9
    mask = (test_labels == digit);
    scatter(Y(mask, 1), Y(mask, 2), 30, colors(digit+1, :), 'filled', 'Marker', get_marker(digit+1));
    hold on;
end
title('t-SNE可视化 - 不同标记和颜色');
xlabel('t-SNE 1');
ylabel('t-SNE 2');
legend('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '位置', 'northeastoutside');

%% 4. 任务3: SVM分类

% 使用完整的训练测试集划分
X_train = train_data;
y_train = train_labels;
X_test = test_data;
y_test = test_labels;

fprintf('数据集划分: 训练集%d, 测试集%d\n', size(X_train,1), size(X_test,1));

% 4.1 尝试不同的SVM核函数
fprintf('\n比较不同SVM核函数的性能...\n');
kernels = {'linear', 'rbf', 'polynomial'};
accuracies = zeros(length(kernels), 1);
training_times = zeros(length(kernels), 1);

for i = 1:length(kernels)
    kernel = kernels{i};
    fprintf('训练 %s 核SVM...\n', kernel);
    
    tic;
    switch kernel
        case 'linear'
            template = templateSVM('KernelFunction', 'linear', 'Standardize', true);
        case 'rbf'
            template = templateSVM('KernelFunction', 'rbf', 'Standardize', true, 'KernelScale', 'auto');
        case 'polynomial'
            template = templateSVM('KernelFunction', 'polynomial', 'Standardize', true, 'PolynomialOrder', 3);
    end
    
    model = fitcecoc(X_train, y_train, 'Learners', template, 'Coding', 'onevsone');
    training_time = toc;
    training_times(i) = training_time;
    
    y_pred = predict(model, X_test);
    accuracy = sum(y_pred == y_test) / length(y_test);
    accuracies(i) = accuracy;
    
    fprintf('%s 核SVM - 准确率: %.2f%%, 训练时间: %.2f秒\n', ...
        kernel, accuracy * 100, training_time);
end

% 可视化不同核函数的性能比较
figure('Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
bar(accuracies * 100, 'FaceColor', [0.3, 0.5, 0.9]);
set(gca, 'XTickLabel', kernels);
ylabel('准确率 (%)');
title('不同SVM核函数性能比较');
grid on;

subplot(1, 2, 2);
bar(training_times, 'FaceColor', [0.9, 0.5, 0.3]);
set(gca, 'XTickLabel', kernels);
ylabel('训练时间 (秒)');
title('不同SVM核函数训练时间比较');
grid on;

% 4.2 使用最佳核函数进行最终训练
fprintf('\n使用最佳核函数进行最终训练...\n');
[best_accuracy, best_idx] = max(accuracies);
best_kernel = kernels{best_idx};

switch best_kernel
    case 'linear'
        final_template = templateSVM('KernelFunction', 'linear', 'Standardize', true);
    case 'rbf'
        final_template = templateSVM('KernelFunction', 'rbf', 'Standardize', true, 'KernelScale', 'auto');
    case 'polynomial'
        final_template = templateSVM('KernelFunction', 'polynomial', 'Standardize', true, 'PolynomialOrder', 3);
end

final_model = fitcecoc(X_train, y_train, 'Learners', final_template, 'Coding', 'onevsone');
y_test_pred = predict(final_model, X_test);
final_accuracy = sum(y_test_pred == y_test) / length(y_test);

% 4.3 显示详细分类报告
fprintf('\n=== 详细分类报告 ===\n');
class_accuracy = zeros(10, 1);
for digit = 0:9
    mask = (y_test == digit);
    if sum(mask) > 0
        digit_accuracy = sum(y_test_pred(mask) == digit) / sum(mask);
        class_accuracy(digit+1) = digit_accuracy;
        fprintf('数字 %d 的准确率: %.2f%% (%d/%d)\n', ...
            digit, digit_accuracy * 100, sum(y_test_pred(mask) == digit), sum(mask));
    end
end

% 显示混淆矩阵和各类别准确率
figure('Position', [100, 100, 1400, 600]);

subplot(1, 2, 1);
confusionchart(y_test, y_test_pred);
title(sprintf('SVM分类混淆矩阵 (%s核, 准确率: %.2f%%)', best_kernel, final_accuracy * 100));

subplot(1, 2, 2);
bar(0:9, class_accuracy * 100, 'FaceColor', [0.2, 0.7, 0.4]);
xlabel('数字类别');
ylabel('准确率 (%)');
title('各数字类别分类准确率');
ylim([0, 100]);
grid on;

% 添加数值标签
for i = 1:10
    text(i-1, class_accuracy(i)*100 + 2, sprintf('%.1f%%', class_accuracy(i)*100), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

%% 辅助函数
function accuracy = calculate_clustering_accuracy(cluster_labels, true_labels, k)
    % 将聚类标签与真实标签进行最佳匹配
    accuracy = 0;
    for i = 1:k
        cluster_mask = (cluster_labels == i);
        if sum(cluster_mask) > 0
            true_labels_in_cluster = true_labels(cluster_mask);
            most_common_label = mode(true_labels_in_cluster);
            cluster_accuracy = sum(true_labels_in_cluster == most_common_label) / sum(cluster_mask);
            accuracy = accuracy + cluster_accuracy * sum(cluster_mask);
        end
    end
    accuracy = accuracy / length(cluster_labels);
end

function marker = get_marker(index)
    markers = {'o', '+', '*', 'x', 's', 'd', '^', 'v', '>', '<'};
    marker = markers{mod(index-1, length(markers)) + 1};
end