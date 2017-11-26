% printing option
more off;

% read files
D_tr = csvread('spambasetrain.csv');
D_ts = csvread('spambasetest.csv');

% construct x and y for training and testing
X_tr = D_tr(:, 1:end - 1);
y_tr = D_tr(:, end);
X_ts = D_ts(:, 1:end - 1);
y_ts = D_ts(:, end);

% number of training / testing samples
n_tr = size(D_tr, 1);
n_ts = size(D_ts, 1);

% add 1 as a feature
X_tr = [ones(n_tr, 1) X_tr];
X_ts = [ones(n_ts, 1) X_ts];
lr = 1; % learning rate

for k = 1:4
    if k ~= 1
        lr = lr * (10 ^ -2);
    end
    % perform gradient descent :: logistic regression
    n_vars = size(X_tr, 2); % number of variables
    w = zeros(n_vars, 1); % initialize parameter w
    tolerance = 1e-2; % tolerance for stopping criteria
 
    iter = 0; % iteration counter
    max_iter = 1000; % maximum iteration
    accuracyTrainList = [];
    accuracyTestList = [];
    while true
        iter = iter + 1; % start iteration
     
        % calculate gradient
        grad = zeros(n_vars, 1); % initialize gradient
        grad = X_tr.'*y_tr- X_tr.' * (exp(X_tr * w) ./ (1 + exp(X_tr * w)));
     
        % take step
        % w_new = w + .....              % take a step using the learning rate
        w_new = w + lr * grad;
        %printf('iter = %d, mean abs gradient = %0.3f\n', iter, mean(abs(grad)));
        %fflush(stdout);
     
        % stopping criteria and perform update if not stopping
        if mean(abs(grad)) < tolerance
            w = w_new;
            break;
        else
            w = w_new;
        end
        % use w for prediction
        predTrain = zeros(n_tr, 1); % initialize prediction vector
        y1prTrain = exp(X_tr * w) ./ (1 + exp(X_tr * w));
        predTrain = y1prTrain >= 0.5;
        countTrain = 0;
        predTest = zeros(n_ts, 1);
        y1prTest = exp(X_ts * w) ./ (1 + exp(X_ts * w));
        predTest = y1prTest >= 0.5;
        countTest = 0;
        for i = 1:n_tr
            if predTrain(i) == y_tr(i)
                countTrain = countTrain + 1;
            end
        end
        AccuracyTrain = countTrain / n_tr
        accuracyTrainList(iter)=AccuracyTrain;
        for i = 1:n_ts
            if predTest(i) == y_ts(i)
                countTest = countTest + 1;
            end
        end
        AccuracyTest = countTest / n_ts
        accuracyTestList(iter)=AccuracyTest;
        if iter >= max_iter
            break;
        end
    end
    figure;
    plot([1:iter], accuracyTrainList, [1:iter], accuracyTestList)
end

