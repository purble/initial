function ans = my_attempt1(n_hid, lr_rbm, lr_classification, n_iterations)
	% load data
	a4_init();
	% first train the rbm
	fprintf(': %d\n', lr_classification);
	global report_calls_to_sample_bernoulli;
	report_calls_to_sample_bernoulli = false;
	global data_sets;
	rbm_w = optimize([n_hid, 256], ...
					 @(rbm_w, data) cd1(rbm_w, data.inputs), ... % discard labels
					 data_sets.training, ...
					 lr_rbm, ...
					 n_iterations);
	% rmb_w is now a weight matrix of <n_hid> x <number of visible units, i.e. 256>
	show_rbm(rbm_w);
	input_to_hid = rbm_w; % n_hid x n_vis
	% calculate the hidden layer representation of the labeled data
	hidden_representation = logistic(input_to_hid * data_sets.training.inputs); % n_hid x n_data
	% train hid to class
	data_2.inputs = hidden_representation;
	data_2.targets = data_sets.training.targets;
	hid_to_class = optimize([10, n_hid], @(model, data) classification_phi_gradient(model, data), data_2, lr_classification, n_iterations);
	% report results
	for data_details = reshape({'training', data_sets.training, 'validation', data_sets.validation, 'test', data_sets.test}, [2, 3]),
		data_name = data_details{1};
		data = data_details{2};
		hid_input = input_to_hid * data.inputs; % n_hid x n_data
		hid_output = logistic(hid_input); % n_hid x n_data
		class_input = hid_to_class * hid_output; % 10 x n_data
		class_normalizer = log_sum_exp_over_rows(class_input);
		log_class_prob = class_input - repmat(class_normalizer, size(class_input, 1), 1);
		error_Rate = mean(double(argmax_over_rows(class_input) ~= argmax_over_rows(data.targets)));
		loss = -mean(sum(log_class_prob .* data.targets, 1));
		fprintf('For the %s data, the classification cross-entropy loss is %f, and the classification error rate (i.e. the misclassification rate) is %f\n', data_name, loss, error_Rate);
		if (strcmp(data_name,'validation')==1) ans = loss; end;
	end
	fprintf('\n');
	report_calls_to_sample_bernoulli = true;
end

function indices = argmax_over_rows(matrix)
	[dump, indices] = max(matrix);
end

function ret = log_sum_exp_over_rows(matrix)
	maxs_small = max(matrix, [], 1);
	maxs_big = repmat(maxs_small, size(matrix,1), 1);
	ret = log(sum(exp(matrix - maxs_big))) + maxs_small;
end

function d_phi_by_d_input_to_class = classification_phi_gradient(input_to_class, data)
	% This is about a very simple model: there is an input layer, and a softmax output layer. There are no hidden layers, and no biases.
	% This returns the gradient of phi (a.k.a. negative the loss) for the <input_to_class> matrix.
	% <input_to_class> is a matrix of size <number of classes> by <number of input units>.
	% <data> has fields .inputs (matrix of size <number of input units> by <number of data cases>) and .targets (matrix of size <number of classes> by <number of data cases>).
	% first forward pass
	class_input = input_to_class * data.inputs; % (10 x n_vis) x (n_vis x n_data) = 10 x n_data
	class_normalizer =  log_sum_exp_over_rows(class_input);
	log_class_prob = class_input - repmat(class_normalizer, size(class_input, 1), 1);
	class_prob = exp(log_class_prob);
	d_loss_by_d_class_input = -(data.targets - class_prob) ./ size(data.inputs, 2); % 10 x n_data
	d_loss_by_d_input_to_class = d_loss_by_d_class_input * data.inputs'; % 10 x 10
	d_phi_by_d_input_to_class = -d_loss_by_d_input_to_class; 
end



function ret = cd1(rbm_w, visible_data)
	% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
	% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
	% The returned value is the gradient approximation produced by CD-1. It's of the same type shape as <rbm_w>

	visible_data  = sample_bernoulli(visible_data); % as per instructino in question 8

	hu1 = logistic(rbm_w * visible_data); % n_hid x n_data
	hu1 = sample_bernoulli(hu1); % n_hid x n_data
	gd1 = (hu1) * (visible_data)' / size(visible_data, 2); % n_hid x n_vis
	rv2 = logistic(rbm_w' * hu1); % n_vis x n_data
	rv2 = sample_bernoulli(rv2); % "Reconstruction" for the visible units
	hu2 = logistic(rbm_w * rv2); % n_hid x n_data
	% For question 6 uncomment the following ; generally we do not do so because If you go through the math (either on your own on with your fellow students on the forum), you'll see that sampling the hidden state that results from the "reconstruction" visible state is useless: it does not change the expected value of the gradient estimate that CD-1 produces; it only increases its variance. More variance means that we have to use a smaller learning rate, and that means that it'll learn more slowly; in other words, we don't want more variance, especially if it doesn't give us anything pleasant to compensate for that slower learning.
	% hu2 = sample_bernoulli(hu2);
	gd2 = (hu2 * rv2') / size(rv2, 2);
	ret = gd1 - gd2;
end

function ret = logistic(input)
	ret = 1 ./ (1 + exp(-input));
end

function model = optimize(model_shape, gradient_function, training_data, learning_rate, n_iterations)
	% This trains a model that's defined by a single matrix of weights.
	% <model_shape> is the shape of the array of weights.
	% <gradient_function> is a function that takes parameters <model> and <data> and returns the gradient (or approximate gradient in the case of CD-1) of the function that we are maximizing. Note the contrast with the loss function as in PA3, which we were minimizing. The returned gradient is an array of the same slope as the provided <model> parameter.
	% This uses mini-batches of size 100, momentum of 0.9, no weight decay, and no early stopping.
	% This returns the matrix of weights of the trained model.
	model = (a4_rand(model_shape, prod(model_shape)) * 2 - 1) * 0.1;
	momentum_speed = zeros(model_shape);
	mini_batch_size = 100;
	start_of_next_mini_batch = 1;
	for iteration_number = 1:n_iterations,
		mini_batch = extract_mini_batch(training_data, start_of_next_mini_batch, mini_batch_size);
		start_of_next_mini_batch = mod(start_of_next_mini_batch + mini_batch_size, size(training_data.inputs, 2));
		gradient = gradient_function(model, mini_batch);
		momentum_speed = 0.9 * momentum_speed + gradient;
		model = model + momentum_speed * learning_rate;
	end
end

function a4_init()
	global randomness_source;
	load a4_randomness_source;

	global data_sets;
	temp = load('data_set'); % same as in PA3
	data_sets = temp.data;

	% ...
end

function mini_batch = extract_mini_batch(data_set, start_i, n_cases)
	mini_batch.inputs = data_set.inputs(:, start_i : start_i + n_cases - 1);
	mini_batch.targets = data_set.targets(:, start_i : start_i + n_cases - 1);
end

function ret = a4_rand(requested_size, seed)
	global randomness_source
    start_i = mod(round(seed), round(size(randomness_source, 2) / 10)) + 1;

	if start_i + prod(requested_size) >= size(randomness_source, 2) + 1,
		error('a4_rand failed to generate an array of that size (too big)')
	end
	ret = reshape(randomness_source(start_i : start_i+prod(requested_size)-1), requested_size);
end

function binary = sample_bernoulli(probabilities)
	global report_calls_to_sample_bernoulli
	if report_calls_to_sample_bernoulli,
		fprintf('sample_bernoulli() was called with a matrix of size %d by %d. ', size(probabilities, 1), size(probabilities, 2));
	end
	seed = sum(probabilities(:));
	binary = +(probabilities > a4_rand(size(probabilities), seed)); % the "+" is to avoid the "logical" data type, which just confuses things.
end

function show_rbm(rbm_w) % n_hid x n_vis
	n_hid = size(rbm_w, 1);
	n_rows = ceil(sqrt(n_hid));
	blank_lines = 4;
	distance = 16 + blank_lines;
	to_show = zeros([n_rows * distance + blank_lines, n_rows * distance + blank_lines]);
	for i = 0:n_hid-1,
		row_i = floor(i / n_rows);
		col_i = mod(i, n_rows);
		pixels = reshape(rbm_w(i+1, :), [16, 16]).';
		row_base = row_i*distance + blank_lines;
		col_base = col_i*distance + blank_lines;
		to_show(row_base+1:row_base+16, col_base+1:col_base+16) = pixels;
	end
	extreme = max(abs(to_show(:)));
	try
		imagesc(to_show, [-extreme, extreme]);
		title('hidden units of the RBM');
	catch err
		fprintf('Failed to display the RBM. You are definitely missing out an interesting picture.\n');
	end
end