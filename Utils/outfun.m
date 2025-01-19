
% Define an output function to log the parameters at each iteration
function stop = outfun(x, ~, state)
    global history;
    stop = false; % Continue iterations
    if strcmp(state, 'iter') % Log only during iterations
        history = [history; x]; % Append current parameters to history
    end
end