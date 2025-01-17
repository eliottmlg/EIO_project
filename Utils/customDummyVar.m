function dummyMatrix = customDummyVar(inputVector)
    % Check input validity
    if ~isvector(inputVector)
        error('Input must be a vector.');
    end
    
    % Ensure the input vector is a column vector
    inputVector = inputVector(:);
    
    % Get unique categories and their indices
    [uniqueValues, ~, categoryIndices] = unique(inputVector, 'stable');
    
    % Initialize the dummy variable matrix
    numRows = length(inputVector);
    numCategories = length(uniqueValues);
    dummyMatrix = zeros(numRows, numCategories);
    
    % Fill in the dummy variable matrix
    for i = 1:numCategories
        dummyMatrix(:, i) = (categoryIndices == i);
    end
end
