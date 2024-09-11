import torch


def transform_array(input_array):
    # Ensure input is a PyTorch tensor
    if not isinstance(input_array, torch.Tensor):
        input_array = torch.tensor(input_array)

    # Ensure input is of shape (N, 2) and contains only 0s and 1s
    assert input_array.shape[1] == 2, "Input array must have 2 columns"
    assert torch.all((input_array == 0) | (input_array == 1)), "Input array must contain only 0s and 1s"

    # Create the transformation matrix
    transform_matrix = torch.tensor([
        [1, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

    # Convert input to indices
    indices = input_array[:, 0] * 2 + input_array[:, 1]

    # Use indices to select appropriate rows from the transform matrix
    output = transform_matrix[:, indices].t()

    return output


# Example usage:
input_array = torch.tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

result = transform_array(input_array)
print(result)
