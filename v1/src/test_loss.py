"""Tests for the the contrastive loss function."""

from loss import ContrastiveLoss
import torch


def test_smoke_test_no_duplicates():
    """Simple smoke test without duplicates.

    This test ensures that we can use a smaller set of candidate embeddings, and
    not trigger the remove duplicates stream successfully.
    """
    loss = ContrastiveLoss(
        remove_duplicates=False, temperature=1.0, learnable_temperature=False
    )
    roi_embeddings = torch.rand(10, 256)
    candidate_embeddings = torch.rand(5, 256)
    y_true = {"class_indices": torch.randint(0, 5, (10,))}
    loss_value, y_pred = loss(roi_embeddings, candidate_embeddings, y_true)
    assert loss_value.shape == ()
    assert y_pred.shape == (10,)


def test_smoke_test_with_duplicates():
    """Simple smoke test with duplicates.

    This test ensures that we can use a larger set of candidate embeddings,
    matching the batch size, and trigger the remove duplicates stream
    successfully.
    """
    loss = ContrastiveLoss(
        remove_duplicates=True, temperature=1.0, learnable_temperature=False
    )
    roi_embeddings = torch.rand(10, 256)
    candidate_embeddings = torch.rand(10, 256)
    y_true = {"class_indices": torch.randint(0, 10, (10,))}
    loss_value, y_pred = loss(roi_embeddings, candidate_embeddings, y_true)
    assert loss_value.shape == ()
    assert y_pred.shape == (10,)


def test_temperature():
    """Simple test to check the temperature parameter.

    This test ensures that we can use a temperature parameter, and that it
    changes the output of the loss function.
    """
    loss = ContrastiveLoss(
        remove_duplicates=True, temperature=2.0, learnable_temperature=False
    )
    roi_embeddings = torch.rand(10, 256)
    candidate_embeddings = torch.rand(10, 256)
    y_true = {"class_indices": torch.randint(0, 10, (10,))}
    loss_value_high_temp, y_pred_high_temp = loss(
        roi_embeddings, candidate_embeddings, y_true
    )

    loss = ContrastiveLoss(
        remove_duplicates=True, temperature=0.5, learnable_temperature=False
    )
    roi_embeddings = torch.rand(10, 256)
    candidate_embeddings = torch.rand(10, 256)
    y_true = {"class_indices": torch.randint(0, 10, (10,))}
    loss_value_low_temp, y_pred_low_temp = loss(
        roi_embeddings, candidate_embeddings, y_true
    )

    assert loss_value_high_temp != loss_value_low_temp
    assert y_pred_high_temp.shape == (10,)
    assert y_pred_low_temp.shape == (10,)


def test_learnable_temperature():
    """Test to check the learnable temperature parameter.

    This test ensures that the temperature parameter is updated during training.
    """
    loss = ContrastiveLoss(
        remove_duplicates=True, temperature=1.0, learnable_temperature=True
    )
    optimizer = torch.optim.SGD(loss.parameters(), lr=0.01)
    roi_embeddings = torch.rand(10, 256)
    candidate_embeddings = torch.rand(10, 256)
    y_true = {"class_indices": torch.randint(0, 10, (10,))}

    initial_temperature = loss.temperature.item()
    for _ in range(10):
        optimizer.zero_grad()
        loss_value, _ = loss(roi_embeddings, candidate_embeddings, y_true)
        loss_value.backward()
        optimizer.step()

    updated_temperature = loss.temperature.item()
    assert initial_temperature != updated_temperature


def test_no_duplicates_with_learnable_temperature():
    """Test with no duplicates and learnable temperature.

    This test ensures that the loss function works correctly when there are no
    duplicates and the temperature parameter is learnable.
    """
    loss = ContrastiveLoss(
        remove_duplicates=False, temperature=1.0, learnable_temperature=True
    )
    optimizer = torch.optim.SGD(loss.parameters(), lr=0.01)
    roi_embeddings = torch.rand(10, 256)
    candidate_embeddings = torch.rand(5, 256)
    y_true = {"class_indices": torch.randint(0, 5, (10,))}

    for _ in range(10):
        optimizer.zero_grad()
        loss_value, y_pred = loss(roi_embeddings, candidate_embeddings, y_true)
        loss_value.backward()
        optimizer.step()

    assert loss_value.shape == ()
    assert y_pred.shape == (10,)


def test_invalid_shapes():
    """Test to check invalid input shapes.

    This test ensures that the loss function raises an assertion error when the
    input shapes are invalid.
    """
    loss = ContrastiveLoss(
        remove_duplicates=True, temperature=1.0, learnable_temperature=False
    )
    roi_embeddings = torch.rand(10, 256)
    candidate_embeddings = torch.rand(5, 128)  # Invalid shape
    y_true = {"class_indices": torch.randint(0, 5, (10,))}

    try:
        loss(roi_embeddings, candidate_embeddings, y_true)
    except AssertionError:
        pass
    else:
        assert False, "Expected an assertion error due to invalid input shapes."


def test_contrastive_loss_value_no_duplicates():
    """Test to ensure the loss is calculated accurately without duplicates."""
    loss = ContrastiveLoss(
        remove_duplicates=False, temperature=1, learnable_temperature=False
    )
    # Here, the cosine similarity between the positive pair is perfect and the
    # negative pair is perpendicular ([0, 1] and [1, 0]).
    roi_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    candidate_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y_true = {"class_indices": torch.tensor([0, 1])}
    loss_value, y_pred = loss(roi_embeddings, candidate_embeddings, y_true)
    expected_loss_value = torch.nn.functional.cross_entropy(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    assert torch.isclose(
        loss_value, expected_loss_value
    ), f"Expected {expected_loss_value}, but got {loss_value}"
    assert torch.isclose(y_pred, torch.tensor([0, 1])).all()


def test_contrastive_loss_value_with_duplicates():
    """Test to ensure the loss value is calculated accurately with duplicates."""
    loss = ContrastiveLoss(
        remove_duplicates=True, temperature=1, learnable_temperature=False
    )
    roi_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    candidate_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    # Here, we denote the third candidate as a duplicate of the first.
    y_true = {"class_indices": torch.tensor([0, 1, 0])}
    loss_value, y_pred = loss(roi_embeddings, candidate_embeddings, y_true)
    # Because the third candidate is a duplicate of the first, the loss should be
    # be the same as if that candidate never existed.
    expected_loss_value = torch.nn.functional.cross_entropy(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
    )

    assert torch.isclose(
        loss_value, expected_loss_value
    ), f"Expected {expected_loss_value}, but got {loss_value}"
    assert torch.isclose(y_pred, torch.tensor([0, 1, 0])).all()
