from helper_interp import (
    create_model, create_optimizer, train_model,
    setup_data, plot_training_history,
    visualize_network_architecture, visualize_sample_digits, visualize_prediction,
    plot_activations, get_sample_by_digit, get_sample_by_index,
    train_probe, visualize_probe_weights, compare_probes_across_layers
)
import matplotlib.pyplot as plt

train_images, train_labels, test_images, test_labels, train_loader, test_loader = setup_data()

visualize_sample_digits(train_images, train_labels)

visualize_network_architecture()

# TRAINING PARAMETERS: CAN BE MODIFIED
EPOCHS = 20
LEARNING_RATE = 0.001

# Create and train the model
model = create_model()
optimizer = create_optimizer(model, lr=LEARNING_RATE)
history = train_model(model, optimizer, train_loader, test_loader, epochs=EPOCHS)
# Plot training progress
plot_training_history(history)

DIGIT_TO_VIEW = 2  # Try values 0-9

image, label = get_sample_by_digit(test_images, test_labels, DIGIT_TO_VIEW)
visualize_prediction(model, image, label)

DIGIT_TO_EXPLORE = 2  # Change this! (0-9)

image, label = get_sample_by_digit(test_images, test_labels, DIGIT_TO_EXPLORE)
plot_activations(model, image, label)

POSITIVE_DIGITS = [0, 6, 8, 9]  # Digits WITH loops
NEGATIVE_DIGITS = [1, 2, 3, 4, 5, 7]  # Digits WITHOUT loops
LAYER_TO_PROBE = 2  # Which layer to probe (0-4)

probe, accuracy = train_probe(
    model, train_loader, test_loader,
    positive_digits=POSITIVE_DIGITS,
    negative_digits=NEGATIVE_DIGITS,
    layer_num=LAYER_TO_PROBE
)

print(f"\nThe probe can distinguish {POSITIVE_DIGITS} from {NEGATIVE_DIGITS}")
print(f"at layer {LAYER_TO_PROBE} with {accuracy:.1%} accuracy!")

visualize_probe_weights(probe, title=f"Probe Weights for Layer {LAYER_TO_PROBE}")

results = compare_probes_across_layers(
    model, train_loader, test_loader,
    positive_digits=POSITIVE_DIGITS,
    negative_digits=NEGATIVE_DIGITS
)

