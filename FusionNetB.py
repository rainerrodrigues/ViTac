import taichi as ti

ti.init(arch=ti.gpu)

# Input image shape
input_shape = (256, 256)

# Number of channels for each feature map
num_channels = [64, 128, 256, 512, 1024]

# Number of features to extract from each level
num_features = [32, 64, 128, 256, 512]

# Weighted sum weights for each level
weights = [0.2, 0.2, 0.2, 0.2, 0.2]

# Define the CNN
@ti.data_oriented
class CNN:
    def __init__(self, input_shape, num_channels, num_features):
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.num_features = num_features

        # Define the CNN layers
        self.conv1 = ti.Conv2d(num_channels[0], num_channels[1], kernel_size=3, padding=1)
        self.conv2 = ti.Conv2d(num_channels[1], num_channels[2], kernel_size=3, padding=1)
        self.conv3 = ti.Conv2d(num_channels[2], num_channels[3], kernel_size=3, padding=1)
        self.conv4 = ti.Conv2d(num_channels[3], num_channels[4], kernel_size=3, padding=1)

        self.fc1 = ti.Dense(num_channels[4] * (input_shape[0] // 16) * (input_shape[1] // 16), num_features[4])
        self.fc2 = ti.Dense(num_features[4], num_features[3])
        self.fc3 = ti.Dense(num_features[3], num_features[2])
        self.fc4 = ti.Dense(num_features[2], num_features[1])
        self.fc5 = ti.Dense(num_features[1], num_features[0])

    def __call__(self, x):
        # CNN forward pass
        x = ti.functional.relu(self.conv1(x))
        x = ti.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = ti.functional.relu(self.conv2(x))
        x = ti.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = ti.functional.relu(self.conv3(x))
        x = ti.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = ti.functional.relu(self.conv4(x))
        x = x.reshape([-1, self.num_channels[4] * (self.input_shape[0] // 16) * (self.input_shape[1] // 16)])
        x = ti.functional.relu(self.fc1(x))
        x = ti.functional.relu(self.fc2(x))
        x = ti.functional.relu(self.fc3(x))
        x = ti.functional.relu(self.fc4(x))
        x = self.fc5(x)

        return x

# Define the FusionNet-B model
@ti.data_oriented
class FusionNetB:
    def __init__(self, input_shape, num_channels, num_features, weights):
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.num_features = num_features
        self.weights = weights

        # Define the CNN and fusion layers
        self.cnn = CNN(input_shape, num_channels, num_features)
        self.fusion1 = ti.Dense(sum(num_features), num_features[0])
        self.fusion2 = ti.Dense(sum(num_features), num_features[1])
        self.fusion3 = ti.Dense(sum(num_features), num_features[2])
        self.fusion4 = ti.Dense(sum(num_features), num_features[3])
        self.fusion5 = ti.Dense(sum(num_features), num_features[4])

    def __call__(self, x):
        # CNN forward pass
        cnn_output = self.cnn(x)

        # Fusion layers
        fusion_input = ti.concatenate([x, cnn_output], dim=1)
        fusion_output1 = ti.functional.relu(self.fusion1(fusion_input))
        fusion_output2 = ti.functional.relu(self.fusion2(fusion_input))
        fusion_output3 = ti.functional.relu(self.fusion3(fusion_input))
        fusion_output4 = ti.functional.relu(self.fusion4(fusion_input))
        fusion_output5 = ti.functional.relu(self.fusion5(fusion_input))

        # Weighted sum of fusion outputs
        weighted_sum = (
            self.weights[0] * fusion_output1 +
            self.weights[1] * fusion_output2 +
            self.weights[2] * fusion_output3 +
            self.weights[3] * fusion_output4 +
            self.weights[4] * fusion_output5
        )

        return weighted_sum

# Create an instance of FusionNetB
fusion_net = FusionNetB(input_shape, num_channels, num_features, weights)

# Output
input_data = ti.Matrix.field(3, dtype=ti.f32, shape=input_shape)
output = fusion_net(input_data)
