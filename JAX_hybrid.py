import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Tuple

class ResNetBlock(nn.Module):
    channels: int
    down_sample: bool = False

    @nn.compact
    def __call__(self, x):
        strides = (2, 1) if self.down_sample else (1, 1)
        residual = x

        x = nn.Conv(self.channels, (3, 3), strides=strides[0], padding='SAME',
                    kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.relu(x)

        x = nn.Conv(self.channels, (3, 3), strides=strides[1], padding='SAME',
                    kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=False)(x)

        if self.down_sample:
            residual = nn.Conv(self.channels, (1, 1), strides=2, padding='SAME',
                               kernel_init=nn.initializers.kaiming_normal())(residual)
            residual = nn.BatchNorm(use_running_average=False)(residual)

        x += residual
        return nn.relu(x)

class ResNet18Encoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding='SAME',
                    kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2), padding='SAME')

        skip1 = ResNetBlock(64)(x)
        x = ResNetBlock(64)(skip1)

        skip2 = ResNetBlock(128, down_sample=True)(x)
        x = ResNetBlock(128)(skip2)

        skip3 = ResNetBlock(256, down_sample=True)(x)
        x = ResNetBlock(256)(skip3)

        skip4 = ResNetBlock(512, down_sample=True)(x)
        x = ResNetBlock(512)(skip4)

        return skip1, skip2, skip3, skip4, x

class DecoderBlock(nn.Module):
    filters: int

    @nn.compact
    def __call__(self, x, skip):
        x = nn.ConvTranspose(self.filters, (2, 2), strides=(2, 2), padding='SAME')(x)
        skip = nn.ConvTranspose(self.filters, (2, 2), strides=(2, 2), padding='SAME')(skip)
        x = jnp.concatenate([x, skip], axis=-1)

        x = nn.Conv(self.filters, (3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.relu(x)

        x = nn.Conv(self.filters, (3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.relu(x)

        return x

class FinalUpsampleBlock(nn.Module):
    filters: int = 64
    num_classes: int = 2

    @nn.compact
    def __call__(self, x):
        x = nn.ConvTranspose(self.filters, (2, 2), strides=(2, 2), padding='SAME')(x)
        x = nn.Conv(self.filters, (3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.relu(x)

        x = nn.Conv(self.num_classes, (1, 1), padding='SAME')(x)
        return nn.sigmoid(x) if self.num_classes == 1 else nn.softmax(x)

class UNetResNet18(nn.Module):
    input_shape: Tuple[int]
    num_classes: int = 2

    @nn.compact
    def __call__(self, x):
        skip1, skip2, skip3, skip4, encoded = ResNet18Encoder()(x)

        d1 = DecoderBlock(512)(encoded, skip4)
        d2 = DecoderBlock(256)(d1, skip3)
        d3 = DecoderBlock(128)(d2, skip2)
        d4 = DecoderBlock(64)(d3, skip1)

        out = FinalUpsampleBlock(64, self.num_classes)(d4)
        return out



import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X_train, X_val, y_train, y_val = train_test_split(Reconstructed_sigma1, tumor_masks, test_size=0.2, random_state=42)

num_classes = 2
y_train = jax.nn.one_hot(y_train, num_classes)
y_val = jax.nn.one_hot(y_val, num_classes)

class TrainState(train_state.TrainState):
    pass

def compute_metrics(logits, labels):
    loss = optax.softmax_cross_entropy(logits, labels).mean()
    accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
    return {'loss': loss, 'accuracy': accuracy}

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = model.apply({'params': params}, batch['image'])
        loss = optax.softmax_cross_entropy(logits, batch['label']).mean()
        return loss, logits
    grads, logits = jax.grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch['label'])
    return state, metrics

@jax.jit
def eval_step(params, batch):
    logits = model.apply({'params': params}, batch['image'])
    return compute_metrics(logits, batch['label'])

def train_model(state, train_ds, val_ds, epochs, batch_size):
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    for epoch in range(epochs):
        batch_metrics = []
        for i in range(0, len(train_ds['image']), batch_size):
            batch = {
                'image': train_ds['image'][i:i+batch_size],
                'label': train_ds['label'][i:i+batch_size]
            }
            state, metrics = train_step(state, batch)
            batch_metrics.append(metrics)
        train_metrics = {
            k: np.mean([m[k] for m in batch_metrics]) for k in batch_metrics[0]
        }
        val_metrics = eval_step(state.params, val_ds)
        print(f"Epoch {epoch+1} | Train Acc: {train_metrics['accuracy']:.3f} | Val Acc: {val_metrics['accuracy']:.3f}")
        for k in train_metrics:
            history[k].append(train_metrics[k])
            history[f"val_{k}"].append(val_metrics[k])
    return state, history

rng = jax.random.PRNGKey(0)
variables = model.init(rng, X_train[:1])
params = variables['params']
tx = optax.adam(learning_rate=1e-3)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

train_ds = {'image': X_train, 'label': y_train}
val_ds = {'image': X_val, 'label': y_val}

state, history = train_model(state, train_ds, val_ds, epochs=20, batch_size=16)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
