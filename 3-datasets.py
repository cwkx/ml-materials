# Run each section from the teaching directory, with webserver.py running.
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('teaching.mplstyle')

# Image folders: a batch has axes [batch, channels, height, width].
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_dataset = datasets.ImageFolder('hymenoptera_data/train', train_transform)
test_dataset = datasets.ImageFolder('hymenoptera_data/val', test_transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

x, labels = next(iter(train_loader))
print('Classes:', train_dataset.classes)
print('Images [batch, channels, height, width]:', x.shape, x.dtype)
print('Labels [batch]:', labels.shape, labels.dtype)
print('First label:', train_dataset.classes[labels[0].item()])

# Undo the channel normalisation before displaying the colours.
mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
grid = torchvision.utils.make_grid((x * std + mean).clamp(0, 1))
plt.imsave('image-folder.png', grid.permute(1, 2, 0).numpy())

# Fashion-MNIST: one channel per image; flatten only the image axes.
train_dataset = datasets.FashionMNIST('data', train=True, download=True,
    transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
x, labels = next(iter(train_loader))
print('Fashion images [batch, channels, height, width]:', x.shape)
print('Fashion labels [batch]:', labels.shape)
flat = x.flatten(start_dim=1)
print('Flattened images [batch, features]:', flat.shape)
print('One image [channels, height, width]:', x[0].shape)

grid = torchvision.utils.make_grid(x)
plt.imsave('fashion-mnist.png', grid.permute(1, 2, 0).numpy())

# Spoken digits: arrays from Practical 7, using its fixed training split.
# Files: https://github.com/cwkx/research-data/tree/main/datasets
with np.load('audio.npz', allow_pickle=False) as data:
    keep = data['split'] == 0
    audio = torch.from_numpy(data['waveforms'][keep]).float() / 32767
    labels = torch.from_numpy(data['labels'][keep]).long()
    voices = torch.from_numpy(data['voices'][keep]).long()
    spectrograms = torch.from_numpy(data['spectrograms'][keep]).float().unsqueeze(1)

sample_rate = 8000
print('Speech [recordings, samples]:', audio.shape, audio.dtype)
print('Digit labels [recordings]:', labels.shape, labels.dtype)
print('Voice labels [recordings]:', voices.shape, voices.dtype)
print('Spectrograms [recordings, channel, frequency, time]:', spectrograms.shape)
print('Digit labels 0–8 mean spoken digits 1–9; voice labels are 0 or 1.')

audio_dataset = TensorDataset(audio, labels, voices)
audio_loader = DataLoader(audio_dataset, batch_size=8, shuffle=True)
wave_batch, label_batch, voice_batch = next(iter(audio_loader))
print('Audio batch [batch, samples]:', wave_batch.shape)
print('Label batch [batch]:', label_batch.shape)
print('Voice batch [batch]:', voice_batch.shape)
conv_batch = wave_batch.unsqueeze(1)
print('Conv1d input [batch, channels, samples]:', conv_batch.shape)

i = 0
seconds = torch.arange(audio.shape[1]) / sample_rate
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(seconds.numpy(), audio[i].numpy(), linewidth=0.8)
axes[0].set(title=f'Spoken digit {labels[i].item() + 1}: {tuple(audio[i].shape)}',
            xlabel='Time (seconds)', ylabel='Amplitude', ylim=(-1, 1))
axes[1].grid(False)
axes[1].imshow(spectrograms[i, 0].numpy(), origin='lower', aspect='auto',
               extent=(0, audio.shape[1] / sample_rate, 0, sample_rate / 2000),
               cmap='gray_r')
axes[1].set(title=f'Spectrogram: {tuple(spectrograms[i, 0].shape)}',
            xlabel='Time (seconds)', ylabel='Frequency (kHz)')
fig.savefig('speech.png')
plt.close(fig)

# Multimodal arrays from Practical 9. Load only the signals used here.
with np.load('multimodal.npz', allow_pickle=False) as data:
    bach = torch.from_numpy(data['bach'])
    bach_sample_rate = int(data['bach_sample_rate'].item())
    parrot = torch.from_numpy(data['parrot'])

# Bach: a mono recording, stored as [samples, channels].
print('Bach [samples, channels]:', bach.shape, bach.dtype)
print('Bach sample rate:', bach_sample_rate, 'Hz')
print('Bach duration:', len(bach) / bach_sample_rate, 'seconds')
bach_batch = bach.transpose(0, 1).unsqueeze(0)
print('Bach Conv1d input [batch, channels, samples]:', bach_batch.shape)
clip = bach_batch[:, :, :2048]
print('First 2048 samples, preserving batch and channel axes:', clip.shape)

seconds = torch.arange(len(bach)) / bach_sample_rate
start, end = round(2 * bach_sample_rate), round(2.05 * bach_sample_rate)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(seconds.numpy(), bach[:, 0].numpy(), linewidth=0.5)
axes[0].set(title=f'Bach: {tuple(bach.shape)}', xlabel='Time (seconds)', ylabel='Amplitude')
axes[1].plot(seconds[start:end].numpy(), bach[start:end, 0].numpy(), linewidth=0.8)
axes[1].set(title='50 ms of individual samples', xlabel='Time (seconds)', ylabel='Amplitude')
fig.savefig('bach.png')
plt.close(fig)

# Parrot: move the channel axis, add a batch axis, then resize the spatial axes.
print('Parrot [height, width, channels]:', parrot.shape, parrot.dtype)
image = parrot.permute(2, 0, 1).float() / 255
print('Parrot [channels, height, width]:', image.shape, image.dtype)
image_batch = image.unsqueeze(0)
print('Parrot [batch, channels, height, width]:', image_batch.shape)
thumbnail = F.interpolate(image_batch, size=(224, 224), mode='bilinear',
                          align_corners=False, antialias=True)
print('Resized batch:', thumbnail.shape)
print('Red channel [height, width]:', image[0].shape)
pixels = parrot.reshape(-1, 3)
print('Pixels [pixel, RGB]:', pixels.shape)
# Use permute to reorder axes; reshape alone does not move the channel axis.

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(parrot.numpy())
axes[0].set_title(f'Original: {tuple(parrot.shape)}')
display_image = thumbnail[0].permute(1, 2, 0)
print('Matplotlib expects [height, width, channels]:', display_image.shape)
axes[1].imshow(display_image.numpy())
axes[1].set_title(f'Resized: {tuple(display_image.shape)}')
for ax in axes:
    ax.set_axis_off()
fig.savefig('parrot.png')
plt.close(fig)
