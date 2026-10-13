# In another terminal: python webserver.py
# Open port 8000 on the machine running the server in your browser.
# Run these examples from the teaching directory.
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Save figures on a server without a display.
import matplotlib.pyplot as plt

plt.style.use('teaching.mplstyle')
rng = np.random.default_rng(0)

# Images: the browser discovers each new file automatically.
fig, ax = plt.subplots()
ax.imshow(rng.random((32, 32, 3)))
ax.set_title('Random image')
ax.set_axis_off()
fig.savefig('image.png')
plt.close(fig)

# Histogram.
fig, ax = plt.subplots()
ax.hist(rng.standard_normal(1000), bins=35)
ax.set(title='Normal samples', xlabel='Value', ylabel='Count')
fig.savefig('histogram.png')
plt.close(fig)

# Scatter plot.
fig, ax = plt.subplots()
ax.scatter(*rng.standard_normal((2, 100)), alpha=0.7)
ax.set(title='Random points', xlabel='x', ylabel='y')
fig.savefig('scatter.png')
plt.close(fig)

# Live updates: overwrite the same file to update the same card.
# These are example curves; the later lectures plot real training metrics.
epochs = np.arange(1, 31)
train_loss = np.exp(-epochs / 9) + 0.05 * rng.random(30)
test_loss = np.exp(-epochs / 11) + 0.12 + 0.05 * rng.random(30)
for step in range(1, 31):
    fig, ax = plt.subplots()
    ax.plot(epochs[:step], train_loss[:step], label='Train')
    ax.plot(epochs[:step], test_loss[:step], label='Test')
    ax.set(title='Example learning curves', xlabel='Epoch', ylabel='Loss',
           xlim=(1, 30), ylim=(0, 1.2))
    ax.legend()
    fig.savefig('plots.png')
    plt.close(fig)
    time.sleep(0.2)
