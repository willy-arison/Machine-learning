from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

def save_tensorboard_data(list_model, v=None):
    import os
    my_plots = {
        'train_loss': ['Training Loss', 'Training Loss Curve', 'training_loss.png', 'Epochs/Steps', 'loss'],
        'val_loss': ['Validation Loss', 'Validation Loss Over Time', 'val_loss.png',  'Epochs/Steps', 'loss'],
        'val_acc': ['Validation Accuracy', 'Validation Accuracy Over Time', 'val_acc.png',  'Epochs/Steps', 'Accuracy'],
    }

    for key, value in my_plots.items():
        plt.figure(figsize=(10, 5))
        for model_name in list_model:
            # Load TensorBoard logs
            log_dir = f"logs/{model_name}"
            # take the latest version
            version = max([
                int(d.split('_')[-1]) for d in os.listdir(log_dir) if d.startswith("version_")
            ])

            # if you want for specific version
            if v is not None:
                version = v

            event_acc = EventAccumulator(f"logs/{model_name}/version_{version}")
            event_acc.Reload()
            # Extract scalar data
            data = event_acc.Scalars(key)
            steps = [x.step for x in data]
            values = [x.value for x in data]
            plt.plot(steps, values, label=f'{value[0]} with {model_name}')

        plt.xlabel(value[-2])
        plt.ylabel(value[-1])
        plt.title(value[1])
        plt.legend()

        # save path
        save_path = f"plot_{version}/{value[-3]}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == '__main__':
    save_tensorboard_data(['GHF', 'Logistic', 'Tanh', 'ReLU', 'Mish', 'LeakyReLU'], v=0)