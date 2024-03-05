import toml

import fpgaconvnet.optimiser.cli
import wandb

if __name__ == "__main__":

    with open("examples/sweep/sweep_throughput_unet.toml", "r") as f:
        sweep_config = toml.load(f)

    model_name = sweep_config['model_name']
    sweep_config.pop('model_name')

    sweep_id = wandb.sweep(
        sweep_config, entity="fpgaconvnet", project=f"fpgaconvnet-{model_name}-{sweep_config['metric']['name']}")

    wandb.agent(sweep_id, entity="fpgaconvnet",
                function=fpgaconvnet.optimiser.cli.main)
