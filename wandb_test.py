import wandb

# Initialize W&B run
wandb.init(
    entity="dgrish1-western-governors-university",
    project="nyc_airbnb_public"
)

# Log a simple metric
wandb.log({"test_accuracy": 0.95})

wandb.finish()
