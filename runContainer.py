# Matias D. Molina - molinamatiasd@gmail.com - linkedin.com/in/matiasmolina

import docker
from settings import port_number

client = docker.from_env()

# Build the image
client.images.build(path=".", tag="waste-classifier")

# Run the container
container = client.containers.run(
    "waste-clasiffier",
    detach=True,
    ports={f"{port_number}/tcp": port_number}
)

print(f"Container {container.id} is running")

