import docker
from docker.errors import NotFound
from docker.types import DeviceRequest
import os
import sys
import platform  # For checking the operating system

# The Docker SDK client
client = docker.from_env()

for image in client.images.list():
    print(image.tags)

# Image name, defaulting to "sg2ada:latest" if not provided as an environment variable
image_name = os.getenv('IMAGE', 'sg2ada:latest')

# The rest of the arguments provided to the script
rest_args = sys.argv[1:]
print(rest_args)
working_dir = r"C:\Users\Aitor\experimentacion\stylegan2-ada-pytorch"
cur_dir = os.getcwd()
os.chdir(working_dir)
try:
    # Inspect the image to see if it exists
    client.images.get(image_name)
    gpu_request = DeviceRequest(device_ids=["0"], capabilities=[['gpu']])
    # Prepare common arguments for container run
    container_kwargs = {
        'image': image_name,
        'command': rest_args,
        'shm_size': '2g',
        'detach': False,
        'auto_remove': False,
        'volumes': {os.getcwd(): {'bind': '/scratch', 'mode': 'rw'}},
        'working_dir': '/scratch',
        'environment': {'HOME': '/scratch'},
        'device_requests': [gpu_request]
    }

    # Add user settings for Unix-like systems only
    if platform.system() != 'Windows':
        container_kwargs['user'] = f"{os.getuid()}:{os.getgid()}"

    # Run the container
    container = client.containers.run(**container_kwargs)
    result = container.wait()
    exit_code = result['StatusCode']
    if exit_code == 0:
        print("Docker command succesful")
    else:
        print("Docker command failed.")
        print(result)

    os.chdir(cur_dir)

except NotFound:
    # Image does not exist
    print(f"Unknown container image: {image_name}")
    sys.exit(1)
except docker.errors.APIError as e:
    # Handle other Docker API errors
    print(f"Error starting container: {e}")
    sys.exit(1)
except docker.errors.ContainerError as e:
    print(f"Container execution error {e}")