import subprocess
from Taxi_v3 import TaxiEnvironment

try:
    subprocess.run(['python', 'Taxi_v3.py'], check=True)
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")

taxi_env = TaxiEnvironment()
taxi_env.Train()
taxi_env.Evaluate()