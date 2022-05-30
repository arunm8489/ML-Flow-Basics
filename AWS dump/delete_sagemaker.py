from utils.sagemaker_integration import remove_deployed_model
from utils.common import read_config

if __name__=="__main__":
    config = read_config("./config.yaml")
    response = remove_deployed_model(config)
    print(response)

