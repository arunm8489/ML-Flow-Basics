from click import command
import mlflow.sagemaker as mfs
import boto3
import os
import json 
import subprocess


def upload(s3_bucket_name, folder_name):
    try:
        print('Uploading....')
        # command = f"aws s3 sync {folder_name} s3://{s3_bucket_name}"
        output = subprocess.run(["aws","s3","sync","{}".format(folder_name), "s3://{}".format(s3_bucket_name)],
        stdout=subprocess.PIPE,encoding='utf-8',shell=True)
        # output = subprocess.run([command],stdout=subprocess.PIPE,encoding='utf-8',shell=True)
        print('Saved to bucket')
        return f"Done uploading. {output.stdout}"
    except Exception as e:
        return f"Error occured during uploading {e}"


def switch_model_aws_sagemaker(config):
    """
    change model in aws sagemaker
    """
    try:
        app_name = config['params']['app_name']
        execution_arn = config['params']['execution_role_arn']
        region = config['params']['region']
        s3_bucket_name = config['params']['s3_bucket_name']
        experiment_id = config['params']['experiment_id']
        run_id = config['params']['run_id']
        model_name = config['params']['model_name']
        image_ecr_uri = config['params']['image_ecr_uri']
        model_uri = f's3://{s3_bucket_name}/{experiment_id}/{run_id}/artifacts/{model_name}'


        response = mfs.deploy(app_name=app_name,
                              model_uri=model_uri,
                              execution_role_arn=execution_arn,
                              region_name=region,
                              image_url=image_ecr_uri,
                              mode= mfs.DEPLOYMENT_MODE_REPLACE)

        
        return f"Model switched: {response}"

    except Exception as e:
        return f"Error in Model switching: {e.__str__()}"
    
def remove_deployed_model(config=None):

    try:
        app_name = config['params']['app_name']
        region = config['params']['region']

        mfs.delete(app_name=app_name, region_name=region)
        return "Endpoint succesfully deleted: {app_name}"

    except Exception as e:
        return "Error while deleting endpoint: {e.__str__()}"


def deploy_aws_sagemaker(config):
    try:
        app_name = config['params']['app_name']
        execution_arn = config['params']['execution_role_arn']
        region = config['params']['region']
        s3_bucket_name = config['params']['s3_bucket_name']
        experiment_id = config['params']['experiment_id']
        run_id = config['params']['run_id']
        model_name = config['params']['model_name']
        image_ecr_uri = config['params']['image_ecr_uri']
        model_uri = f's3://{s3_bucket_name}/{experiment_id}/{run_id}/artifacts/{model_name}'


        response = mfs.deploy(app_name=app_name,
                              model_uri=model_uri,
                              execution_role_arn=execution_arn,
                              region_name=region,
                              image_url=image_ecr_uri,
                              mode = mfs.DEPLOYMENT_MODE_CREATE)

        
        return f"Deployment sucessfull"

    except Exception as e:
        return f"Error occured while deployment: {e.__str__()}"
    
def remove_deployed_model(config=None):

    try:
        app_name = config['params']['app_name']
        region = config['params']['region']

        mfs.delete(app_name=app_name, region_name=region)
        return "Endpoint succesfully deleted: {app_name}"

    except Exception as e:
        return "Error while deleting endpoint: {e.__str__()}"


def query(input_json, config=None):
    try:
        app_name = config['params']['app_name']
        region = config['params']['region']
        client = boto3.session.Session().client("sagemaker-runtime",region)
        response = client.invoke_endpoint(
            EndpointName= app_name,
            Body= input_json,
            ContentType= 'application/json'
        )
        return json.loads(response['Body'].read().decode("ascii"))
    except Exception as e:
        return f"Error Occurred While Prediction : {e.__str__()}"
    


  

    







