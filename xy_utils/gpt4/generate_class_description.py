#Note: The openai-python library support for Azure OpenAI is in preview.
import sys
pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

import os
import openai
import time
import functools
import signal
import torch

from utils.constants import COCO_PANOPTIC_CLASSES


def run(class_name, openai_version):

    if openai_version == 0:
        openai.api_type = "azure"
        openai.api_base = ""
        openai.api_version = "2023-03-15-preview"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        deployment_id = "gpt4"
    elif openai_version == 1:
        openai.api_type = "azure"
        openai.api_base = ""
        openai.api_version = "2023-03-15-preview"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        deployment_id = "gpt4a"
    elif openai_version == 2:
        openai.api_key = os.getenv("OPENAI_API_KEY_AZURE")
        openai.api_base ='' # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
        openai.api_type = 'azure'
        openai.api_version = '2023-03-15-preview' # this may change in the future
        deployment_id='gpt-4-32k-0314' #This will correspond to the custom name you chose for your deployment when you deployed a model.
    elif openai_version == 3:
        openai.api_base ='' # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
        openai.api_type = 'azure'
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_version = '2023-07-01-preview' # this may change in the future
        deployment_id='gpt-4-32k-0613' #This will correspond to the custom name you chose for your deployment when you deployed a model.
    else:
        print(openai_version)
        assert False


    content = '''
    Describe {} in a long sentence without any word contains its name.
    '''.format(class_name)

    def timeout(seconds, error_message = 'OpenAI call timed out'):
        def decorated(func):
            def _handle_timeout(signum, frame):
                raise TimeoutError(error_message)

            def wrapper(*args, **kwargs):
                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                return result

            return functools.wraps(func)(wrapper)
        return decorated

    @timeout(300)
    def openai_call(deployment_id, prompt):
        try:
            response = openai.ChatCompletion.create(
                engine=deployment_id, 
                max_tokens=1500,
                temperature=0.,
                messages=[{"role":"user", "content":prompt}])
            return response
        except Exception as e:
            if 'triggering Azure OpenAI’s content management policy' in str(e):
                return 'continue'
            else:
                raise

    response = openai_call(deployment_id, content)
    # print(response)
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    coco_classes = [x.replace('-other','').replace('-merged','').replace('-stuff','') for x in COCO_PANOPTIC_CLASSES]
    coco_description = []
    openai_version = 0
    for cnt, class_name in enumerate(coco_classes):
        print(cnt)
        success = False
        while not success:
            print('hhihi')
            try:
                response = run(class_name, openai_version)
                coco_description.append(response)
                success = True
            except:
                traceback.print_exc()
                openai_version += 1
                success = False
    
    torch.save(coco_description, 'coco_description.pt')