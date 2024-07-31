import time
from typing import List, Dict, Optional
import PIL
from aixblock_ml.model import AIxBlockMLBase
import cv2
# from ultralytics import YOLO
import os
# from git import Repo
import zipfile
import subprocess
import requests
# from IPython.display import Image
import yaml
import threading
import shutil

import asyncio
import logging
import signal
import logging
import base64
from PIL import Image
from io import BytesIO

from centrifuge import CentrifugeError, Client, ClientEventHandler, SubscriptionEventHandler

import base64
import hmac
import json
import hashlib
from dashboard import promethus_grafana

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("{0}/{1}.log".format("logs", "pytorch-ddp")),
    logging.StreamHandler()
])
# def promethus(job):
#     from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
#     from prometheus_client.exposition import basic_auth_handler

#     def my_auth_handler(url, method, timeout, headers, data):
#         username = 'admin'
#         password = 'admin'
#         return basic_auth_handler(url, method, timeout, headers, data, username, password)

#     registry = CollectorRegistry()
#     g = Gauge('job_last_success_unixtime', 'Last time a batch job successfully finished', registry=registry)
#     g.set_to_current_time()
#     push_to_gateway('103.160.78.156:9091', job=job, registry=registry, handler=my_auth_handler)

def base64url_encode(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=")

def generate_jwt(user, channel=""):
    """Note, in tests we generate token on client-side - this is INSECURE
    and should not be used in production. Tokens must be generated on server-side."""
    hmac_secret = "d0a70289-9806-41f6-be6d-f4de5fe298fb"  # noqa: S105 - this is just a secret used in tests.
    header = {"typ": "JWT", "alg": "HS256"}
    payload = {"sub": user}
    if channel:
        # Subscription token
        payload["channel"] = channel
    encoded_header = base64url_encode(json.dumps(header).encode("utf-8"))
    encoded_payload = base64url_encode(json.dumps(payload).encode("utf-8"))
    signature_base = encoded_header + b"." + encoded_payload
    signature = hmac.new(hmac_secret.encode("utf-8"), signature_base, hashlib.sha256).digest()
    encoded_signature = base64url_encode(signature)
    jwt_token = encoded_header + b"." + encoded_payload + b"." + encoded_signature
    return jwt_token.decode("utf-8")

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# )
# cf_logger = logging.getLogger("centrifuge")
# cf_logger.setLevel(logging.DEBUG)

async def get_client_token() -> str:
    return generate_jwt("42")

async def get_subscription_token(channel: str) -> str:
    return generate_jwt("42", channel)

class ClientEventLoggerHandler(ClientEventHandler):
    async def on_connected(self, ctx):
        logging.info("Connected to server")

class SubscriptionEventLoggerHandler(SubscriptionEventHandler):
    async def on_subscribed(self, ctx):
        logging.info("Subscribed to channel")

def setup_client(channel_log):
    client = Client(
        "wss://rt.aixblock.io/centrifugo/connection/websocket",
        events=ClientEventLoggerHandler(),
        get_token=get_client_token,
        use_protobuf=False,
    )

    sub = client.new_subscription(
        channel_log,
        events=SubscriptionEventLoggerHandler(),
        # get_token=get_subscription_token,
    )

    return client, sub

async def send_log(sub, log_message):
    try:
        await sub.publish(data={"log": log_message})
    except CentrifugeError as e:
        print(e)
        # logging.error("Error publish: %s", e)

# def log_training_progress(log_message):
#     client.disconnect()  # Đóng kết nối

async def log_training_progress(sub, log_message):
    await send_log(sub, log_message)

def run_train(command, channel_log="training_logs"):
    def run():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            client, sub = setup_client(channel_log)
            async def main():
                await client.connect()
                await sub.subscribe()
                await log_training_progress(sub, "Training started")
                log_file_path = "logs/pytorch-ddp.log"
                if os.path.exists(log_file_path):
                    os.remove(log_file_path)
                
                # Tạo lại file log
                with open(log_file_path, 'w') as log_file:
                    log_file.write("")

                last_position = 0  # Vị trí đã đọc đến trong file log
                await log_training_progress(sub, "Training training")
                # promethus_grafana.promethus_push_to("training")
                
                while True:
                    try:
                        current_size = os.path.getsize(log_file_path)
                        if current_size > last_position:
                            with open(log_file_path, "r") as log_file:
                                log_file.seek(last_position)
                                new_lines = log_file.readlines()
                                # print(new_lines)
                                for line in new_lines:
                                    print("--------------" ,f"{line.strip()}")
                        #             # Thay thế đoạn này bằng code để gửi log
                                    await log_training_progress(sub, f"{line.strip()}")
                            last_position = current_size  

                        time.sleep(5)
                    except Exception as e:
                        print(e)

                # promethus_grafana.promethus_push_to("finish")
                # await log_training_progress(sub, "Training completed")
                # await client.disconnect()
                # loop.stop()

            try:
                loop.run_until_complete(main())
            finally:
                loop.close()  # Đảm bảo vòng lặp được đóng lại hoàn toàn
    
        except Exception as e:
            print(e)

    thread_start = threading.Thread(target=run)
    thread_start.start()
    subprocess.run(command)
    try:
        promethus_grafana.promethus_push_to("finish")
    except:
        pass

def fetch_logs(channel_log="training_logs"):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    client, sub = setup_client(channel_log)

    async def run():
        await client.connect()
        await sub.subscribe()
        history = await sub.history(limit=-1)
        logs = []
        for pub in history.publications:
            log_message = pub.data.get('log')
            if log_message:
                logs.append(log_message)
        await client.disconnect()
        return logs

    return loop.run_until_complete(run())

HOST_NAME = os.environ.get('HOST_NAME',"http://127.0.0.1:8080")
# To print all environment variables
# for key, value in os.environ.items():
#     print(f"{key} = {value}")

css = """
@import "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css";

.aixblock__title .md p {
    display: block;
    text-align: center;
    font-size: 2em;
    font-weight: 700;
}

.aixblock__tabs .tab-nav {
    justify-content: center;
    gap: 8px;
    padding-bottom: 1rem;
    border-bottom: none !important;
}

.aixblock__tabs .tab-nav > button {
    border-radius: 8px;
    border: 1px solid #DEDEEC;
    height: 32px;
    padding: 8px 10px;
    text-align: center;
    line-height: 1em;
}

.aixblock__tabs .tab-nav > button.selected {
    background-color: #5050FF;
    border-color: #5050FF;
    color: #FFFFFF;
}

.aixblock__tabs .tabitem {
    padding: 0;
    border: none;
}

.aixblock__tabs .tabitem .gap.panel {
    background: none;
    padding: 0;
}

.aixblock__input-image,
.aixblock__output-image {
    border: solid 2px #DEDEEC !important;
}

.aixblock__input-image {
    border-style: dashed !important;
}

footer {
    display: none !important;
}

button.secondary,
button.primary {
    border: none !important;
    background-image: none !important;
    color: white !important;
    box-shadow: none !important;
    border-radius: 8px !important;
}

button.primary {
    background-color: #5050FF !important;
}

button.secondary {
    background-color: #F5A !important;
}

.aixblock__input-buttons {
    justify-content: flex-end;
}

.aixblock__input-buttons > button {
    flex: 0 !important;
}

.aixblock__trial-train {
    text-align: center;
    margin-top: 2rem;
}
"""

js = """
window.addEventListener("DOMContentLoaded", function() {
    function process() {
        let buttonsContainer = document.querySelector('.aixblock__input-image')?.parentElement?.nextElementSibling;
        
        if (!buttonsContainer) {
            setTimeout(function() {
                process();
            }, 100);
            return;
        }
        
        document.querySelectorAll('.aixblock__input-image').forEach(function(ele) {
            ele.parentElement.nextElementSibling.classList.add('aixblock__input-buttons');
        });
    }
    
    process();
});
"""

def download_checkpoint(weight_zip_path, project_id, checkpoint_id, token):
    url = f"{HOST_NAME}/api/checkpoint_model_marketplace/download/{checkpoint_id}?project_id={project_id}"
    payload = {}
    headers = {
        'accept': 'application/json',
        # 'Authorization': 'Token 5d3604c4c57def9a192950ef7b90d7f1e0bb05c1'
        'Authorization': f'Token {token}'
    }
    response = requests.request("GET", url, headers=headers, data=payload) 
    checkpoint_name = response.headers.get('X-Checkpoint-Name')

    if response.status_code == 200:
        with open(weight_zip_path, 'wb') as f:
            f.write(response.content)
        return checkpoint_name
    
    else: 
        return None

def download_dataset(data_zip_dir, project_id, dataset_id, token):
    # data_zip_path = os.path.join(data_zip_dir, "data.zip")
    url = f"{HOST_NAME}/api/dataset_model_marketplace/download/{dataset_id}?project_id={project_id}"
    payload = {}
    headers = {
        'accept': 'application/json',
        'Authorization': f'Token {token}'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    dataset_name = response.headers.get('X-Dataset-Name')
    if response.status_code == 200:
        with open(data_zip_dir, 'wb') as f:
            f.write(response.content)
        return dataset_name
    else:
        return None

def upload_checkpoint(checkpoint_model_dir, project_id, token):
    url = f"{HOST_NAME}/api/checkpoint_model_marketplace/upload/"

    payload = {
        "type_checkpoint": "ml_checkpoint",
        "project_id": f'{project_id}',
        "is_training": True
    }
    headers = {
        'accept': 'application/json',
        'Authorization': f'Token {token}'
    }

    checkpoint_name = None

    # response = requests.request("POST", url, headers=headers, data=payload) 
    with open(checkpoint_model_dir, 'rb') as file:
        files = {'file': file}
        response = requests.post(url, headers=headers, files=files, data=payload)
        checkpoint_name = response.headers.get('X-Checkpoint-Name')

    print("upload done")
    return checkpoint_name

class MyModel(AIxBlockMLBase):

    is_init_model_trial = False
    model_trial_public_url = ""
    model_trial_local_url = ""

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """ 
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}''')
        return []

    def fit(self, event, data, **kwargs):
        """

        
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

    def action(self, project, command, collection, **kwargs):
        import os

        print(f"""
                project: {project},
                command: {command},
              
                kwargs: {kwargs}
            """)
              #  collection: {collection},
        if command.lower() == "train":
            try:
                clone_dir = os.path.join(os.getcwd())
                file_train = kwargs.get("num_epochs", "main.py")
                epochs = kwargs.get("num_epochs", 10)
                imgsz = kwargs.get("imgsz", 224)
                project_id = kwargs.get("project_id")
                token = kwargs.get("token")
                checkpoint_version = kwargs.get("checkpoint_version")
                checkpoint_id = kwargs.get("checkpoint")
                dataset_version = kwargs.get("dataset_version")
                dataset_id = kwargs.get("dataset")
                channel_log = kwargs.get("channel_log", "training_logs")
                world_size = kwargs.get("world_size", "1")
                rank = kwargs.get("rank", "0")
                master_add = kwargs.get("master_add")
                master_port = kwargs.get("master_port", "12345")
                # entry_file = kwargs.get("entry_file")
                configs = kwargs.get("configs")
                # local_rank = kwargs.get("local_rank", "0")

                if imgsz > 224:
                    imgsz = 224
                
                def func_train_model(clone_dir, project_id, imgsz, epochs, token, checkpoint_version, checkpoint_id, dataset_version, dataset_id):
                    os.makedirs(f'{clone_dir}/data_zip', exist_ok=True)

                    weight_path = os.path.join(clone_dir, f"models")
                    dataset_path = "data"

                    if checkpoint_version and checkpoint_id:
                        weight_path = os.path.join(clone_dir, f"models/{checkpoint_version}")
                        if not os.path.exists(weight_path):
                            weight_zip_path = os.path.join(clone_dir, "data_zip/weights.zip")
                            checkpoint_name = download_checkpoint(weight_zip_path, project_id, checkpoint_id, token)
                            if checkpoint_name:
                                print(weight_zip_path)
                                with zipfile.ZipFile(weight_zip_path, 'r') as zip_ref:
                                    zip_ref.extractall(weight_path)

                    if dataset_version and dataset_id:
                        dataset_path = os.path.join(clone_dir, f"datasets/{dataset_version}")
                        if not os.path.exists(dataset_path):
                            data_zip_dir = os.path.join(clone_dir, "data_zip/data.zip")
                            dataset_name = download_dataset(data_zip_dir, project_id, dataset_id, token)
                            if dataset_name: 
                                # if not os.path.exists(dataset_path):
                                with zipfile.ZipFile(data_zip_dir, 'r') as zip_ref:
                                    zip_ref.extractall(dataset_path)
                    # files = [os.path.join(weight_path, filename) for filename in os.listdir(weight_path) if os.path.isfile(os.path.join(weight_path, filename))]
                    # train_dir = os.path.join(os.getcwd(),f"yolov9/runs/train")

                    # script_path = os.path.join(os.getcwd(),f"yolov9/train.py")

                    train_dir = os.path.join(os.getcwd(), "models")
                    log_dir = os.path.join(os.getcwd(), "logs")
                    os.makedirs(train_dir, exist_ok=True)
                    # train_dir = os.path.join(os.getcwd(), "models")

                    os.environ["LOGLEVEL"] = "ERROR"
                    # Lệnh torchrun

                    if configs and configs["entry_file"] != "":
                        command = [
                            "torchrun",
                            "--nproc_per_node", "1", #< count gpu card in compute
                            # "--rdzv-backend", "c10d",
                            "--node-rank", f'{rank}',
                            "--nnodes", f'{world_size}',
                            "--rdzv-endpoint", f'{master_add}:{master_port}',
                            "--master-addr", f'{master_add}',
                            "--master-port", f'{master_port}',
                            f'{configs["entry_file"]}'
                        ]

                        if configs["arguments"] and len(configs["arguments"])>0:
                            args = configs["arguments"]
                            for arg in args:
                                command.append(arg['name'])
                                command.append(arg['value'])

                    else:
                        command = [
                            "torchrun",
                            "--nproc_per_node", "1", #< count gpu card in compute
                            "--rdzv-backend", "c10d",
                            "--node-rank", f'{rank}',
                            "--nnodes", f'{world_size}',
                            "--rdzv-endpoint", f'{master_add}:{master_port}',
                            "--master-addr", f'{master_add}',
                            "--master-port", f'{master_port}',
                            "main.py",
                            "--epochs", f'{epochs}',
                            "--root", f'{dataset_path}',
                            "--save_path", f'{train_dir}/last.pt',
                            "--log_dir", log_dir,
                        ]
                    
                    print(command)
                    # subprocess.run(command)
                    run_train(command, channel_log)
                    checkpoint_model = f'{train_dir}/last.pt'
                    print(checkpoint_model)
                    if os.path.exists(checkpoint_model):
                        # print(checkpoint_model)
                        checkpoint_name = upload_checkpoint(checkpoint_model, project_id, token)
                        if checkpoint_name:
                            weight_path_final = os.path.join(clone_dir, "models", checkpoint_name)
                            os.makedirs(weight_path_final, exist_ok=True)
                            shutil.copy(checkpoint_model, weight_path_final)

                train_thread = threading.Thread(target=func_train_model, args=(clone_dir, project_id, imgsz, epochs, token, checkpoint_version, checkpoint_id, dataset_version, dataset_id, ))

                train_thread.start()

                return {"message": "train completed successfully"}
            
            except Exception as e:
                print(e)
                return {"message": f"train failed: {e}"}
        
        elif command.lower() == "tensorboard":
            def run_tensorboard():
                train_dir = os.path.join(os.getcwd(), "{project_id}")
                log_dir = os.path.join(os.getcwd(), "logs")
                p = subprocess.Popen(f"tensorboard --logdir {log_dir} --host 0.0.0.0 --port=6006 --load_fast=false", stdout=subprocess.PIPE, stderr=None, shell=True)
                out = p.communicate()
                print(out)

            tensorboard_thread = threading.Thread(target=run_tensorboard)
            tensorboard_thread.start()
            return {"message": "tensorboardx started successfully"}
        
        elif command.lower() == "dashboard":
            link = promethus_grafana.generate_link_public("ml_00")
            return {"Share_url": link}
        
        elif command.lower() == "predict":
            import numpy as np
            import torch
            import torch.nn as nn
            import cv2
            import os

            data = kwargs.get("data")
            print(data)
            if data:
                img = data["image"]
                width = img.shape[1]
                height = img.shape[0]
            else:
                data = kwargs.get("image")
                from PIL import Image
                import base64
                from io import BytesIO
                im =  Image.open(BytesIO(base64.b64decode(data)))
                height, width = im.size
                img = np.array(im)
            #https://stackoverflow.com/questions/75324341/yolov8-get-predicted-bounding-box
            def IoU(pred, target, iou_threshold = 0.7):
                results = []
                #target: bbox when annotation
                # determine the coordinates of the intersection rectangle
                print(f"pred:{pred},target:{target}")
                x_left = np.maximum(pred[:,0], target[:,0])
                y_top = np.maximum(pred[:,1], target[:,1])
                x_right = np.minimum(pred[:,0]+pred[:,2], target[:,0]+target[:,2])
                y_bottom = np.minimum(pred[:,1]+pred[:,3], target[:,1]+target[:,3])

                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                print(f"x_left:{x_left},y_top:{y_top},x_right:{x_right},y_bottom:{y_bottom}")
                # compute the areas of the union
                pred_area = pred[:,2] * pred[:,3]
                target_area =  target[:,2] *  target[:,3]
                print(f"pred_area:{pred_area},target_area:{target_area}")
                union_area = pred_area + target_area - intersection_area
                print(f"union_area:{union_area}")
                iou = intersection_area / union_area
                print(f"iou:{iou},iou_threshold:{iou_threshold}")

                results.append({
                    "x_left": x_left[0],
                    "y_top": y_top[0], 
                    "x_right": x_right[0], 
                    "y_bottom": y_bottom[0],
                    "iou": iou
                })

                return results
                 
            # Định nghĩa mô hình Net
            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.fc = nn.Linear(784, 10)

                def forward(self, x):
                    return self.fc(x.view(x.size(0), -1))

            # Hàm dự đoán
            def predict(model, data, device):
                # Load và tiền xử lý ảnh
                # img = data['img']
                im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Resize ảnh
                im = cv2.resize(im, (28, 28))
                tensor = torch.tensor(im / 255.0).unsqueeze(0).unsqueeze(0).float()  # Chuẩn hóa và thêm kích thước batch và channel

                # Chuyển tensor đến thiết bị phù hợp
                tensor = tensor.to(device)

                # Thực hiện dự đoán
                with torch.no_grad():
                    output = model(tensor)

                    _, prediction = torch.max(output, 1)
                    res = IoU(output.cpu().numpy(), output.cpu().numpy(), iou_threshold = 0.7)
                    predicted_label = prediction.item()
                    print(f'Prediction: {prediction}')
                
                return res[0], predicted_label

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model = Net()
            model = nn.DataParallel(model)  # Đảm bảo sử dụng DataParallel khi tạo mô hình

            # Load state_dict từ file
            state_dict = torch.load("./models/last.pt")

            # Tạo new_state_dict với tiền tố "module." phù hợp với cấu trúc DataParallel
            new_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith('module.'):
                    k = 'module.' + k
                new_state_dict[k] = v


            # Load state_dict vào mô hình
            model.load_state_dict(new_state_dict)
            model.to(device)
            model.eval()
            print("Loaded model from checkpoint.")

            res, predicted_label = predict(model,data, device)
            results = []
            scores = res['iou']
            if scores is not None:
                avg_score = sum(scores)/max(len(scores), 1)
            else:
                avg_score = 1

            results.append({
                    # "from_name": self.from_name,
                    # "to_name": self.to_name,
                    "type": 'rectanglelabels',
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "points": [float(res['x_left']), float(res['y_top']), float(res['x_right']), float(res['y_bottom'])],
                        "rectanglelabels":  predicted_label
                    },
                    "score": avg_score
                })
            
            return {
                "message": "predict completed successfully",
                'result': results,
                'score': avg_score
            }
            # except:
            #     return {"message": "predict failed", "result": None}
        
        elif command.lower() == "toolbar":
            try:
                im =  Image.open(BytesIO(base64.b64decode(data['image'])))
                height, width = im.size
                import base64
                from PIL import Image
                from io import BytesIO
                train_dir = os.path.join(os.getcwd(),"pytorch-distributed", "models")
                checkpoint_model = f'{train_dir}/last.pth'
                im = im.resize((784, 3), Image.LANCZOS)
                img_width, img_height = im.size
                tensor = transform_image(BytesIO(base64.b64decode(data['image'])))
                prediction = get_prediction(tensor,checkpoint_model)
                #https://stackoverflow.com/questions/75324341/yolov8-get-predicted-bounding-box
                import io
                from PIL import Image
                results.append({
                        "from_name": self.from_name,
                        "to_name": self.to_name,
                        "type": 'rectanglelabels',
                        "original_width": img_width,
                        "original_height": img_height,
                        "image_rotation": 0,
                        "value": {
                            "points": prediction.item(),
                            "rectanglelabels":  [str(prediction.item())]
                        },
                        "score":  0
                    })
                if scores is not None:
                    avg_score = sum(scores)/max(len(scores), 1)
                else:
                    avg_score = 1
                return {
                    "message": "predict completed successfully",
                    'result': results,
                    'score': avg_score
                }
            except:
                return {"message": "predict failed", "result": None}
        
        elif command.lower() == "logs":
            channel_log = kwargs.get("channel_log", "training_logs")
            logs = fetch_logs(channel_log)
            
            return {"message": "command not supported", "result": logs}
        
        elif command.lower() == "stop":
            subprocess.run(["pkill", "-9", "-f", "main.py"])
            return {"message": "command not supported", "result": "Done"}
        else:
            return {"message": "command not supported", "result": None}

            # return {"message": "train completed successfully"}

    def model(self, project, **kwargs):
        import gradio as gr

        css = """
        .feedback .tab-nav {
            justify-content: center;
        }

        .feedback button.selected{
            background-color:rgb(115,0,254); !important;
            color: #ffff !important;
        }

        .feedback button{
            font-size: 16px !important;
            color: black !important;
            border-radius: 12px !important;
            display: block !important;
            margin-right: 17px !important;
            border: 1px solid var(--border-color-primary);
        }

        .feedback div {
            border: none !important;
            justify-content: center;
            margin-bottom: 5px;
        }

        .feedback .panel{
            background: none !important;
        }


        .feedback .unpadded_box{
            border-style: groove !important;
            width: 500px;
            height: 345px;
            margin: auto;
        }

        .feedback .secondary{
            background: rgb(225,0,170);
            color: #ffff !important;
        }

        .feedback .primary{
            background: rgb(115,0,254);
            color: #ffff !important;
        }

        .upload_image button{
            border: 1px var(--border-color-primary) !important;
        }
        .upload_image {
            align-items: center !important;
            justify-content: center !important;
            border-style: dashed !important;
            width: 500px;
            height: 345px;
            padding: 10px 10px 10px 10px
        }
        .upload_image .wrap{
            align-items: center !important;
            justify-content: center !important;
            border-style: dashed !important;
            width: 500px;
            height: 345px;
            padding: 10px 10px 10px 10px
        }

        .webcam_style .wrap{
            border: none !important;
            align-items: center !important;
            justify-content: center !important;
            height: 345px;
        }

        .webcam_style .feedback button{
            border: none !important;
            height: 345px;
        }

        .webcam_style .unpadded_box {
            all: unset !important;
        }

        .btn-custom {
            background: rgb(0,0,0) !important;
            color: #ffff !important;
            width: 200px;
        }

        .title1 {
            margin-right: 90px !important;
        }

        .title1 block{
            margin-right: 90px !important;
        }

        """
        local_url = os.getenv('LOCAL_URL', None)
        share_url = os.getenv('SHARE_URL', None)

        if not share_url:
            with gr.Blocks(css=css) as demo2:
                with gr.Row():
                    with gr.Column(scale=10):
                        gr.Markdown(
                            """
                            # Theme preview: `AIxBlock`
                            """
                        )

                import numpy as np

                def predict(input_img):
                    import cv2
                    result_t = self.action(project, "predict", collection="", data={"image": input_img})
                    # message = result['message']
                    # results = result['result']
                    # overall_score = result['score']
                    if result_t['result']:
                        for result in result_t['result']:
                            points = result['value']['points']
                            rectangle_label = result['value']['rectanglelabels']
                            x_left = points[0]
                            y_top = points[1]
                            x_right = points[2]
                            y_bottom = points[3]

                            # Vẽ hộp giới hạn
                            # Vẽ hộp giới hạn trên ảnh
                            cv2.rectangle(input_img, (int(x_left), int(y_top)), (int(x_right), int(y_bottom)), (0, 255, 0), 2)


                            # Thêm chú thích vào hộp giới hạn
                            label_text = f'{rectangle_label} (IoU: {1:.2f})'
                            print(label_text)
                            cv2.putText(input_img, label_text, (int(x_left), int(y_top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        # boxes = result['result']['boxes']
                        # names = result['result']['names']
                        # labels = result['result']['labels']

                        # for box, label in zip(boxes, labels):
                        #     box = [int(i) for i in box]
                        #     label = int(label)
                        #     input_img = cv2.rectangle(input_img, box, color=(255, 0, 0), thickness=2)
                        #     # input_img = cv2.(input_img, names[label], (box[0], box[1]), color=(255, 0, 0), size=1)
                        #     input_img = cv2.putText(input_img, names[label], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        #                             (0, 255, 0), 2)

                    return input_img

                def get_checkpoint_list(project):
                    print("GETTING CHECKPOINT LIST")
                    print(f"Proejct: {project}")
                    import os
                    checkpoint_list = [i for i in os.listdir("yolov8/models") if i.endswith(".pt")]
                    checkpoint_list = [f"<a href='./yolov8/checkpoints/{i}' download>{i}</a>" for i in
                                    checkpoint_list]
                    if os.path.exists(f"yolov8/{project}"):
                        for folder in os.listdir(f"yolov8/{project}"):
                            if "train" in folder:
                                project_checkpoint_list = [i for i in
                                                        os.listdir(f"yolov8/{project}/{folder}/weights") if
                                                        i.endswith(".pt")]
                                project_checkpoint_list = [
                                    f"<a href='./yolov8/{project}/{folder}/weights/{i}' download>{folder}-{i}</a>"
                                    for i in project_checkpoint_list]
                                checkpoint_list.extend(project_checkpoint_list)

                    return "<br>".join(checkpoint_list)

                
                with gr.Tabs(elem_classes=["feedback"]) as parent_tabs:
                    with gr.TabItem("Demo", id=0):
                        with gr.Row():
                            gr.Markdown("## Input", elem_classes=["title1"])
                            gr.Markdown("## Output", elem_classes=["title1"])
                    
                        gr.Interface(predict,
                                gr.Image(elem_classes=["upload_image"], sources="upload", container=False, height=345,
                                        show_label=False),
                                gr.Image(elem_classes=["upload_image"], container=False, height=345, show_label=False),
                                allow_flagging=False
                            )

                    with gr.TabItem("Webcam", id=1):
                        gr.Image(elem_classes=["webcam_style"], sources="webcam", container=False, show_label=False,
                                height=450)

                    with gr.TabItem("Video", id=2):
                        gr.Image(elem_classes=["upload_image"], sources="clipboard", height=345, container=False,
                                show_label=False)
        

                gradio_app, local_url, share_url = demo2.launch(share=True, quiet=True, prevent_thread_lock=True,
                                                            server_name='0.0.0.0', show_error=True, server_port=5566)
                
                os.environ['LOCAL_URL'] = local_url
                os.environ['SHARE_URL'] = share_url

        return {"share_url": share_url, 'local_url': local_url}

    def model_trial(self, project, **kwargs):
        while self.is_init_model_trial:
            time.sleep(1)

        if len(self.model_trial_local_url) > 0 and len(self.model_trial_public_url) > 0:
            return {"share_url": self.model_trial_public_url, 'local_url': self.model_trial_local_url}

        self.is_init_model_trial = True
        print(kwargs)
        import gradio as gr

        def mt_predict(input_img):
            import cv2
            result = self.action(project, "predict", collection="", data={"image": input_img})
            print(result)
            if result['result']:
                boxes = result['result']['boxes']
                names = result['result']['names']
                labels = result['result']['labels']

                for box, label in zip(boxes, labels):
                    box = [int(i) for i in box]
                    label = int(label)
                    input_img = cv2.rectangle(input_img, box, color=(255, 0, 0), thickness=2)
                    # input_img = cv2.(input_img, names[label], (box[0], box[1]), color=(255, 0, 0), size=1)
                    input_img = cv2.putText(input_img, names[label], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 2)

            return input_img

        def mt_trial_training(dataset_choosen):
            print(f"Training with {dataset_choosen}")
            result = self.action(project, "train", collection="", data=dataset_choosen)
            return result['message']

        def mt_download_btn(evt: gr.SelectData):
            print(f"Downloading {dataset_choosen}")
            return f'<a href="/yolov8/datasets/{evt.value}" class="aixblock__download-button"><i class="fa fa-download"></i> Download</a>'

        def mt_get_checkpoint_list(project):
            print("GETTING CHECKPOINT LIST")
            print(f"Proejct: {project}")
            import os
            checkpoint_list = [i for i in os.listdir("yolov8/models") if i.endswith(".pt")]
            checkpoint_list = [f"<a href='./yolov8/checkpoints/{i}' download>{i}</a>" for i in
                               checkpoint_list]
            if os.path.exists(f"yolov8/{project}"):
                for folder in os.listdir(f"yolov8/{project}"):
                    if "train" in folder:
                        project_checkpoint_list = [i for i in
                                                   os.listdir(f"yolov8/{project}/{folder}/weights") if
                                                   i.endswith(".pt")]
                        project_checkpoint_list = [
                            f"<a href='./yolov8/{project}/{folder}/weights/{i}' download>{folder}-{i}</a>"
                            for i in project_checkpoint_list]
                        checkpoint_list.extend(project_checkpoint_list)

            return "<br>".join(checkpoint_list)

        def mt_tab_changed(tab):
            if tab == "Download":
                mt_get_checkpoint_list(project=project)

        def mt_upload_file(file):
            return "File uploaded!"

        with gr.Blocks(css=css, js=js) as demo:
            gr.Markdown("AIxBlock", elem_classes=["aixblock__title"])

            with gr.Tabs(elem_classes=["aixblock__tabs"]):
                with gr.TabItem("Demo", id=0):
                    # with gr.Row():
                    #     gr.Markdown("Input", elem_classes=["aixblock__input-title"])
                    #     gr.Markdown("Output", elem_classes=["aixblock__output-title"])

                    gr.Interface(
                        mt_predict,
                        gr.Image(elem_classes=["aixblock__input-image"], container=False, height=345),
                        gr.Image(elem_classes=["aixblock__output-image"], container=False, height=345),
                        allow_flagging="never",
                    )

                # with gr.TabItem("Webcam", id=1):
                #     gr.Image(elem_classes=["webcam_style"], sources="webcam", container = False, show_label = False, height = 450)

                # with gr.TabItem("Video", id=2):
                #     gr.Image(elem_classes=["upload_image"], sources="clipboard", height = 345,container = False, show_label = False)

                # with gr.TabItem("About", id=3):
                #     gr.Label("About Page")

                with gr.TabItem("Trial Train", id=2):
                    gr.Markdown("Dataset template to prepare your own and initiate training")
                    # with gr.Row():
                    # get all filename in datasets folder
                    datasets = [(f"dataset{i}", name) for i, name in enumerate(os.listdir('./datasets'))]
                    dataset_choosen = gr.Dropdown(datasets, label="Choose dataset", show_label=False, interactive=True,
                                                  type="value")
                    # gr.Button("Download this dataset", variant="primary").click(download_btn, dataset_choosen, gr.HTML())
                    download_link = gr.HTML("""<em>Please select a dataset to download</em>""")
                    dataset_choosen.select(mt_download_btn, None, download_link)

                    # when the button is clicked, download the dataset from dropdown
                    # download_btn
                    gr.Markdown("Upload your sample dataset to have a trial training")
                    # gr.File(file_types=['tar','zip'])

                    gr.Interface(
                        mt_predict,
                        gr.File(elem_classes=["aixblock__input-image"], file_types=['tar', 'zip'], container=False),
                        gr.Label(elem_classes=["aixblock__output-image"], container=False),
                        allow_flagging="never",
                    )

                    with gr.Column(elem_classes=["aixblock__trial-train"]):
                        gr.Button("Trial Train", variant="primary").click(mt_trial_training, dataset_choosen, None)
                        gr.HTML(value=f"<em>You can attemp up to {2} FLOps</em>")

                # with gr.TabItem("Download"):
                #     with gr.Column():
                #         gr.Markdown("## Download")
                #         with gr.Column():
                #             gr.HTML(get_checkpoint_list(project))

        gradio_app, local_url, share_url = demo.launch(share=True, quiet=True, prevent_thread_lock=True,
                                                       server_name='0.0.0.0', show_error=True)
        self.model_trial_public_url = share_url
        self.model_trial_local_url = local_url
        self.is_init_model_trial = False

        return {"share_url": share_url, 'local_url': local_url}

    def download(self, project, **kwargs):
        return super(self).download(project, **kwargs)

# results.append({
# 	             	'from_name': self.from_name,
# 	             	'to_name': self.to_name,
# 	             	"original_width": img_width,
# 	             	"original_height": img_height,
# 	             	'type': 'rectanglelabels',
# 	             	'value': {
# 	                    'rectanglelabels':[name_],
# 	                    'x': x_min / img_width * 100,
# 	                    'y': y_min / img_height * 100,
# 	                    'width': (x_max - x_min) / img_width * 100,
# 	                    'height': (y_max - y_min) / img_height * 100
# 	                },
# 	                'score': confidence
#                 })
#                 all_scores.append(confidence)
#                 print(results)

#         avg_score = sum(all_scores) / max(len(all_scores), 1)
        
#         return [{
#             'result': results,
#             'score': avg_score
#         }]
# results.append({
#                 "from_name": self.from_name,
#                 "to_name": self.to_name,
#                 "type": 'polygonlabels',
#                 "original_width": img_width,
#                 "original_height": img_height,
#                 "image_rotation": 0,
#                 "value": {
#                     "points": points,
#                     "polygonlabels": [output_label]
#                 },
#                 "id": i,
#                 "score": scores[i]    
#             })
#         if scores is not None:
#             avg_score = sum(scores)/max(len(scores), 1)
#         else:
#             avg_score = 1
#         return [{
#             'result': results,
#             'score': avg_score
#         }]
# import cv2
# import numpy as np
# def mask_to_polygons(self, mask, max_width, max_height):
#         # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
#         # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
#         # Internal contours (holes) are placed in hierarchy-2.
#         # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
#         mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
#         res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
#         hierarchy = res[-1]
#         if hierarchy is None:  # empty mask
#             return [], False
#         has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
#         res = res[-2][0]
#         res = res.flatten()

#         # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
#         # We add 0.5 to turn them into real-value coordinate space. A better solution
#         # would be to first +0.5 and then dilate the returned polygon by 0.5.
#         # res = [list(x + 0.5) for x in res if len(x) >= 6]
#         res = list(res+0.5) if len(res) >=6 else []
#         res = [[res[i]/max_width*100, res[i+1]/max_height*100] for i in range(0, len(res), 2)]

#         return res, has_holes
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io



# Image to Tensor
def transform_image(image_bytes):
	transform = transforms.Compose([
		transforms.Grayscale(num_output_channels=1),
		transforms.Resize((28, 28)),
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))            # the mean = 0.1307 & standard deviation = 0.3081
	])

	image = Image.open(io.BytesIO(image_bytes))
    # torch.unsqueeze returns a new tensor with a dimension of size one inserted at the specified position.
    
	return transform(image).unsqueeze(0)                      # 0 means first dimension


# predict
def get_prediction(image_tensor,model_path):
    # load Model
    ## Creating the Neural Network
    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            self.fc = nn.Linear(784, 3)

        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))
    
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNet, self).__init__()
            self.layer1 = nn.Linear(784, 3)
            # self.relu = nn.ReLU()
            # self.layer2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            output = self.layer1(x)
            output = self.relu(output)
            output = self.layer2(output)
            return output

    input_size = 784
    hidden_size = 3
    num_classes = 3 #load class of project
    model = Net()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
   
    # model =  NeuralNet(input_size, hidden_size, num_classes)
    # model =  Net()
    with torch.no_grad():
        batch = image_tensor.to(device)
        output = model( batch )
        output = torch.argmax(output, 1)
        print(output)
    # PATH = 'mnist_ffn.pth'
    # model.load_state_dict(torch.load(model_path,map_location='cpu'))
    # checkpoint=torch.load(model_path,map_location='cpu')
   
    # model = checkpoint['model']
    # print(checkpoint)
    model = model.to(device)
    model.eval()
    images = image_tensor.reshape(-1, 28*28)
    outputs = model(images)
    _, prediction = torch.max(outputs, 1)

    return prediction