import logging 
import os 
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs")

print("Current working directory:", os.getcwd())
print("Logs path:", logs_path)

os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)
print("Log file path:", LOG_FILE_PATH)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(levelno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


# if __name__=="__main__":  
#     logging.info("Logging has started")
    