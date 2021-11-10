"""
author: isai
email: isaiyamudhu@gmail.com
"""
from utils.all_utils import save_model, save_plot
from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd
import numpy as np
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s ] %(message)s"
logging_dir = "oneNeuron_logs"
os.makedirs(logging_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(logging_dir,"running_logs.log"), level=logging.INFO, format=logging_str, filemode="a")


def main(data,eta,epochs,modelFileNam,plotFileNam):
    df = pd.DataFrame(data)
    #print(df)
    logging.info(f"This is actual dataframe \n {df}")
    X,y = prepare_data(df)

    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)
    _ = model.total_loss() #dummy variable

    save_model(model,modelFileNam)   
    save_plot(df,plotFileNam,model)

if __name__ == "__main__": # entry point
    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,1,1,1],
    }
    ETA = 0.001 # 0 and 1
    EPOCHS = 100
    try:
        logging.info(">>> STARTING TRAINING >>>")
        main(data=OR,eta=ETA,epochs=EPOCHS,modelFileNam="or.model",plotFileNam="or.png")
        logging.info("<<< TRAINING DONE SUCCESSFULLY!!! <<<\n")
    except Exception as e:
        logging.exception(e)
        raise e