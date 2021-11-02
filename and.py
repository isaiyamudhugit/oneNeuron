from utils.all_utils import save_model, save_plot
from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd
import numpy as np

def main(data,eta,epochs,modelFileNam,plotFileNam):
    df = pd.DataFrame(data)
    print(df)
    X,y = prepare_data(df)

    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)
    _ = model.total_loss() #dummy variable

    save_model(model,modelFileNam)   
    save_plot(df,plotFileNam,model)

if __name__ == "__main__": # entry point
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }
    ETA = 0.3 # 0 and 1
    EPOCHS = 10
    main(data=AND,eta=ETA,epochs=EPOCHS,modelFileNam="and.model",plotFileNam="and.png")