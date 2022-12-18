import torch
import sys
import pandas as pd
import numpy as np
from test_pygad_my_model import control_model
from walking import supervised_sim


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Input and output files needed")
        exit(1)

    ## load dataset
    X = pd.read_csv(sys.argv[1], header=None)
    Y = pd.read_csv(sys.argv[2], header=None)

    # print("X is", X) [100000 rows x 19 columns]

    X = X.astype("float32")
    Y = Y.astype("float32")

    
    control_model.to("cuda")
    control_model.linear_relu_stack.to("cuda")
    # define the optimization
    criterion = torch.nn.MSELoss().to("cuda")
    optimizer = torch.optim.SGD(control_model.parameters(), lr=0.01, momentum=0.9)

    test_train_rate = 0.75

   
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for epoch in range(20):
            print("Epoch ", epoch)
            
            for i in range(round(len(X) * test_train_rate)):

                # clear the gradients
                optimizer.zero_grad(set_to_none=True)
                # print(X.values[:, i])
                # print(X.values[i, :])

                # compute the model output
                print(X.values[i, :].reshape(1,19).shape)
                yhat = control_model(torch.Tensor(X.values[i, :].reshape(1,19)).to("cuda"))
                # calculate loss
                print(yhat)
                print(yhat.dtype)
                print(torch.Tensor(Y.values[i, :].reshape(1,12)).to("cuda").dtype)
                loss = criterion(yhat.to("cuda"), torch.Tensor(Y.values[i, :].reshape(1,12)).to("cuda"))
                # credit assignment
                #loss = loss.to("cuda")
                loss.backward()
                # update model weights
                optimizer.step()
    torch.cuda.current_stream().wait_stream(s)
        


    print("Error is")
    # compute the model output
    yhat = control_model(torch.Tensor(X.values[round(len(X) * test_train_rate):, :]).to("cuda"))
    # calculate loss
    loss = criterion(yhat, torch.Tensor(Y.values[round(len(Y) * test_train_rate):, :]).to("cuda"))
    # credit assignment
    loss.backward()

    print(loss)

    supervised_sim(control_model, 10000, True)
