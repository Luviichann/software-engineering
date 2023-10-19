

def train_model(model,epochs,loss_fn,optimizer,train_set):
    for epoch in range(epochs+1):
        loss_train = 0.0
        for X,y in train_set:

            y_pre = model(X)
            loss = loss_fn(y_pre,y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
        if epoch%10 == 0:
            print('epoch: ',epoch,'  loss:  ',loss_train/len(train_set))
    
