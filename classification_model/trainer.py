from config import *
import torch
import torch.optim as optim

def validation(model,X_test_tensor,y_test_tensor,y_test):
   model.eval()
   with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, X_test_tensor)
    test_accuracy = torch.sum(torch.argmax(test_outputs, dim=1) == y_test_tensor).item() / len(y_test)
    print(f"Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy:.4f}")


def train(model,X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor,y_train,y_test):
   optimizer = optim.Adam(model.parameters(), lr=learning_rate)
   train_losses, val_losses, train_acc, val_acc = [], [], [], []
   for epoch in range(num_epochs):
       model.train()
       outputs = model(X_train_tensor)  
       loss = criterion(outputs, y_train_tensor)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

       model.eval()
       with torch.no_grad():
          train_loss = loss.item()
          train_accuracy = torch.sum(torch.argmax(outputs, dim=1) == y_train_tensor).item() / len(y_train)
          val_outputs = model(X_test_tensor)
          val_loss = criterion(val_outputs, y_test_tensor)
          val_accuracy = torch.sum(torch.argmax(val_outputs, dim=1) == y_test_tensor).item() / len(y_test)
       train_losses.append(train_loss)
       val_losses.append(val_loss.item())
       train_acc.append(train_accuracy)
       val_acc.append(val_accuracy)
       print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}")
   torch.save(model.state_dict(),"classifier_model.pt")
   return train_losses,val_losses,train_acc,val_accuracy