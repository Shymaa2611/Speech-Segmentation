from dataset import prepare_dataset
from model import AudioClassifier
from trainer import train
from visualization import plot_loss,plot_accuracy

if __name__=="__main__":
    data_dir = "D:\\MachineCourse\\musan\\musan"
    X_train_tensor,X_test_tensor,y_train_tensor,y_test_tensor,X_train,y_train,y_test,num_classes=prepare_dataset(data_dir)
    input_size = X_train.shape[2]
    model = AudioClassifier(input_size, num_classes)
    train_losses,val_losses,train_acc,val_accuracy=train(model,X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor,y_train,y_test)
    plot_loss(train_losses,val_losses)
    plot_accuracy(train_acc,val_accuracy)
    #prediction
    

    


