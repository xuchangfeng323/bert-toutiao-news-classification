from metrics import Metrics
from torch import device
from sched import scheduler
import swanlab
from tqdm import tqdm
import torch
from transformers import BertTokenizer
class trainer:
    def __init__(self,config):
        self.model=config.model
        self.optimizer=config.optimizer
        self.scheduler=config.scheduler
        self.device=config.device
        self.num_epochs=config.num_epochs
        self.tokenizer=config.tokenizer
        self.config=config
        self.loss_fn = config.loss_fn
        self.model.to(self.device)
        self.metrics = Metrics(num_classes=15)
        self.best_accuracy = 0.0
        
    def train(self,traindataLoader, devdataLoader, testdataLoader):
        
        swanlab.init(
            project="demo1",  
            name="bert",                
            config={
                "num_epochs": self.config.num_epochs,
                "learning_rate": self.config.lr,
                "batch_size": self.config.batch_size,
                "model": "bert-base-chinese"
            }
        )
        for epoch in range(self.num_epochs):
            self.model.train()
            total_train_loss = 0
            progress_bar = tqdm(traindataLoader, desc=f"Epoch {epoch + 1}/{self.num_epochs} [Train]", position=0, leave=True)
            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                labels = batch['labels']
                logits = self.model(batch['input_ids'], batch['attention_mask'])
                loss=self.loss_fn(logits, labels)
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                total_train_loss += loss.item()
                progress_bar.set_postfix({"Loss": loss.item()})
                if step % 50 == 0:
                    swanlab.log({
                        "train/loss_step": loss.item(),
                        "train/learning_rate": self.optimizer.param_groups[0]['lr']
                    }, step=epoch * len(self.traindataLoader) + step)
            avg_train_loss = total_train_loss / len(self.traindataLoader)
            swanlab.log({
                "train/loss_epoch": avg_train_loss
            }, step=epoch)
            self.eval(epoch, devdataLoader)
        self.test(testdataLoader)
        swanlab.finish()
            
    def load_model(self, path):
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
           
            self.model.load_state_dict(checkpoint)
        
        
        
    def eval(self,epoch, devdataLoader):
        self.model.eval()
        
        total_eval_loss = 0
        eval_correct = 0
        total_samples = 0
        progress_bar = tqdm(devdataLoader, desc="Evaluation", position=0, leave=True)
        with torch.no_grad():
            for batch in progress_bar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch['labels']
                logits = self.model(batch['input_ids'], batch['attention_mask'])
                loss_fn = self.loss_fn
                loss = loss_fn(logits, labels)
                total_eval_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                eval_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                if epoch==self.num_epochs-1:
                    self.metrics.add(predictions, labels)
        
        results = self.metrics.get_results()
        print(results)
        self.metrics.reset()
        avg_eval_loss = total_eval_loss / len(devdataLoader)
        eval_accuracy = eval_correct / total_samples  
        print(f"Eval Accuracy: {eval_accuracy:.4f}")
        
        swanlab.log({
            "eval/loss": avg_eval_loss,
            "eval/accuracy": eval_accuracy,
            "eval/results": results
        })

        if eval_accuracy > self.best_accuracy:
            self.best_accuracy = eval_accuracy
            
            
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'accuracy': eval_accuracy,
            }
            torch.save(best_checkpoint, f"models/best_model.pth")
        
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss.item(),
            'accuracy': eval_accuracy,
        }
        torch.save(checkpoint, f'/checkpoints/checkpoint_{epoch}.pth')
    def test(self, testdataLoader):
        self.model.eval()
        total_test_loss = 0
        test_correct = 0
        total_samples = 0
        progress_bar = tqdm(testdataLoader, desc="Testing", position=0, leave=True)
        with torch.no_grad():
            for batch in progress_bar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch['labels']
                logits = self.model(batch['input_ids'], batch['attention_mask'])
                loss_fn = self.loss_fn
                loss = loss_fn(logits, labels)
                total_test_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                test_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                self.metrics.add(predictions, labels)
        results = self.metrics.get_results()
        swanlab.log({
            "test/results": results
        })
        print("Test Results:")
        print(results)
        self.metrics.reset()
        avg_test_loss = total_test_loss / len(testdataLoader)
        test_accuracy = test_correct / total_samples  
        print(f"Test Accuracy: {test_accuracy:.4f}")
        swanlab.log({
            "test/loss": avg_test_loss,
            "test/accuracy": test_accuracy
        })
        
    def predict(self, text, model_path=None):
        if model_path is not None:
            self.load_model(model_path)

        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.model(inputs['input_ids'], inputs['attention_mask'])
            predictions = torch.argmax(logits, dim=-1)
            print(f"预测类别: {predictions.item()}")
            return predictions.item()

        
        
        
        
        
        