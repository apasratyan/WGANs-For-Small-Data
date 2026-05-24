from torch import nn, optim
import pickle
from tqdm import tqdm

def train(loaders, classifier_class, device, iters_per_epoch, n_epochs, p_grid, save_path):
    def train_epoch(model, optimizers, schedulers, loader, loss_logs, n_iters, p):
        model.train()
        cross_entropy = nn.CrossEntropyLoss()
        for i in range(n_iters):
            batch, classes = loader.__iter__().__next__()
            batch = batch.to(device)
            classes = classes.to(device)
            
            indices = (torch.rand(batch.shape[0]) < p).to(device)
            with torch.no_grad():
                gen = wgan.generate(indices.sum().item(), classes = classes[indices])
                batch[indices] = gen
            
            optimizers['clf'].zero_grad()
            logits = classifier(batch)
            accuracy = (logits.argmax(dim = 1) == classes).float().mean()
            clf_loss = cross_entropy(logits, classes)
            clf_loss.backward()
            optimizers['clf'].step()
            
            loss_logs[0].append(clf_loss.item())
            loss_logs[1].append(accuracy.item())

    @torch.no_grad()
    def test_epoch(model, loader, loss_logs):
        model.eval()
        cross_entropy = nn.CrossEntropyLoss()
        mean_clf_loss = 0
        mean_accuracy = 0
        for batch, classes in loader:
            batch = batch.to(device)
            classes = classes.to(device)
            logits = classifier(batch)
            preds = logits.argmax(dim = 1)
            accuracy = (logits.argmax(dim = 1) == classes).float().mean()
            clf_loss = cross_entropy(logits, classes)
            
            mean_clf_loss += clf_loss.item() * batch.shape[0]
            mean_accuracy += accuracy.item() * batch.shape[0]
        return mean_clf_loss / len(loader.dataset), mean_accuracy / len(loader.dataset)

    def train_n_epochs(model, optimizers, schedulers, train_loader, test_loader, loss_logs, iters_per_epoch, n_epochs, p = 0.0):
        train_loss, train_acc, test_loss, test_acc = loss_logs

        for i in tqdm(range(n_epochs)):
            print(f'epoch {i + 1}')

            train_epoch(model, optimizers, schedulers, train_loader, [train_loss, train_acc], iters_per_epoch, p)
            test_result = test_epoch(model, test_loader, [test_loss, test_acc])
            print(f'loss/acc: {test_result[0]:.5f}/{test_result[1]:.5f}')
            test_loss.append(test_result[0])
            test_acc.append(test_result[1])
            schedulers['clf'].step()

    loss_logs = [[[], [], [], []] for i in range(p_grid.shape[0])]
    pred_logs = []

    train_loader = loaders['train']
    test_loader = loaders['test']
    
    for i, p in enumerate(p_grid):
        classifier = classifier_class().to(device)
        optimizers = {}
        optimizers['clf'] = optim.AdamW(classifier.parameters(), lr = 2e-4, weight_decay = 1e-3)
        schedulers = {}
        schedulers['clf'] = optim.lr_scheduler.LinearLR(optimizers['clf'], start_factor = 1, end_factor = 0, total_iters = iters_per_epoch * n_epochs)
        train(classifier, optimizers, schedulers, train_loader, test_loader, loss_logs[i], iters_per_epoch, n_epochs, p)
        classifier.eval()
        pred_logs.append([])
        for batch, _ in test_loader:
            with torch.no_grad():
                logits = classifier(batch.to(device)).cpu()
                preds = logits.argmax(dim = 1)
                pred_logs[-1].append(preds)
        pred_logs[-1] = torch.concat(pred_logs[-1])

    logs = {
        'loss_logs' : loss_logs,
        'pred_logs' : pred_logs
    }

    with open(save_path, 'wb') as f:
        pickle.dump(logs, f)
