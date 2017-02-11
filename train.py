import os
import sys
import torch
import torch.nn.functional as F


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), feature.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = corrects/batch.batch_size * 100.0
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%'.format(steps, loss.data[0], accuracy))
            if steps % args.test_interval == 0:
                eval(dev_iter, model, args)
            if steps % args.save_interval == 0:
                save_prefix = os.path.join(args.save_dir, 'snapshot')
                save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
                torch.save(model, save_path)


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), feature.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    accuracy = corrects/size * 100.0
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))


def predict(x, model):
    output = model(x)
    _, predicted = torch.max(output, 1)
    return predicted
