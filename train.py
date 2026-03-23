import os
import sys
import torch
import torch.nn.functional as F
from mydatasets import clean_str


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    if getattr(args, 'optimizer', 'adam') == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, rho=0.95)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            model.train()
            feature, target = batch
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            # Kim's paper: L2 weight constraint on fc layer (rescale if norm > max_norm)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'fc1.weight' in name:
                        norms = param.norm(2, dim=-1, keepdim=True)
                        desired = torch.clamp(norms, max=args.max_norm)
                        param.mul_(desired / (norms + 1e-8))

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1] == target).sum()
                accuracy = 100.0 * corrects / feature.size(0)
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             accuracy.item(),
                                                                             corrects.item(),
                                                                             feature.size(0)))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps, best acc: {:.4f}%'.format(
                            args.early_stop, best_acc))
                        return best_acc
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)

    return best_acc


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    total = 0
    with torch.no_grad():
        for batch in data_iter:
            feature, target = batch
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            logit = model(feature)
            loss = F.cross_entropy(logit, target, reduction='sum')

            avg_loss += loss.item()
            corrects += (torch.max(logit, 1)[1] == target).sum().item()
            total += target.size(0)

    avg_loss /= total
    accuracy = 100.0 * corrects / total
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       total))
    return accuracy


def predict(text, model, text_vocab, label_vocab, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    tokens = clean_str(text).lower().split()
    text_ids = [text_vocab.stoi.get(t, 1) for t in tokens]  # 1 = <unk>
    x = torch.tensor([text_ids], dtype=torch.long)
    if cuda_flag:
        x = x.cuda()
    with torch.no_grad():
        output = model(x)
    _, predicted = torch.max(output, 1)
    return label_vocab.itos[predicted.item()]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
