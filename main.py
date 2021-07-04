# coding: utf-8
import argparse
import math
import torch
import torch.nn as nn
import time
import os
import matplotlib.pyplot as plt

from data import Corpus
from Transformer import TransformerModel
from RNN import RNNModel,MogLSTM
from smoothloss import LabelSmoothLoss


parser = argparse.ArgumentParser(description='PyTorch Language Model')

## about model configuration
parser.add_argument('--model', type=str, help='language model from RNN or Transformer')

## for both models
parser.add_argument('--ninp', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=50,
                    help='size of hidden units')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--optimizer', type=str, default='SGD',
                    help='optimizer used')
parser.add_argument('--scheduler', type=str, default='CosineAnnealing',
                    help='scheduler used')
parser.add_argument('--tied', action='store_true', default = False,
                    help='tie the word embedding and softmax weights')
parser.add_argument('--xavier', action='store_true', default=False,
                    help='Use Xavier Initialization')



## for RNN model
parser.add_argument('--rnn_type', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--bidirectional', action='store_true',default=False,
                    help='Use Bidirectional Recurrence')

## for Transformer model
parser.add_argument('--nheads', type=int, default=5,
                    help='number of heads in self-attention')

## about model training and evaluation
parser.add_argument('--train_batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
parser.add_argument('--bptt', type=int, default=35, metavar='N',
                    help='sequence length')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')


## parallel training
parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

args = parser.parse_args()


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def train(model):
    model = model.cuda()
    model.train() # Turn on the train mode
    total_loss = 0.
    epoch_loss = 0.
    start_time = time.time()
    log_interval = 200
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.train_batch_size)
    # break sentences by length of args.bptt. 'batch' is the serial number of subsentences and i is first word number of subsentences
    for batch, i in enumerate(range(0, data_loader.train_data.size(0) - 1, args.bptt)): 
        data, targets = data_loader.get_batch(data_loader.train_data, i)
        data = data.cuda()
        targets = targets.cuda()
        optimizer.zero_grad()
        
        if args.model == 'Transformer':
            output, att_weights = model(data)
        else:
            ## assign hidden to a new variable for each batch for RNN model
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)

        output = output.view(-1, ntokens)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        
        total_loss += loss.item()
        epoch_loss += len(data)*loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval    ## avg crossentropy loss over log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.4f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(data_loader.train_data) // args.bptt, optimizer.state_dict()['param_groups'][0]['lr'],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

    return epoch_loss / len(data_loader.train_data)    ## average cross-entropy loss in training set


# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.

def evaluate(model, data_source):
    model.eval() # Turn on the evaluation mode
    total_loss = 0.
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.eval_batch_size)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = data_loader.get_batch(data_source, i)
            data = data.cuda()
            targets = targets.cuda()
            if args.model == 'Transformer':
                output, att_weights = model(data)
            else:
                hidden = repackage_hidden(hidden)
                output, hidden = model(data, hidden)
                
            
            output = output.view(-1, ntokens)
            loss = criterion(output, targets)
            total_loss += len(data)*loss.item()
    if args.model == 'Transformer':
        return total_loss / len(data_source), att_weights, data 
    else:
        return total_loss / len(data_source)
        ## average cross-entropy loss in validation set
    
    



if __name__ == '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    
    # load data
    data_loader = Corpus(args.train_batch_size,
                        args.eval_batch_size, args.bptt)
    ntokens = data_loader.get_ntokens()


    if args.model == 'RNN':
        model = RNNModel(args.rnn_type, ntokens, args.ninp,
                        args.nhid, args.nlayers,args.dropout,
                        args.bidirectional,args.tied,args.xavier)

    elif args.model == 'Transformer':
        model = TransformerModel(ntokens, args.ninp, args.nheads, args.nhid, args.nlayers)



    lr = args.lr  
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9,nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    criterion = torch.nn.CrossEntropyLoss()
    
    ## print model information
    nl = '\n'
    print("{:-^50s}".format("model information"))
    print(f'model:{args.model}{nl}embedding size:{args.ninp}{nl}hidden size:{args.nhid}{nl}lr:{args.lr}{nl}layers:{args.nlayers}{nl}optimizer:{args.optimizer}{nl}')
    print("{:-^50s}".format("training log"))
    
    ## WARNING:if using adam/adamW as optimizer, the assigned lr (when bigger than 0.1) cannot be used since it causes gradient explosion
    # if args.optimizer == 'AdamW':
    #     optimizer = torch.optim.AdamW(model.parameters())
    # elif args.optimizer == 'SGD':
    #     optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9,nesterov=True)

        
    # if args.scheduler == 'ReduceLROnPlateau':
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=1, cooldown=3)
    # elif args.scheduler == 'CosineAnnealing':
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2)
    
    # if torch.cuda.device_count() > 1:
    #     torch.distributed.init_process_group(backend="nccl")
    #     local_rank = torch.distributed.get_rank()
    #     torch.cuda.set_device(local_rank)
    #     device = torch.device("cuda", local_rank)
    #     model.to(device)
    #     model = torch.nn.parallel.DistributedDataParallel(model,
    #                                                   device_ids=[local_rank],
    #                                                   output_device=local_rank)

    # elif torch.cuda.device_count() == 1:
    model = model.cuda()
    # Train Function
    best_val_loss = float("inf")
    best_model = None
    best_att_weight = None
    train_loss_epoch = []
    val_loss_epoch = []
    
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss = train(model)
            print(f'training loss of epoch {epoch} is {train_loss}')
            if args.model == 'Transformer':
                val_loss, att_weight, data = evaluate(model, data_loader.val_data)
            else:
                val_loss = evaluate(model, data_loader.val_data)
            
            print(f'validation loss of epoch {epoch} is {val_loss}')
            train_loss_epoch.append(train_loss)
            val_loss_epoch.append(val_loss)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                torch.save(best_model, f'../checkpoints/best_{args.model}.pt')
            scheduler.step()
        ## note optimizer.step() occurs per batch (iteration), while scheduler.step() occurs per epoch.
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    ## print loss curves with matplotlib
    fig = plt.figure(figsize = (6,8))
    ax = fig.add_subplot(211)
    ax.plot(train_loss_epoch)
    ax.plot(val_loss_epoch)
    ax.set_title('loss curves',fontsize=16)
    ax.set(xlabel='epoch',ylabel='loss')
    ax.legend(['training loss','validation loss'],loc='best')
    ay = fig.add_subplot(212)
    ay.plot([math.exp(x) for x in train_loss_epoch])
    ay.plot([math.exp(x) for x in val_loss_epoch])
    ay.set_title('perplexity',fontsize=16)
    ay.set(xlabel='epoch',ylabel='ppl')
    ay.legend(['training ppl','validation ppl'],loc='best')
    fig.subplots_adjust(top=1.5)
    plt.savefig(f'../training_curves/{args.model}_curves', bbox_inches='tight')

    ## evaluate best_model in test set
    if args.model == 'Transformer':
        info = {}
        test_loss , att_weight, data = evaluate(best_model, data_loader.test_data)
        info['data'] = data
        info['att_weight'] = att_weight
        torch.save(info, '../best_att_weight.pt')
    else:
        test_loss = evaluate(best_model, data_loader.test_data)
    
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)




