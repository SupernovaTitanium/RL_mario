import torch
from torch.autograd import Variable
for i in range(1):
    torch.manual_seed(i)
    x1=torch.FloatTensor([1.0]).detach()
    x2=torch.FloatTensor([2.0]).detach()
    x3=Variable(torch.FloatTensor([3.0]),requires_grad=True)
    loss1 = x1*x3+x1
    loss2 = x2*x3+x2
    loss=loss1+0.5*loss2
    # y1 = torch.nn.functional.gumbel_softmax(x1,1,hard=False)
    # y2 = torch.nn.functional.softmax(x2,dim=-1)
    # y3 = torch.zeros(4)
    # y3[torch.argmax(y2)]=1
    # y4 = y3-y2.detach()+y2
    # loss1 = torch.dot(x3,y1)
    # loss2 = torch.dot(x3,y4)
    # loss1.backward()
    # loss2.backward()
    loss.backward()

    # a= torch.exp(x1[0])
    
    # b= torch.exp(x1[1])
    # print(x1.grad)
    # print(x2.grad)
    print(x3.grad)
    # print(y1)
    # print(y2)


