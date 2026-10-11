import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class SingleHeadSiameseAttention(nn.Module):
    # 在计算support特征编码时，不同类别之间的信息会有交互，因为此时class类不再是batch维度
    # 而计算query和support时，不同类别之间的信息会有交互，因为此时class类不再是batch维度
    # 计算特征编码(这里是下一层的输入，不是category向量)时，
    # (类别数为batch_size*441*256) @ (类别数为batch_size*256*5) => （441*5）@（5*256）=> out(类别数为batch_size,441,256)送入下一层
    # 计算query和support时，query的batch为batch_size
    """ Single-Head Attention Module. Weights for Q and K are shared in a Siamese manner. No proj weights for V."""
    def __init__(self, d_model):
        super().__init__()
        self.n_head = 1
        self.d_model = d_model
        self.w_qk = nn.Linear(self.d_model, self.n_head * self.d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=np.power(self.d_model, 0.5))
        nn.init.normal_(self.w_qk.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_model)))

        self.dummy = nn.Parameter(torch.Tensor(1, self.d_model))
        nn.init.normal_(self.dummy)

        self.linear1 = nn.Sequential(nn.Linear(self.d_model, self.d_model // 2), nn.ReLU(inplace=True))
        self.linear2 = nn.Sequential(nn.Linear(self.d_model, self.d_model // 2), nn.ReLU(inplace=True))
        self.linear3 = nn.Linear(self.d_model * 2, self.d_model)

    def forward(self, q, k, v, tsp, supp_class_id):
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        # tsp为[batch_size,eposide_size, 256]
        sz_b, len_tsp, _ = tsp.size()

        residual = q
        q = self.w_qk(q).view(sz_b, len_q, self.n_head, self.d_model)
        k = self.w_qk(k).view(sz_b, len_k, self.n_head, self.d_model)
        v = v.view(sz_b, len_v, self.n_head, self.d_model)

        # tsp = tsp.view(sz_b, len_v, self.n_head, self.d_model)
        # 这里要注意，batch_size要与qkv一致，而len_tsp则是自己的
        # 因为我们引入了背景信息，而tsp仍然保持为eposide(5)的设置，因为在计算交叉注意力时仍有五个tsp
        # 比如在支持类提取时，有两个正常类，三个背景类，则batch=2，此时qkv输入为[2,2,256],tsp输入为[2,5,256]
        # qkv在上面view成了[2,2,1,256],所以这里tsp也要变成[2,5,1,256]才行
        # k和v在下面均会变为[2,5,1,256],则tsp也要变成[2,5,1,256]才行
        tsp = tsp.view(sz_b, len_tsp, self.n_head, self.d_model)

        # 因为类别提取和正向传播时不同类别之间均要交互，所以拓展batch_size维度
        dummy = self.dummy.reshape(1, 1, 1, self.d_model).expand(sz_b, -1, self.n_head, -1)
        dummy_v = torch.zeros(sz_b, 1, self.n_head, self.d_model, device=v.device)

        # 创建迭代器
        k_iter = iter(k.split(1, dim=1))  # 将 k 按 class 维度拆分成多个形状为 [BZ, 1, n_head, 256] 的张量
        list_k = [next(k_iter) if x != 100 else dummy for x in supp_class_id]
        # 假如batch为2，则k_cat为[2,5,1,256](support提取和正向传播均适用)
        k_cat = torch.cat(list_k, dim=1)
        v_iter = iter(v.split(1, dim=1))  # 将 k 按 class 维度拆分成多个形状为 [BZ, 1, n_head, 256] 的张量
        list_v = [next(v_iter) if x != 100 else dummy_v for x in supp_class_id]
        v_cat = torch.cat(list_v, dim=1)

        # k = torch.cat([k, dummy], dim=1)
        # v = torch.cat([v, dummy_v], dim=1)
        # tsp = torch.cat([tsp, dummy_v], dim=1)
        k = torch.cat([k_cat, dummy], dim=1)
        v = torch.cat([v_cat, dummy_v], dim=1)
        tsp = torch.cat([tsp, dummy_v], dim=1)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_model)  # (n_head * b) x lq x d_model
        # 因为batch不为定值，故要用len_tsp(与tsp一致)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_tsp + 1, self.d_model)  # (n_head * b) x lk x d_model
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_tsp + 1, self.d_model)  # (n_head * b) x lv x d_model
        tsp = tsp.permute(2, 0, 1, 3).contiguous().view(-1, len_tsp + 1, self.d_model)  # (n_head * b) x lv x d_model

        # k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k + 1, self.d_model)  # (n_head * b) x lk x d_model
        # v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v + 1, self.d_model)  # (n_head * b) x lv x d_model
        # tsp = tsp.permute(2, 0, 1, 3).contiguous().view(-1, len_v + 1, self.d_model)  # (n_head * b) x lv x d_model

        output, attn, log_attn = self.attention(q, k, v)
        tsp, _, _ = self.attention(q, k, tsp)

        output = output.view(self.n_head, sz_b, len_q, self.d_model)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n_head * d_model)

        tsp = tsp.view(self.n_head, sz_b, len_q, self.d_model)
        tsp = tsp.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n_head * d_model)

        output1 = self.linear1(output * residual)
        output2 = self.linear2(residual - output)
        output = self.linear3(
            torch.cat([output1, output2, residual], dim=2)
        )

        return output, tsp
    
    # def forward(self, q, k, v, tsp, supp_class_id):
    #     sz_b, len_q, _ = q.size()
    #     sz_b, len_k, _ = k.size()
    #     sz_b, len_v, _ = v.size()

    #     # tsp为[batch_size,eposide_size, 256]
    #     sz_b, len_tsp, _ = tsp.size()

    #     residual = q
    #     q = self.w_qk(q).view(sz_b, len_q, self.n_head, self.d_model)
    #     k = self.w_qk(k).view(sz_b, len_k, self.n_head, self.d_model)
    #     v = v.view(sz_b, len_v, self.n_head, self.d_model)

    #     # tsp = tsp.view(sz_b, len_v, self.n_head, self.d_model)
    #     # 这里要注意，batch_size要与qkv一致，而len_tsp则是自己的
    #     # 因为我们引入了背景信息，而tsp仍然保持为eposide(5)的设置，因为在计算交叉注意力时仍有五个tsp
    #     # 比如在支持类提取时，有两个正常类，三个背景类，则batch=2，此时qkv输入为[2,2,256],tsp输入为[2,5,256]
    #     # qkv在上面view成了[2,2,1,256],所以这里tsp也要变成[2,5,1,256]才行
    #     # k和v在下面均会变为[2,5,1,256],则tsp也要变成[2,5,1,256]才行
    #     tsp = tsp.view(sz_b, len_tsp, self.n_head, self.d_model)

    #     # 因为类别提取和正向传播时不同类别之间均要交互，所以拓展batch_size维度
    #     dummy = self.dummy.reshape(1, 1, 1, self.d_model).expand(sz_b, -1, self.n_head, -1)
    #     dummy_v = torch.zeros(sz_b, 1, self.n_head, self.d_model, device=v.device)

    #     # 创建迭代器
    #     k_iter = iter(k.split(1, dim=1))  # 将 k 按 class 维度拆分成多个形状为 [BZ, 1, n_head, 256] 的张量
    #     list_k = [next(k_iter) if x != 100 else dummy for x in supp_class_id]
    #     # 假如batch为2，则k_cat为[2,5,1,256](support提取和正向传播均适用)
    #     k_cat = torch.cat(list_k, dim=1)
    #     v_iter = iter(v.split(1, dim=1))  # 将 k 按 class 维度拆分成多个形状为 [BZ, 1, n_head, 256] 的张量
    #     list_v = [next(v_iter) if x != 100 else dummy_v for x in supp_class_id]
    #     v_cat = torch.cat(list_v, dim=1)

    #     # k = torch.cat([k, dummy], dim=1)
    #     # v = torch.cat([v, dummy_v], dim=1)
    #     # tsp = torch.cat([tsp, dummy_v], dim=1)
    #     # k = torch.cat([k_cat, dummy], dim=1)
    #     # v = torch.cat([v_cat, dummy_v], dim=1)
    #     # tsp = torch.cat([tsp, dummy_v], dim=1)
    #     k = k_cat
    #     v = v_cat

    #     q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_model)  # (n_head * b) x lq x d_model
    #     # 因为batch不为定值，故要用len_tsp(与tsp一致)
    #     k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_tsp, self.d_model)  # (n_head * b) x lk x d_model
    #     v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_tsp, self.d_model)  # (n_head * b) x lv x d_model
    #     tsp = tsp.permute(2, 0, 1, 3).contiguous().view(-1, len_tsp, self.d_model)  # (n_head * b) x lv x d_model

    #     # k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k + 1, self.d_model)  # (n_head * b) x lk x d_model
    #     # v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v + 1, self.d_model)  # (n_head * b) x lv x d_model
    #     # tsp = tsp.permute(2, 0, 1, 3).contiguous().view(-1, len_v + 1, self.d_model)  # (n_head * b) x lv x d_model

    #     output, attn, log_attn = self.attention(q, k, v)
    #     tsp, _, _ = self.attention(q, k, tsp)

    #     output = output.view(self.n_head, sz_b, len_q, self.d_model)
    #     output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n_head * d_model)

    #     tsp = tsp.view(self.n_head, sz_b, len_q, self.d_model)
    #     tsp = tsp.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n_head * d_model)

    #     output1 = self.linear1(output * residual)
    #     output2 = self.linear2(residual - output)
    #     output = self.linear3(
    #         torch.cat([output1, output2, residual], dim=2)
    #     )

    #     return output, tsp
    