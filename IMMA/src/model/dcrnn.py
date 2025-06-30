import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from scipy.sparse import linalg
import pickle

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_graph_data(pkl_filename):
    try:
        # METRLA and PEMSBAY
        _, _, adj_mx = load_pickle(pkl_filename)
    except ValueError:
        # PEMS3478
        adj_mx = load_pickle(pkl_filename)
    return adj_mx


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    # 如果是torch tensor，转换为numpy
    if hasattr(adj, 'cpu'):
        adj = adj.cpu().numpy()
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    # 如果是torch tensor，转换为numpy
    if hasattr(adj_mx, 'cpu'):
        adj_mx = adj_mx.cpu().numpy()
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    # 如果是torch tensor，转换为numpy
    if hasattr(adj_mx, 'cpu'):
        adj_mx = adj_mx.cpu().numpy()
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)

class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


class DCGRUCell(torch.nn.Module):
    def __init__(self, num_units, max_diffusion_step, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True, device=None):
        """

        :param num_units:
        :param max_diffusion_step:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru
        self._filter_type = filter_type
        self._device = device

        self._fc_params = LayerParams(self, 'fc')
        self._gconv_params = LayerParams(self, 'gconv')

    def _build_supports(self, adj_mx):
        """动态构建supports"""
        supports = []
        if self._filter_type == "laplacian":
            supports.append(calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif self._filter_type == "random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
        elif self._filter_type == "dual_random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
            supports.append(calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(calculate_scaled_laplacian(adj_mx))
        
        sparse_supports = []
        for support in supports:
            sparse_supports.append(self._build_sparse_matrix(support, self._device))
        return sparse_supports

    @staticmethod
    def _build_sparse_matrix(L, device):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape).to(device)
        return L

    def forward(self, inputs, hx, adj_mx):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)
        :param adj_mx: (num_nodes, num_nodes) adjacency matrix

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        # 动态获取节点数量
        num_nodes = adj_mx.shape[0]
        
        # 动态构建supports
        supports = self._build_supports(adj_mx)
        
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self._gconv
        else:
            fn = self._fc
        value = torch.sigmoid(fn(inputs, hx, output_size, bias_start=1.0, num_nodes=num_nodes, supports=supports))
        value = torch.reshape(value, (-1, num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, num_nodes * self._num_units))
        u = torch.reshape(u, (-1, num_nodes * self._num_units))

        c = self._gconv(inputs, r * hx, self._num_units, num_nodes=num_nodes, supports=supports)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0, num_nodes=None, supports=None):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * num_nodes, -1))
        state = torch.reshape(state, (batch_size * num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value

    def _gconv(self, inputs, state, output_size, bias_start=0.0, num_nodes=None, supports=None):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, num_nodes, -1))
        state = torch.reshape(state, (batch_size, num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(supports) * self._max_diffusion_step + 1  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size)).to(x.device)
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = self._gconv_params.get_biases(output_size, bias_start).to(x.device)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, num_nodes * output_size])
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, **model_kwargs):
        self.max_diffusion_step = int(model_kwargs.get("max_diffusion_step", 1))
        self.cl_decay_steps = int(model_kwargs.get("tf_decay_steps", 500))
        self.filter_type = model_kwargs.get("filter_type", "random_walk")
        self.num_rnn_layers = int(model_kwargs.get("num_rnn_layers", 1))
        self.rnn_units = int(model_kwargs.get("rnn_units", 2))


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))  # 默认输入维度为1
        self.seq_len = int(model_kwargs.get('seq_len', 12))  # for the encoder
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step,
                       filter_type=self.filter_type, device=model_kwargs.get("device")) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj_mx, hidden_state=None):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, num_nodes * input_dim)
        :param adj_mx: adjacency matrix (num_nodes, num_nodes)
        :param hidden_state: (num_layers, batch_size, hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        num_nodes = adj_mx.shape[0]
        hidden_state_size = num_nodes * self.rnn_units
        
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, hidden_state_size),
                                       device=inputs.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj_mx)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.output_dim = int(model_kwargs.get('output_dim', 1))  # 默认输出维度为1
        self.horizon = int(model_kwargs.get('horizon', 12))  # 默认预测长度与输入序列长度相同
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step,
                       filter_type=self.filter_type, device=model_kwargs.get("device")) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj_mx, hidden_state=None):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, num_nodes * output_dim)
        :param adj_mx: adjacency matrix (num_nodes, num_nodes)
        :param hidden_state: (num_layers, batch_size, hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, num_nodes * output_dim)
                 hidden_state # shape (num_layers, batch_size, hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj_mx)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        num_nodes = adj_mx.shape[0]
        output = projected.view(-1, num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class DCRNN(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        super().__init__()
        
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.encoder_model = EncoderModel(**model_kwargs)
        self.decoder_model = DecoderModel(**model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('tf_decay_steps', 500))
        self.use_curriculum_learning = bool(model_kwargs.get('use_teacher_forcing', False))

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs, adj_mx):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param adj_mx: adjacency matrix (num_nodes, num_nodes)
        :return: encoder_hidden_state: (num_layers, batch_size, hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], adj_mx, encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, adj_mx):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, hidden_state_size)
        :param adj_mx: adjacency matrix (num_nodes, num_nodes)
        :return: output: (self.horizon, batch_size, num_nodes * output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        num_nodes = adj_mx.shape[0]
        go_symbol = torch.zeros((batch_size, num_nodes * self.decoder_model.output_dim),
                                device=encoder_hidden_state.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, adj_mx,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, history_data: torch.Tensor, adj_mx: torch.Tensor):
        """Feedforward function of DCRNN.

        Args:
            history_data (torch.Tensor): history data with shape [B, C, N, T]
            adj_mx (torch.Tensor): adjacency matrix with shape [N, N]

        Returns:
            torch.Tensor: prediction with shape [B, N, T]
        """
        # 从邻接矩阵获取节点数量
        num_nodes = adj_mx.shape[0]
        
        # reshape data
        batch_size, channels, _, seq_len = history_data.shape
        history_data = history_data.permute(0, 3, 2, 1)  # [B, T, N, C]
        history_data = history_data.reshape(batch_size, seq_len, num_nodes * channels)      # [B, T, N*C]
        history_data = history_data.transpose(0, 1)         # [T, B, N*C]

        # DCRNN
        encoder_hidden_state = self.encoder(history_data, adj_mx)
        outputs = self.decoder(encoder_hidden_state, adj_mx)      # [T, B, N*C_out]

        # reshape to B, N, T
        T, B, _ = outputs.shape
        outputs = outputs.transpose(0, 1)  # [B, T, N*C_out]
        outputs = outputs.view(B, T, num_nodes,
                               self.decoder_model.output_dim)
        outputs = outputs.squeeze(-1)  # [B, T, N]
        outputs = outputs.transpose(1, 2)  # [B, N, T]

        return outputs
    
def print_model_params(model):
    param_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("%-40s\t%-30s\t%-30s" % (name, list(param.shape), param.numel()))
            param_count += param.numel()
    print("%-40s\t%-30s" % ("Total trainable params", param_count))


if __name__ == "__main__":
    from torchinfo import summary
    
    # 创建一个示例邻接矩阵
    num_nodes = 207
    adj_mx = torch.randn(num_nodes, num_nodes)
    
    model = DCRNN(
        device=torch.device("cuda:7"),
        input_dim=1,
        output_dim=1,
        horizon=12, 
    ).to(torch.device("cuda:7"))
    
    # 创建示例数据
    history_data = torch.randn(64, 1, num_nodes, 12).to(torch.device("cuda:7"))  # [B, C, N, T]
    adj_mx = adj_mx.to(torch.device("cuda:7"))
    
    # 测试forward
    output = model(history_data, adj_mx)
    print("Output shape:", output.shape)  # 应该是 [64, 207, 12]