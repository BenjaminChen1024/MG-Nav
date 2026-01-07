import torch
import torch.nn as nn
import math
from depth_anything.depth_anything_v2.dpt import DepthAnythingV2
from geometry_encoders import create_geometry_encoder, GeometryEncoderConfig

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)  # (seq_len,)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_len)
        position_encoding = self.position_embedding(position_ids)  # (batch_size, seq_len, embed_dim)
        return position_encoding

class NavDP_RGBD_Backbone(nn.Module):
    def __init__(self,
                 image_size=224,
                 embed_size=512,
                 memory_size=8,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.memory_size = memory_size
        self.image_size = image_size
        self.embed_size = embed_size
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.rgb_model = DepthAnythingV2(**model_configs['vits'])
        self.rgb_model = self.rgb_model.pretrained.float()
        self.rgb_model.eval()
        self.preprocess_mean = torch.tensor([0.485,0.456,0.406],dtype=torch.float32)
        self.preprocess_std = torch.tensor([0.229,0.224,0.225],dtype=torch.float32)
            
        self.depth_model = DepthAnythingV2(**model_configs['vits'])
        self.depth_model = self.depth_model.pretrained.float()
        self.depth_model.train()
        self.former_query = LearnablePositionalEncoding(384,self.memory_size*16)
        self.former_pe = LearnablePositionalEncoding(384,(self.memory_size+1)*256) 
        self.former_net = nn.TransformerDecoder(nn.TransformerDecoderLayer(384,8,batch_first=True),2)
        self.project_layer = nn.Linear(384,embed_size)
        
    def forward(self,images,depths):
        with torch.no_grad():
            if len(images.shape) == 4:
                tensor_images = torch.as_tensor(images,dtype=torch.float32,device=self.device).permute(0,3,1,2)
                tensor_images = tensor_images.reshape(-1,3,self.image_size,self.image_size)
                tensor_norm_images = (tensor_images - self.preprocess_mean.reshape(1,3,1,1).to(self.device))/self.preprocess_std.reshape(1,3,1,1).to(self.device)
                image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0]
            elif len(images.shape) == 5:
                tensor_images = torch.as_tensor(images,dtype=torch.float32,device=self.device).permute(0,1,4,2,3)
                B,T,C,H,W = tensor_images.shape
                tensor_images = tensor_images.reshape(-1,3,self.image_size,self.image_size)
                tensor_norm_images = (tensor_images - self.preprocess_mean.reshape(1,3,1,1).to(self.device))/self.preprocess_std.reshape(1,3,1,1).to(self.device)
                image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0].reshape(B,T*256,-1)
            if len(depths.shape) == 4:
                tensor_depths = torch.as_tensor(depths,dtype=torch.float32,device=self.device).permute(0,3,1,2)
                tensor_depths = tensor_depths.reshape(-1,1,self.image_size,self.image_size)
                tensor_depths = torch.concat([tensor_depths,tensor_depths,tensor_depths],dim=1)
                depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0]
            elif len(depths.shape) == 5:
                tensor_depths = torch.as_tensor(depths,dtype=torch.float32,device=self.device).permute(0,1,4,2,3)
                B,T,C,H,W = tensor_depths.shape
                tensor_depths = tensor_depths.reshape(-1,1,self.image_size,self.image_size)
                tensor_depths = torch.concat([tensor_depths,tensor_depths,tensor_depths],dim=1)
                depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0].reshape(B,T*256,-1)
            former_token = torch.concat((image_token,depth_token),dim=1) + self.former_pe(torch.concat((image_token,depth_token),dim=1))
            former_query = self.former_query(torch.zeros((image_token.shape[0], self.memory_size * 16, 384),device=self.device))
            memory_token = self.former_net(former_query,former_token)
            memory_token = self.project_layer(memory_token)
            return memory_token

class NavDP_ImageGoal_Backbone(nn.Module):
    def __init__(self,
                 image_size=224,
                 embed_size=512,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.embed_size = embed_size
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.imagegoal_encoder = DepthAnythingV2(**model_configs['vits'])
        self.imagegoal_encoder = self.imagegoal_encoder.pretrained.float()
        self.imagegoal_encoder.patch_embed.proj = nn.Conv2d(in_channels=6,
                                                            out_channels = self.imagegoal_encoder.patch_embed.proj.out_channels,
                                                            kernel_size = self.imagegoal_encoder.patch_embed.proj.kernel_size,
                                                            stride = self.imagegoal_encoder.patch_embed.proj.stride,
                                                            padding = self.imagegoal_encoder.patch_embed.proj.padding)
        self.imagegoal_encoder.eval()
        self.project_layer = nn.Linear(384,embed_size)
        
    def forward(self,images):
        with torch.no_grad():
            assert len(images.shape) == 4 # B,C,H,W
            tensor_images = torch.as_tensor(images,dtype=torch.float32,device=self.device).permute(0,3,1,2)
            image_token = self.imagegoal_encoder.get_intermediate_layers(tensor_images)[0].mean(dim=1)
            image_token = self.project_layer(image_token)
            return image_token

class NavDP_PixelGoal_Backbone(nn.Module):
    def __init__(self,
                 image_size=224,
                 embed_size=512,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.embed_size = embed_size
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.pixelgoal_encoder = DepthAnythingV2(**model_configs['vits'])
        self.pixelgoal_encoder = self.pixelgoal_encoder.pretrained.float()
        self.pixelgoal_encoder.patch_embed.proj = nn.Conv2d(in_channels=4,
                                                            out_channels = self.pixelgoal_encoder.patch_embed.proj.out_channels,
                                                            kernel_size = self.pixelgoal_encoder.patch_embed.proj.kernel_size,
                                                            stride = self.pixelgoal_encoder.patch_embed.proj.stride,
                                                            padding = self.pixelgoal_encoder.patch_embed.proj.padding)
        self.pixelgoal_encoder.eval()
        self.project_layer = nn.Linear(384,embed_size)
        
    def forward(self,images):
        with torch.no_grad():
            assert len(images.shape) == 4 # B,C,H,W
            tensor_images = torch.as_tensor(images,dtype=torch.float32,device=self.device).permute(0,3,1,2)
            image_token = self.pixelgoal_encoder.get_intermediate_layers(tensor_images)[0].mean(dim=1)
            image_token = self.project_layer(image_token)
            return image_token

class GeometryFeatureMerger(nn.Module):
    """Unified merger for geometry features from different encoders.
    
    Supports different merger types:
    - "avg": Average pooling across spatial merge dimensions
    - "attention": Attention-based merger (not implemented yet)
    """
    
    def __init__(self, output_dim: int, hidden_dim: int, context_dim: int, 
                 spatial_merge_size: int = 2, temporal_merge_size: int = 2, merger_type: str = "attention", token_num: int =16):
        super().__init__()
        self.merger_type = merger_type
        self.input_dim = context_dim * temporal_merge_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.spatial_merge_size = spatial_merge_size
        self.temporal_merge_size = temporal_merge_size
        self.token_num = token_num
        
        if merger_type == "avg":
            ##⚠️ 这里根本不需要zero init，跟controlnet不一样。 如果zero_init，反而会导致不会有梯度传进来
            self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        elif merger_type == "attention":
            # Add attention-based merger for future extensibility
            raise NotImplementedError("Attention merger not implemented yet")
        else:
            raise ValueError(f"Unknown merger type: {merger_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the merger."""
        with torch.no_grad():
            batch, n_image, num_patch_h, num_patch_w, dim = x.shape
            x = x.view(
                batch, 
                n_image // self.temporal_merge_size, 
                self.temporal_merge_size, 
                num_patch_h // self.spatial_merge_size, 
                self.spatial_merge_size,
                num_patch_w // self.spatial_merge_size,
                self.spatial_merge_size,
                dim,
            )
            x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
            if self.merger_type == "avg":
                # Average pooling across spatial merge dimensions
                x = x.mean(dim=(5, 6))  # Average over the merge_size dimensions
                x = x.view(
                    batch, 
                    n_image // self.temporal_merge_size,
                    num_patch_h // self.spatial_merge_size, 
                    num_patch_w // self.spatial_merge_size,
                    self.temporal_merge_size * dim,
                )
                x = x.view(-1, self.temporal_merge_size * dim)  # Flatten for projection
                x = self.mlp(x) #(B * token_num, 384)

            else:
                raise NotImplementedError(f"Merger type {self.merger_type} not implemented")
            x = x.reshape(batch, self.token_num, -1)
        return x

    
class NavDP_Geometry_Backbone(nn.Module):
    """
    把 VGGT(geometry_encoder) 接成 NavDP 的几何分支 Backbone。
    - 输入：images，可为 (B,3,H,W) 或 (B,T,3,H,W)
    - 输出：image_token 全局向量 (B, embed_size)，与 NavDP_ImageGoal_Backbone 的接口对齐
    - 需要 token 级别特征时，可调用 encode_tokens()，返回 (B, N_tokens, embed_size)
    """
    def __init__(
        self,
        image_size = 224,
        embed_size = 512,
        device = "cuda:0",
    ):
        super().__init__()
        self.device = device

        self.image_size = image_size
        self.embed_size = embed_size

        # Create geometry encoder configuration

        encoder_config = GeometryEncoderConfig(
            encoder_type="vggt",
            model_path="",
            reference_frame="first",
            freeze_encoder=True
        )
        
        # Create geometry encoder
        self.vggt_encoder = create_geometry_encoder(
            encoder_type=encoder_config.encoder_type,
            model_path=encoder_config.model_path,
            reference_frame=encoder_config.reference_frame,
            freeze_encoder=encoder_config.freeze_encoder,
        )
        self.vggt_encoder.eval()  # 推理态

        self.token_num = 16
        self.temporal_merge_size = 2
        self.spatial_merge_size = int(math.sqrt(self.token_num))  # 16x16 -> 4x4

        self.geometry_merger = GeometryFeatureMerger(
            output_dim=384,
            hidden_dim=4096,
            context_dim=self.vggt_encoder.get_feature_dim(),
            spatial_merge_size=self.spatial_merge_size,
            temporal_merge_size=self.temporal_merge_size,
            merger_type="avg",
            token_num=self.token_num,
        )



    # --------- 公共前向：返回全局向量（与 ImageGoal_Backbone 对齐） ---------
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B,3,H,W) 或 (B,T,3,H,W)
        return: (B, embed_size) 全局几何向量
        """
        assert len(images.shape) == 4 # B,H,W,C
        B, H, W, C = images.shape
        tensor_images = torch.as_tensor(images,dtype=torch.float32,device=self.device).permute(0,3,1,2)
        geom_in = tensor_images.reshape(B, 2, 3, H, W)
        with torch.no_grad():

            batch, n_image, _, height, width = geom_in.shape

            # features (B, n_image, N_tokens, D_vggt)
            features = self.vggt_encoder(geom_in)
            features = features.reshape(batch, n_image, height // self.vggt_encoder.patch_size, width // self.vggt_encoder.patch_size, -1)

            # Merge temporal and spatial tokens
            merged_tokens = self.geometry_merger(features)  # (B, token_num, 384)
            return merged_tokens[0]


if __name__ == "__main__":
    backbone = NavDP_PixelGoal_Backbone()
    backbone = backbone.to("cuda:0")
    images = torch.rand(1,224,224,4)
    print(backbone(images).shape)
    
    backbone = NavDP_ImageGoal_Backbone()
    backbone = backbone.to("cuda:0")
    images = torch.rand(1,224,224,6)
    print(backbone(images).shape)