import torch
import torch.nn.functional as F
from torch import nn
import math
from functools import partial
from torchmetrics.classification import BinaryJaccardIndex, F1Score, BinaryPrecisionRecallCurve

import lightning.pytorch as pl
from sam.segment_anything.modeling.image_encoder import ImageEncoderViT
from sam.segment_anything.modeling.mask_decoder import MaskDecoder
from sam.segment_anything.modeling.prompt_encoder import PromptEncoder
from sam.segment_anything.modeling.transformer import TwoWayTransformer
from sam.segment_anything.modeling.common import LayerNorm2d

import torchvision


class BilinearSampler(nn.Module):
    def __init__(self, config):
        super(BilinearSampler, self).__init__()
        self.config = config

    def forward(self, feature_maps, sample_points):
        B, D, H, W = feature_maps.shape
        _, N_points, _ = sample_points.shape

        sample_points = (sample_points / self.config.PATCH_SIZE) * 2.0 - 1.0
        sample_points = sample_points.unsqueeze(2)  # [B, N, 1, 2]

        sampled_features = F.grid_sample(feature_maps, sample_points, mode='bilinear', align_corners=False)
        sampled_features = sampled_features.squeeze(dim=-1).permute(0, 2, 1)  # [B, N, D]
        return sampled_features


class RegionSampler(nn.Module):
    def __init__(self, config, grid_size=(4, 8), width_scale=0.5):
        super().__init__()
        self.config = config
        self.H_grid, self.W_grid = grid_size
        self.width_scale = width_scale

    def forward(self, feature_maps, src_points, tgt_points):
        B, D, H, W = feature_maps.shape
        _, N_pairs, _ = src_points.shape

        vector = tgt_points - src_points
        mid_point = (src_points + tgt_points) / 2.0

        normal = torch.stack([-vector[:, :, 1], vector[:, :, 0]], dim=-1)

        u = torch.linspace(-1.2, 1.2, self.W_grid, device=src_points.device)
        v = torch.linspace(-1.0, 1.0, self.H_grid, device=src_points.device)
        grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')

        grid_u = grid_u.view(1, 1, self.H_grid, self.W_grid)
        grid_v = grid_v.view(1, 1, self.H_grid, self.W_grid)

        vec_expanded = vector.view(B, N_pairs, 1, 1, 2)
        norm_expanded = normal.view(B, N_pairs, 1, 1, 2)
        mid_expanded = mid_point.view(B, N_pairs, 1, 1, 2)

        sample_grid = mid_expanded + \
                      grid_u.unsqueeze(-1) * (vec_expanded / 2.0) + \
                      grid_v.unsqueeze(-1) * (norm_expanded / 2.0 * self.width_scale)

        sample_grid = sample_grid.view(B * N_pairs, self.H_grid, self.W_grid, 2)
        sample_grid_norm = (sample_grid / self.config.PATCH_SIZE) * 2.0 - 1.0

        total_pixels = N_pairs * self.H_grid * self.W_grid
        flat_grid = sample_grid_norm.view(B, total_pixels, 1, 2)

        sampled_raw = F.grid_sample(feature_maps, flat_grid, mode='bilinear', align_corners=False)

        # Output: [B, N_pairs, D, H, W] - Dims are explicitly permuted here
        sampled_features = sampled_raw.squeeze(-1).view(B, D, N_pairs, self.H_grid, self.W_grid).permute(0, 2, 1, 3, 4).contiguous()
        return sampled_features


class TopoNet(nn.Module):
    def __init__(self, config, feature_dim, region_channels=32):
        super(TopoNet, self).__init__()
        self.config = config

        self.hidden_dim = 128
        self.heads = 4
        self.num_attn_layers = 3

        self.feature_proj = nn.Linear(feature_dim, self.hidden_dim)

        self.region_cnn = nn.Sequential(
            nn.Conv2d(region_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.region_proj = nn.Linear(128, self.hidden_dim)

        self.pair_proj = nn.Linear(3 * self.hidden_dim + 2, self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        if self.config.TOPONET_VERSION != 'no_transformer':
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_attn_layers)
        self.output_proj = nn.Linear(self.hidden_dim, 1)

    def forward(self, points, point_features, region_features, pairs, pairs_valid):
        point_features = F.relu(self.feature_proj(point_features))

        B, N_samples, N_pairs, D, Hg, Wg = region_features.shape
        region_flat = region_features.reshape(B * N_samples * N_pairs, D, Hg, Wg)

        region_emb = self.region_cnn(region_flat)
        region_emb = F.relu(self.region_proj(region_emb))
        region_emb = region_emb.view(B, N_samples * N_pairs, -1)

        batch_size = B
        pairs_view = pairs.view(batch_size, -1, 2)
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, N_samples * N_pairs).to(pairs.device)

        src_features = point_features[batch_indices, pairs_view[:, :, 0]]
        tgt_features = point_features[batch_indices, pairs_view[:, :, 1]]

        src_points = points[batch_indices, pairs_view[:, :, 0]]
        tgt_points = points[batch_indices, pairs_view[:, :, 1]]
        offset = tgt_points - src_points

        pair_features = torch.concat([src_features, tgt_features, region_emb, offset], dim=2)
        pair_features = F.relu(self.pair_proj(pair_features))

        pair_features = pair_features.view(batch_size * N_samples, N_pairs, -1)
        pairs_valid = pairs_valid.view(batch_size * N_samples, N_pairs)
        all_invalid_pair_mask = torch.eq(torch.sum(pairs_valid, dim=-1), 0).unsqueeze(-1)
        pairs_valid = torch.logical_or(pairs_valid, all_invalid_pair_mask)
        padding_mask = ~pairs_valid

        if self.config.TOPONET_VERSION != 'no_transformer':
            pair_features = self.transformer_encoder(pair_features, src_key_padding_mask=padding_mask)

        _, n_pairs_out, _ = pair_features.shape
        pair_features = pair_features.view(batch_size, N_samples, n_pairs_out, -1)

        logits = self.output_proj(pair_features)
        scores = torch.sigmoid(logits)

        return logits, scores


class _LoRA_qkv(nn.Module):
    def __init__(self, qkv: nn.Module, linear_a_q: nn.Module, linear_b_q: nn.Module, linear_a_v: nn.Module,
                 linear_b_v: nn.Module):
        super().__init__()
        self.weight = qkv.weight
        self.bias = qkv.bias
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = F.linear(x, self.weight, self.bias)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


class SAMRoad(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ... (Encoder Init) ...
        assert config.SAM_VERSION in {'vit_b', 'vit_l', 'vit_h'}
        if config.SAM_VERSION == 'vit_b':
            encoder_embed_dim = 768
            encoder_depth = 12
            encoder_num_heads = 12
            encoder_global_attn_indexes = [2, 5, 8, 11]
        elif config.SAM_VERSION == 'vit_l':
            encoder_embed_dim = 1024
            encoder_depth = 24
            encoder_num_heads = 16
            encoder_global_attn_indexes = [5, 11, 17, 23]
        elif config.SAM_VERSION == 'vit_h':
            encoder_embed_dim = 1280
            encoder_depth = 32
            encoder_num_heads = 16
            encoder_global_attn_indexes = [7, 15, 23, 31]

        prompt_embed_dim = 256
        image_size = config.PATCH_SIZE
        self.image_size = image_size
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size
        encoder_output_dim = prompt_embed_dim

        self.encoder_output_dim = encoder_output_dim

        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)

        if self.config.NO_SAM:
            raise NotImplementedError()
        else:
            self.image_encoder = ImageEncoderViT(
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim
            )

        if self.config.USE_SAM_DECODER:
            self.prompt_encoder = PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            )
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
            self.mask_decoder = MaskDecoder(
                num_multimask_outputs=2,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            )
        else:
            activation = nn.GELU
            self.map_decoder = nn.Sequential(
                nn.ConvTranspose2d(encoder_output_dim, 128, kernel_size=2, stride=2),
                LayerNorm2d(128),
                activation(),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                activation(),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                activation(),
                nn.ConvTranspose2d(32, 2, kernel_size=2, stride=2),
            )

        # 注册 Hook
        self.intermediate_features = {}

        def get_activation(name):
            def hook(model, input, output):
                B, H, W, C = output.shape
                self.intermediate_features[name] = output.permute(0, 3, 1, 2).view(B, C, H, W)
            return hook

        target_layer_idx = encoder_depth - 4
        self.image_encoder.blocks[target_layer_idx].register_forward_hook(get_activation('inter_layer'))

        # 中间层投影器
        self.inter_projector = nn.Sequential(
            nn.Conv2d(encoder_embed_dim, 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.GELU()
        )

        # Topo 分支：压缩 + 残差增强 (替换 fusion_conv_topo)
        self.topo_compressor = nn.Sequential(
            nn.Conv2d(encoder_output_dim * 2, encoder_output_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, encoder_output_dim),
            nn.GELU()
        )

        self.topo_res_block = nn.Sequential(
            nn.Conv2d(encoder_output_dim, encoder_output_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, encoder_output_dim),
            nn.GELU(),
            nn.Conv2d(encoder_output_dim, encoder_output_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, encoder_output_dim)
        )

        # 区域特征压缩器
        self.region_channels = 32
        self.region_compressor = nn.Sequential(
            nn.Conv2d(encoder_output_dim, self.region_channels, kernel_size=1, bias=False),
            LayerNorm2d(self.region_channels),
            nn.GELU()
        )

        # 计算总区域通道数
        self.mask_channels = 2
        self.total_region_channels = self.region_channels + self.mask_channels

        self.bilinear_sampler = BilinearSampler(config)
        self.region_sampler = RegionSampler(config, grid_size=(4, 8))
        self.topo_net = TopoNet(config, encoder_output_dim, region_channels=self.total_region_channels)

        #### LORA
        if config.ENCODER_LORA:
            r = self.config.LORA_RANK
            lora_layer_selection = None
            assert r > 0
            if lora_layer_selection:
                self.lora_layer_selection = lora_layer_selection
            else:
                self.lora_layer_selection = list(range(len(self.image_encoder.blocks)))
            self.w_As = []
            self.w_Bs = []
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for t_layer_i, blk in enumerate(self.image_encoder.blocks):
                if t_layer_i not in self.lora_layer_selection:
                    continue
                w_qkv_linear = blk.attn.qkv
                dim = w_qkv_linear.in_features
                w_a_linear_q = nn.Linear(dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, dim, bias=False)
                w_a_linear_v = nn.Linear(dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, dim, bias=False)
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)
                blk.attn.qkv = _LoRA_qkv(w_qkv_linear, w_a_linear_q, w_b_linear_q, w_a_linear_v, w_b_linear_v)
            for w_A in self.w_As:
                nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
            for w_B in self.w_Bs:
                nn.init.zeros_(w_B.weight)

        #### Losses
        if self.config.FOCAL_LOSS:
            self.mask_criterion = partial(torchvision.ops.sigmoid_focal_loss, reduction='mean')
        else:
            self.mask_criterion = torch.nn.BCEWithLogitsLoss()
        self.topo_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        #### Metrics
        self.keypoint_iou = BinaryJaccardIndex(threshold=0.5)
        self.road_iou = BinaryJaccardIndex(threshold=0.5)
        self.topo_f1 = F1Score(task='binary', threshold=0.5, ignore_index=-1)
        self.keypoint_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)
        self.road_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)
        self.topo_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)

        if self.config.NO_SAM:
            return
        with open(config.SAM_CKPT_PATH, "rb") as f:
            ckpt_state_dict = torch.load(f)
            if image_size != 1024:
                new_state_dict = self.resize_sam_pos_embed(ckpt_state_dict, image_size, vit_patch_size,
                                                           encoder_global_attn_indexes)
                ckpt_state_dict = new_state_dict

            matched_names = []
            state_dict_to_load = {}
            for k, v in self.named_parameters():
                if k in ckpt_state_dict and v.shape == ckpt_state_dict[k].shape:
                    matched_names.append(k)
                    state_dict_to_load[k] = ckpt_state_dict[k]

            self.matched_param_names = set(matched_names)
            self.load_state_dict(state_dict_to_load, strict=False)

    def resize_sam_pos_embed(self, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
        new_state_dict = {k: v for k, v in state_dict.items()}
        pos_embed = new_state_dict['image_encoder.pos_embed']
        token_size = int(image_size // vit_patch_size)
        if pos_embed.shape[1] != token_size:
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1)
            new_state_dict['image_encoder.pos_embed'] = pos_embed
            rel_pos_keys = [k for k in state_dict.keys() if 'rel_pos' in k]
            global_rel_pos_keys = [k for k in rel_pos_keys if any([str(i) in k for i in encoder_global_attn_indexes])]
            for k in global_rel_pos_keys:
                rel_pos_params = new_state_dict[k]
                h, w = rel_pos_params.shape
                rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
                rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear',
                                               align_corners=False)
                new_state_dict[k] = rel_pos_params[0, 0, ...]
        return new_state_dict

    def forward(self, rgb, graph_points, pairs, valid):
        x = rgb.permute(0, 3, 1, 2)
        x = (x - self.pixel_mean) / self.pixel_std

        # 1. Encoder & Hook
        final_embedding = self.image_encoder(x)

        mask_input_embedding = final_embedding

        inter_feat = self.intermediate_features['inter_layer']
        inter_feat_proj = self.inter_projector(inter_feat)

        # 拼接
        cat_feat_topo = torch.cat([final_embedding, inter_feat_proj], dim=1)  # [B, 512, H, W]

        # 压缩 + 残差
        topo_feat = self.topo_compressor(cat_feat_topo)
        topo_feat = topo_feat + self.topo_res_block(topo_feat)
        topo_input_embedding = F.gelu(topo_feat)

        B, N_points, _ = graph_points.shape
        _, N_samples, N_pairs, _ = pairs.shape
        pairs_flat = pairs.view(B, N_samples * N_pairs, 2)
        src_indices = pairs_flat[:, :, 0]
        tgt_indices = pairs_flat[:, :, 1]
        batch_indices = torch.arange(B, device=graph_points.device).view(B, 1).expand(B, N_samples * N_pairs)

        src_coords = graph_points[batch_indices, src_indices]
        tgt_coords = graph_points[batch_indices, tgt_indices]

        # 1. Mask Decoder
        image_embeddings = mask_input_embedding
        if self.config.USE_SAM_DECODER:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            low_res_logits, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True
            )
            mask_logits = F.interpolate(
                low_res_logits,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
            mask_scores = torch.sigmoid(mask_logits)
        else:
            mask_logits = self.map_decoder(image_embeddings)
            mask_scores = torch.sigmoid(mask_logits)

        point_features = self.bilinear_sampler(topo_input_embedding, graph_points)

        compressed_embeddings = self.region_compressor(topo_input_embedding)

        mask_guidance = torch.sigmoid(mask_logits).detach()

        target_size = compressed_embeddings.shape[-2:]
        mask_guidance_down = F.interpolate(
            mask_guidance,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        topo_combined_input = torch.cat([compressed_embeddings, mask_guidance_down], dim=1)

        region_features_flat = self.region_sampler(topo_combined_input, src_coords, tgt_coords)

        region_features = region_features_flat.reshape(
            B, N_samples, N_pairs, self.total_region_channels, 4, 8
        )

        topo_logits, topo_scores = self.topo_net(graph_points, point_features, region_features, pairs, valid)

        mask_logits = mask_logits.permute(0, 2, 3, 1)
        mask_scores = mask_scores.permute(0, 2, 3, 1)
        return mask_logits, mask_scores, topo_logits, topo_scores

    def infer_masks_and_img_features(self, rgb):
        x = rgb.permute(0, 3, 1, 2)
        x = (x - self.pixel_mean) / self.pixel_std

        # 1. Backbone
        final_embedding = self.image_encoder(x)

        # === 路径 A: Mask 任务 (极简) ===
        mask_input_embedding = final_embedding

        # === 路径 B: Topo 任务 (增强) ===
        inter_feat = self.intermediate_features['inter_layer']
        inter_feat_proj = self.inter_projector(inter_feat)
        cat_feat_topo = torch.cat([final_embedding, inter_feat_proj], dim=1)

        # Compress + Residual
        topo_feat = self.topo_compressor(cat_feat_topo)
        topo_feat = topo_feat + self.topo_res_block(topo_feat)
        topo_input_embedding = F.gelu(topo_feat)

        # 3. Mask Decoder
        if self.config.USE_SAM_DECODER:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            low_res_logits, iou_predictions = self.mask_decoder(
                image_embeddings=mask_input_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True
            )
            mask_logits = F.interpolate(
                low_res_logits,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
            mask_scores = torch.sigmoid(mask_logits)
        else:
            mask_logits = self.map_decoder(mask_input_embedding)
            mask_scores = torch.sigmoid(mask_logits)

        compressed_embeddings = self.region_compressor(topo_input_embedding)

        target_size = compressed_embeddings.shape[-2:]
        mask_guidance = torch.sigmoid(mask_logits).detach()
        mask_guidance_down = F.interpolate(
            mask_guidance,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        topo_combined_features = torch.cat([topo_input_embedding, mask_guidance_down], dim=1)

        mask_scores = mask_scores.permute(0, 2, 3, 1)

        return mask_scores, topo_combined_features

    def infer_toponet(self, topo_combined_features, graph_points, pairs, valid):
        topo_visual_features = topo_combined_features[:, :self.encoder_output_dim, :, :]  # [B, 256, H, W]
        mask_guidance_down = topo_combined_features[:, self.encoder_output_dim:, :, :]  # [B, 2, H, W]

        B, N_samples, N_pairs, _ = pairs.shape
        pairs_flat = pairs.view(B, N_samples * N_pairs, 2)
        src_indices = pairs_flat[:, :, 0].long()
        tgt_indices = pairs_flat[:, :, 1].long()
        batch_indices = torch.arange(B, device=graph_points.device).view(B, 1).expand(B, N_samples * N_pairs)
        src_coords = graph_points[batch_indices, src_indices]
        tgt_coords = graph_points[batch_indices, tgt_indices]

        point_features = self.bilinear_sampler(topo_visual_features, graph_points)

        compressed_feat = self.region_compressor(topo_visual_features)
        topo_region_input = torch.cat([compressed_feat, mask_guidance_down], dim=1)
        region_features_flat = self.region_sampler(topo_region_input, src_coords, tgt_coords)

        region_features = region_features_flat.reshape(
            B, N_samples, N_pairs, self.total_region_channels, 4, 8
        )

        topo_logits, topo_scores = self.topo_net(graph_points, point_features, region_features, pairs, valid)

        return topo_scores

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")

    def training_step(self, batch, batch_idx):
        rgb, keypoint_mask, road_mask = batch['rgb'], batch['keypoint_mask'], batch['road_mask']
        graph_points, pairs, valid = batch['graph_points'], batch['pairs'], batch['valid']

        mask_logits, mask_scores, topo_logits, topo_scores = self(rgb, graph_points, pairs, valid)

        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)
        mask_loss = self.mask_criterion(mask_logits, gt_masks)

        topo_gt, topo_loss_mask = batch['connected'].to(torch.int32), valid.to(torch.float32)
        topo_loss = self.topo_criterion(topo_logits, topo_gt.unsqueeze(-1).to(torch.float32))

        if torch.isnan(topo_loss).any():
            print("NaN detected in topo loss")

        topo_loss *= topo_loss_mask.unsqueeze(-1)
        topo_loss = topo_loss.sum() / topo_loss_mask.sum()

        loss = mask_loss + topo_loss
        self.log('train_mask_loss', mask_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_topo_loss', topo_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        rgb, keypoint_mask, road_mask = batch['rgb'], batch['keypoint_mask'], batch['road_mask']
        graph_points, pairs, valid = batch['graph_points'], batch['pairs'], batch['valid']
        mask_logits, mask_scores, topo_logits, topo_scores = self(rgb, graph_points, pairs, valid)
        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)
        mask_loss = self.mask_criterion(mask_logits, gt_masks)
        topo_gt, topo_loss_mask = batch['connected'].to(torch.int32), valid.to(torch.float32)
        topo_loss = self.topo_criterion(topo_logits, topo_gt.unsqueeze(-1).to(torch.float32))
        topo_loss *= topo_loss_mask.unsqueeze(-1)
        topo_loss = topo_loss.sum() / topo_loss_mask.sum()
        loss = mask_loss + topo_loss
        self.log('val_mask_loss', mask_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_topo_loss', topo_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        self.keypoint_iou.update(mask_scores[..., 0], keypoint_mask)
        self.road_iou.update(mask_scores[..., 1], road_mask)
        valid = valid.to(torch.int32)
        topo_gt = (1 - valid) * -1 + valid * topo_gt
        self.topo_f1.update(topo_scores, topo_gt.unsqueeze(-1))

    def on_validation_epoch_end(self):
        keypoint_iou = self.keypoint_iou.compute()
        road_iou = self.road_iou.compute()
        topo_f1 = self.topo_f1.compute()
        self.log("keypoint_iou", keypoint_iou)
        self.log("road_iou", road_iou)
        self.log("topo_f1", topo_f1)
        self.keypoint_iou.reset()
        self.road_iou.reset()
        self.topo_f1.reset()

    def test_step(self, batch, batch_idx):
        rgb, keypoint_mask, road_mask = batch['rgb'], batch['keypoint_mask'], batch['road_mask']
        graph_points, pairs, valid = batch['graph_points'], batch['pairs'], batch['valid']
        mask_logits, mask_scores, topo_logits, topo_scores = self(rgb, graph_points, pairs, valid)
        topo_gt, topo_loss_mask = batch['connected'].to(torch.int32), valid.to(torch.float32)
        self.keypoint_pr_curve.update(mask_scores[..., 0], keypoint_mask.to(torch.int32))
        self.road_pr_curve.update(mask_scores[..., 1], road_mask.to(torch.int32))
        valid = valid.to(torch.int32)
        topo_gt = (1 - valid) * -1 + valid * topo_gt
        self.topo_pr_curve.update(topo_scores, topo_gt.unsqueeze(-1).to(torch.int32))

    def on_test_end(self):
        def find_best_threshold(pr_curve_metric, category):
            print(f'======= {category} ======')
            precision, recall, thresholds = pr_curve_metric.compute()
            f1_scores = 2 * (precision * recall) / (precision + recall)
            best_threshold_index = torch.argmax(f1_scores)
            best_threshold = thresholds[best_threshold_index]
            best_precision = precision[best_threshold_index]
            best_recall = recall[best_threshold_index]
            best_f1 = f1_scores[best_threshold_index]
            print(f'Best threshold {best_threshold}, P={best_precision} R={best_recall} F1={best_f1}')

        print('======= Finding best thresholds ======')
        find_best_threshold(self.keypoint_pr_curve, 'keypoint')
        find_best_threshold(self.road_pr_curve, 'road')
        find_best_threshold(self.topo_pr_curve, 'topo')

    def configure_optimizers(self):
        layer_decay = 0.8
        param_groups = []

        if not self.config.FREEZE_ENCODER:
            num_layers = len(self.image_encoder.blocks) + 1
            embed_params = {
                'params': [p for n, p in self.image_encoder.patch_embed.named_parameters()],
                'lr': self.config.BASE_LR * self.config.ENCODER_LR_FACTOR * (layer_decay ** num_layers)
            }
            param_groups.append(embed_params)
            for i, block in enumerate(self.image_encoder.blocks):
                layer_lr = self.config.BASE_LR * self.config.ENCODER_LR_FACTOR * (layer_decay ** (num_layers - 1 - i))
                block_params = {
                    'params': [p for n, p in block.named_parameters() if 'qkv.linear' not in n],
                    'lr': layer_lr
                }
                param_groups.append(block_params)
            final_params = {
                'params': [p for n, p in self.image_encoder.named_parameters()
                           if 'blocks' not in n and 'patch_embed' not in n],
                'lr': self.config.BASE_LR * self.config.ENCODER_LR_FACTOR
            }
            param_groups.append(final_params)

        if self.config.ENCODER_LORA:
            lora_params = {
                'params': [p for k, p in self.image_encoder.named_parameters() if 'qkv.linear_' in k],
                'lr': self.config.BASE_LR
            }
            param_groups.append(lora_params)

        fusion_params = {
            'params': list(self.inter_projector.parameters()) +
                      list(self.topo_compressor.parameters()) +
                      list(self.topo_res_block.parameters()) +
                      list(self.region_compressor.parameters()),
            'lr': self.config.BASE_LR
        }
        param_groups.append(fusion_params)

        if self.config.USE_SAM_DECODER:
            matched_decoder_params = {
                'params': [p for k, p in self.mask_decoder.named_parameters() if
                           'mask_decoder.' + k in self.matched_param_names],
                'lr': self.config.BASE_LR * 0.1
            }
            fresh_decoder_params = {
                'params': [p for k, p in self.mask_decoder.named_parameters() if
                           'mask_decoder.' + k not in self.matched_param_names],
                'lr': self.config.BASE_LR
            }
            param_groups.append(matched_decoder_params)
            param_groups.append(fresh_decoder_params)
        else:
            decoder_params = {
                'params': [p for p in self.map_decoder.parameters()],
                'lr': self.config.BASE_LR
            }
            param_groups.append(decoder_params)

        topo_net_params = {
            'params': [p for p in self.topo_net.parameters()],
            'lr': self.config.BASE_LR
        }
        param_groups.append(topo_net_params)

        optimizer = torch.optim.Adam(param_groups)
        step_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9, ], gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': step_lr}
