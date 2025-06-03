import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from .clip_module import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class BongardDatasetBLIP(Dataset):
    def __init__(
        self,
        data_root,
        data_split="seen_obj_seen_act",
        mode="test",
        base_transform=None,
        query_transform=None,
        with_annotation=False,
    ):
        self.base_transform = base_transform
        self.query_transform = query_transform
        self.data_root = data_root
        self.mode = mode
        self.with_annotation = with_annotation

        assert mode in ["val", "test", "train"]
        if mode == "train":
            data_file = f"{self.data_root}/bongard_splits/bongard_hoi_train.json"
        else:
            data_file = f"{self.data_root}/bongard_splits/bongard_hoi_{self.mode}_{data_split}.json"
        self.task_list = []
        with open(data_file, "r") as fp:
            task_items = json.load(fp)
            for task in task_items:
                task_data = {}
                pos_samples = []
                neg_samples = []
                prompts = []
                sub_classes = []
                act_classes = []
                obj_classes = []
                for sample in task[0]:
                    neg_samples.append(sample["im_path"])
                prompts.append(
                    " ".join(
                        [sample["sub_class"], sample["act_class"], sample["obj_class"]]
                    ).replace("_", " ")
                )
                sub_classes.append(sample["sub_class"].replace("_", " "))
                act_classes.append(sample["act_class"].replace("_", " "))
                obj_classes.append(sample["obj_class"].replace("_", " "))
                for sample in task[1]:
                    pos_samples.append(sample["im_path"])
                prompts.append(
                    " ".join(
                        [sample["sub_class"], sample["act_class"], sample["obj_class"]]
                    ).replace("_", " ")
                )
                sub_classes.append(sample["sub_class"].replace("_", " "))
                act_classes.append(sample["act_class"].replace("_", " "))
                obj_classes.append(sample["obj_class"].replace("_", " "))
                # random split samples into support and query images (6 vs. 1 for both pos and neg samples)
                task_data["pos_samples"] = pos_samples
                task_data["neg_samples"] = neg_samples
                task_data["annotation"] = task[-1].replace("++", " ")
                task_data["prompts"] = prompts
                task_data["subject"] = sub_classes
                task_data["object"] = obj_classes
                task_data["act"] = act_classes
                self.task_list.append(task_data)

    def __len__(self):
        return len(self.task_list)

    def load_image(self, path, transform_type="base_transform"):
        im_path = os.path.join(self.data_root, path.replace("./", ""))
        if not os.path.isfile(im_path):
            print("file not exist: {}".format(im_path))
            if "/pic/image/val" in im_path:
                im_path = im_path.replace("val", "train")
            elif "/pic/image/train" in im_path:
                im_path = im_path.replace("train", "val")
        try:
            image = Image.open(im_path).convert("RGB")
        except:
            print("File error: ", im_path)
            image = Image.open(im_path).convert("RGB")
        trans = getattr(self, transform_type)
        if trans is not None:
            image = trans(image)
        return image

    def __getitem__(self, idx):
        task = self.task_list[idx]
        pos_samples = task["pos_samples"]
        neg_samples = task["neg_samples"]

        pos_query = self.load_image(pos_samples[-1], "query_transform")
        neg_query = self.load_image(neg_samples[-1], "query_transform")

        query_images = [neg_query, pos_query]
        query_labels = torch.Tensor([1, 0]).long()

        return (
            query_images,
            query_labels,
            task["prompts"],
            "hoi",
            task["subject"],
            task["object"],
            task["act"],
            [pos_samples[-1], neg_samples[-1]],
        )


class BongardDataset(Dataset):
    def __init__(
        self,
        data_root,
        data_split="seen_obj_seen_act",
        mode="test",
        base_transform=None,
        query_transform=None,
        with_annotation=False,
    ):
        self.base_transform = base_transform
        # if query_transform is None:
        #     self.query_transform = base_transform
        # else:
        self.query_transform = query_transform
        self.data_root = data_root
        self.mode = mode
        self.with_annotation = with_annotation

        assert mode in ["val", "test"]
        data_file = f"{self.data_root}/bongard_splits/bongard_hoi_{data_split}.json"
        self.task_list = []
        with open(data_file, "r") as fp:
            task_items = json.load(fp)
            for task in task_items:
                task_data = {}
                pos_samples = []
                neg_samples = []
                for sample in task[0]:
                    neg_samples.append(sample["im_path"])
                for sample in task[1]:
                    pos_samples.append(sample["im_path"])

                # random split samples into support and query images (6 vs. 1 for both pos and neg samples)
                task_data["pos_samples"] = pos_samples
                task_data["neg_samples"] = neg_samples
                task_data["annotation"] = task[-1].replace("++", " ")
                self.task_list.append(task_data)

    def __len__(self):
        return len(self.task_list)

    def load_image(self, path, transform_type="base_transform"):
        im_path = os.path.join(self.data_root, path.replace("./", ""))
        if not os.path.isfile(im_path):
            print("file not exist: {}".format(im_path))
            if "/pic/image/val" in im_path:
                im_path = im_path.replace("val", "train")
            elif "/pic/image/train" in im_path:
                im_path = im_path.replace("train", "val")
        try:
            image = Image.open(im_path).convert("RGB")
        except:
            print("File error: ", im_path)
            image = Image.open(im_path).convert("RGB")
        trans = getattr(self, transform_type)
        if trans is not None:
            image = trans(image)
        return image

    def __getitem__(self, idx):
        task = self.task_list[idx]
        pos_samples = task["pos_samples"]
        neg_samples = task["neg_samples"]

        # random.seed(0)
        # random.shuffle(pos_samples)
        # random.shuffle(neg_samples)

        f_pos_support = pos_samples[:-1]
        f_neg_support = neg_samples[:-1]
        pos_images = [self.load_image(f, "base_transform") for f in f_pos_support]
        neg_images = [self.load_image(f, "base_transform") for f in f_neg_support]
        pos_support = torch.stack(pos_images, dim=0)
        neg_support = torch.stack(neg_images, dim=0)

        pos_query = self.load_image(pos_samples[-1], "query_transform")
        neg_query = self.load_image(neg_samples[-1], "query_transform")

        support_images = torch.cat((pos_support, neg_support), dim=0)
        support_labels = torch.Tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).long()
        # query_images = torch.stack([neg_query, pos_query], dim=0).squeeze(1)
        query_images = [neg_query, pos_query]
        query_labels = torch.Tensor([1, 0]).long()

        return support_images, query_images, support_labels, query_labels


DOWNLOAD_ROOT = "~/.cache/clip"


class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        clip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.encoder = clip.visual
        del clip.transformer
        torch.cuda.empty_cache()

        self.cls_head = nn.Linear(embed_dim, n_class)

    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )

        return x


class PromptLearner(nn.Module):
    def __init__(
        self,
        clip_model,
        classnames,
        batch_size=None,
        n_ctx=16,
        ctx_init=None,
        ctx_position="end",
        learned_cls=False,
    ):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if "[CLS]" in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None:
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  # (N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(
                n_cls, 1, ctx_dim, dtype=dtype
            )  # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors)  # to be optimized

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer(
                "token_suffix", embedding[:, 1 + n_ctx + 1 :, :]
            )  # ..., EOS
        else:
            self.register_buffer(
                "token_suffix", embedding[:, 1 + n_ctx :, :]
            )  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors)  # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(
                self.n_cls, 1, self.ctx_dim, dtype=self.dtype
            )  # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None:
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,  # (n_cls, n_ctx, dim)
                        cls,  # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,  # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = (
                    self.split_idx
                )  # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class ClipTestTimeTuning(nn.Module):
    def __init__(
        self,
        device,
        classnames,
        batch_size,
        criterion="cosine",
        arch="ViT-L/14",
        n_ctx=16,
        ctx_init=None,
        ctx_position="end",
        learned_cls=True,
    ):
        super(ClipTestTimeTuning, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(
            clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls
        )
        self.criterion = criterion

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)

    def inference(self, image):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))

        text_features = self.get_text_features()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def contrast_prompt_tuning(self, support_images, support_labels):
        # num_support, c, h, w = support_images.shape

        # support_images = support_images.view(batch_size * num_support, c, h, w)
        # neg_query_images, pos_query_images = torch.split(query_images.view(batch_size * num_query, num_view, c, h, w), 1, dim=0)
        # neg_query_images = neg_query_images.squeeze(0)
        # pos_query_images = pos_query_images.squeeze(0)

        logit_scale = self.logit_scale.exp()

        # Extract features using CLIP
        with torch.no_grad():
            # image_features = self.image_encoder(image.type(self.dtype))

            support_features = self.image_encoder(support_images.type(self.dtype))
            # query_features = self.image_encoder(query_images.type(self.dtype))
            # neg_query_features = self.image_encoder(neg_query_images.type(self.dtype))
            # pos_query_features = self.image_encoder(pos_query_images.type(self.dtype))

            support_features = support_features / support_features.norm(
                dim=-1, keepdim=True
            )
            # query_features = query_features / query_features.norm(dim=-1, keepdim=True)
            # neg_query_features = neg_query_features / neg_query_features.norm(dim=-1, keepdim=True)
            # pos_query_features = pos_query_features / pos_query_features.norm(dim=-1, keepdim=True)

        # text_features = self.clip_model.get_text_features(**text_inputs)
        text_features = self.get_text_features()

        # Calculate logits for support and query images
        support_logits = logit_scale * torch.matmul(support_features, text_features.T)
        # query_logits = logit_scale * torch.matmul(query_features, text_features.T)

        # neg_query_logits = logit_scale * torch.matmul(neg_query_features, text_features.T)
        # pos_query_logits = logit_scale * torch.matmul(pos_query_features, text_features.T)

        # print(support_logits.size(), query_logits.size(), support_labels.size())

        # Reshape support labels to match logits
        support_labels = support_labels.view(-1)

        # Calculate cross-entropy loss
        cross_entropy_loss = F.cross_entropy(support_logits, support_labels)
        # + F.cross_entropy(query_logits, support_labels)

        return cross_entropy_loss

    def inference_bongard(self, query_images):
        # batch_size, num_support, c, h, w = support_images.shape
        # num_query, c, h, w = query_images.shape

        # support_images = support_images.view(batch_size * num_support, c, h, w)
        # query_images = query_images.view(batch_size * num_query, c, h, w)
        neg_query_images, pos_query_images = torch.split(query_images, 1, dim=0)
        # print(neg_query_images.size(), pos_query_images.size())
        logit_scale = self.logit_scale.exp()

        neg_query_features = self.image_encoder(neg_query_images.type(self.dtype))
        pos_query_features = self.image_encoder(pos_query_images.type(self.dtype))
        neg_query_features = neg_query_features / neg_query_features.norm(
            dim=-1, keepdim=True
        )
        pos_query_features = pos_query_features / pos_query_features.norm(
            dim=-1, keepdim=True
        )

        with torch.no_grad():
            text_features = self.get_text_features()
        # query_logits = logit_scale * torch.matmul(query_features, text_features.T)

        neg_query_logits = logit_scale * torch.matmul(
            neg_query_features, text_features.T
        ).mean(dim=0)
        pos_query_logits = logit_scale * torch.matmul(
            pos_query_features, text_features.T
        ).mean(dim=0)

        query_logits = torch.stack([neg_query_logits, pos_query_logits], dim=0)
        return query_logits

    def inference_bongard_single(self, query_images):

        logit_scale = self.logit_scale.exp()

        query_features = self.image_encoder(query_images.type(self.dtype))
        query_features = query_features / query_features.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            text_features = self.get_text_features()
        # query_logits = logit_scale * torch.matmul(query_features, text_features.T)

        query_logits = logit_scale * torch.matmul(query_features, text_features.T).mean(
            dim=0
        )

        return query_logits

    def forward(self, input):
        # if isinstance(input, Tuple):
        if len(input) == 2:
            view_0, view_1 = input
            return self.contrast_prompt_tuning(view_0, view_1)
        elif len(input.size()) == 4:
            return self.inference_bongard_single(input)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input)


def test_time_tuning(model, inputs, optimizer, scaler):
    tta_steps = 64
    for j in range(tta_steps):
        with torch.cuda.amp.autocast():

            cross_entropy_loss = model(inputs)
            loss = cross_entropy_loss  # + avg_entropy(neg_output) + avg_entropy(pos_output)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return
